"""EEG-JEPA proof-of-concept training on synthetic CrossPT-EEG style data."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import build_synthetic_crosspt_loaders
from model.jepa import EEGJEPA, jepa_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEG-JEPA on synthetic CrossPT-EEG style data.")
    parser.add_argument("--task", choices=("all", "coarse", "fine"), default="all")
    parser.add_argument("--fine-group", type=int, default=0)
    parser.add_argument("--n-channels", type=int, default=62)
    parser.add_argument("--seq-len", type=int, default=1000)
    parser.add_argument("--patch-len", type=int, default=50)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--enc-depth", type=int, default=6)
    parser.add_argument("--pred-depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--ema-decay-end", type=float, default=0.999)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--classifier-lr", type=float, default=1.0e-3)
    parser.add_argument("--classifier-weight-decay", type=float, default=0.0)
    parser.add_argument("--num-subjects", type=int, default=16)
    parser.add_argument("--samples-per-subject", type=int, default=480)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/eeg_jepa_poc")
    parser.add_argument("--pretrained-ckpt", default="")
    return parser.parse_args()


def resolve_num_classes(task: str) -> int:
    if task == "all":
        return 80
    if task == "coarse":
        return 40
    if task == "fine":
        return 8
    raise ValueError(f"Unsupported task '{task}'")


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ema_decay_for_step(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return end
    alpha = step / float(total_steps - 1)
    return start + alpha * (end - start)


def freeze_backbone(model: EEGJEPA) -> None:
    for module in (model.patch_embed, model.context_encoder):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
    model.cls_token.requires_grad = False
    model.pos_embed.requires_grad = False


def topk_correct(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    k = min(k, logits.shape[-1])
    indices = logits.topk(k, dim=1).indices
    return int(indices.eq(labels.unsqueeze(1)).any(dim=1).sum().item())


@torch.no_grad()
def evaluate(model: EEGJEPA, loader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_samples = 0

    for inputs, labels, _subjects in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        total_loss += criterion(logits, labels).item()
        total_top1 += topk_correct(logits, labels, 1)
        total_top5 += topk_correct(logits, labels, 5)
        total_samples += labels.numel()

    denom = max(len(loader), 1)
    sample_denom = max(total_samples, 1)
    return total_loss / denom, total_top1 / sample_denom, total_top5 / sample_denom


def main() -> None:
    args = parse_args()
    num_classes = resolve_num_classes(args.task)
    device = resolve_device(args.device)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pretrain_ckpt = output_dir / "pretrain.pt"
    finetune_ckpt = output_dir / "linear_probe.pt"

    _, pretrain_loader, train_loader, test_loader = build_synthetic_crosspt_loaders(
        seq_len=args.seq_len,
        n_channels=args.n_channels,
        num_subjects=args.num_subjects,
        samples_per_subject=args.samples_per_subject,
        batch_size=args.batch_size,
        task=args.task,
        fine_group=args.fine_group,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = EEGJEPA(
        n_channels=args.n_channels,
        seq_len=args.seq_len,
        patch_len=args.patch_len,
        embed_dim=args.embed_dim,
        enc_depth=args.enc_depth,
        pred_depth=args.pred_depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        mask_ratio=args.mask_ratio,
        ema_decay=args.ema_decay,
        num_classes=num_classes,
    ).to(device)

    if args.pretrained_ckpt:
        ckpt = torch.load(args.pretrained_ckpt, map_location=device)
        state = ckpt.get("model_state", ckpt)
        state = dict(state)
        if "classifier.weight" in state and state["classifier.weight"].shape != model.classifier.weight.shape:
            state.pop("classifier.weight", None)
        if "classifier.bias" in state and state["classifier.bias"].shape != model.classifier.bias.shape:
            state.pop("classifier.bias", None)
        model.load_state_dict(state, strict=False)
        model._sync_target_encoder()
        print(f"[load] pretrained checkpoint: {args.pretrained_ckpt}")

    elif args.pretrain_epochs > 0:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.pretrain_epochs))
        total_steps = max(1, args.pretrain_epochs * len(pretrain_loader))
        global_step = 0

        for epoch in range(1, args.pretrain_epochs + 1):
            model.train()
            epoch_loss = 0.0
            for step, (inputs, _labels, _subjects) in enumerate(pretrain_loader, start=1):
                inputs = inputs.to(device)
                pred, target = model.pretrain_forward(inputs)
                loss = jepa_loss(pred, target)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                decay = ema_decay_for_step(args.ema_decay, args.ema_decay_end, global_step, total_steps)
                model.update_target_encoder(ema_decay=decay)

                global_step += 1
                epoch_loss += loss.item()
                if step % args.log_every == 0 or step == len(pretrain_loader):
                    print(
                        f"[pretrain] epoch={epoch}/{args.pretrain_epochs} step={step}/{len(pretrain_loader)} "
                        f"loss={loss.item():.4f} ema={decay:.6f}"
                    )

            scheduler.step()
            print(f"[pretrain] epoch={epoch} mean_loss={epoch_loss / max(1, len(pretrain_loader)):.4f}")

        torch.save({"stage": "pretrain", "model_state": model.state_dict()}, pretrain_ckpt)
        print(f"[save] {pretrain_ckpt}")

    freeze_backbone(model)
    optimizer = Adam(model.classifier.parameters(), lr=args.classifier_lr, weight_decay=args.classifier_weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.finetune_epochs))
    criterion = torch.nn.CrossEntropyLoss()
    best_top1 = -math.inf

    for epoch in range(1, args.finetune_epochs + 1):
        model.classifier.train()
        model.patch_embed.eval()
        model.context_encoder.eval()

        epoch_loss = 0.0
        epoch_top1 = 0
        epoch_top5 = 0
        epoch_samples = 0

        for step, (inputs, labels, _subjects) in enumerate(train_loader, start=1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                tokens = model._embed_patches(inputs)
                all_idx = torch.arange(model.n_patches, device=inputs.device)
                ctx_out = model.encode_context(tokens, all_idx)
                features = ctx_out[:, 0]

            logits = model.classifier(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_top1 += topk_correct(logits, labels, 1)
            epoch_top5 += topk_correct(logits, labels, 5)
            epoch_samples += labels.numel()
            if step % args.log_every == 0 or step == len(train_loader):
                step_top1 = topk_correct(logits, labels, 1) / max(1, labels.numel())
                print(
                    f"[finetune] epoch={epoch}/{args.finetune_epochs} step={step}/{len(train_loader)} "
                    f"loss={loss.item():.4f} top1={step_top1:.4f}"
                )

        scheduler.step()
        train_loss = epoch_loss / max(1, len(train_loader))
        train_top1 = epoch_top1 / max(1, epoch_samples)
        train_top5 = epoch_top5 / max(1, epoch_samples)
        val_loss, val_top1, val_top5 = evaluate(model, test_loader, device)
        print(
            f"[finetune] epoch={epoch} train_loss={train_loss:.4f} train_top1={train_top1:.4f} "
            f"train_top5={train_top5:.4f} val_loss={val_loss:.4f} val_top1={val_top1:.4f} val_top5={val_top5:.4f}"
        )

        if val_top1 > best_top1:
            best_top1 = val_top1
            torch.save({"stage": "linear_probe", "model_state": model.state_dict()}, finetune_ckpt)
            print(f"[save] {finetune_ckpt}")

    test_loss, test_top1, test_top5 = evaluate(model, test_loader, device)
    print(f"[eval] task={args.task} test_loss={test_loss:.4f} top1={test_top1:.4f} top5={test_top5:.4f}")


if __name__ == "__main__":
    main()