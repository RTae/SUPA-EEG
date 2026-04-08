import math
import os

import hydra
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset, TensorDataset

from tqdm import tqdm

from dataset import EEGImageNetDataset, build_synthetic_crosspt_loaders
from preprocessing.de_feat_cal import de_feat_cal
from model.eegnet import EEGNet
from model.jepa import (
    EEGJEPA,
    jepa_loss,
    ema_decay_schedule,
    jepa_evaluate,
    load_jepa_checkpoint,
    extract_all_features,
    build_jepa_downstream,
)
from model.eeg_transformer import EEGTransformer
from model.llm_encoder import LLMEEGEncoder
from model.mlp import MLP
from model.rgnn import RGNN, get_edge_weight
from model.simple_model import SimpleModel
from trainer import build_label_map, train_classifier, topk_correct
from utilities import get_device, build_optimizer, get_benchmark_split

SIMPLE_MODELS = {"svm", "rf", "knn", "dt", "ridge"}


def model_init(cfg: DictConfig, num_classes: int, device: torch.device) -> object:
    name = cfg.model.name.lower()
    if cfg.model.type == "simple":
        model_params = OmegaConf.to_container(cfg.model.params, resolve=True)
        return SimpleModel(name, **model_params)
    if name == "eegnet":
        return EEGNet(cfg, num_classes)
    if name == "mlp":
        return MLP(cfg, num_classes)
    if name == "rgnn":
        edge_index, edge_weight = get_edge_weight()
        return RGNN(device, 62, edge_weight, edge_index, 5, 200, num_classes, 2)
    if name == "eeg_transformer":
        return EEGTransformer(cfg)
    if name == "llm_encoder":
        return LLMEEGEncoder(cfg)
    if name == "jepa":
        return EEGJEPA(
            n_channels=int(cfg.model.get("n_channels", 62)),
            seq_len=int(cfg.model.get("seq_len", 400)),
            patch_len=cfg.model.patch_len,
            embed_dim=cfg.model.embed_dim,
            enc_depth=cfg.model.enc_depth,
            pred_depth=cfg.model.pred_depth,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            dropout=cfg.model.dropout,
            mask_ratio=cfg.model.mask_ratio,
            ema_decay=cfg.model.ema_decay,
        )
    raise ValueError(f"Unknown model: {name}")



@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = get_device()

    is_jepa = cfg.model.name.lower() == "jepa"
    is_transformer = cfg.model.name.lower() in {"eeg_transformer", "llm_encoder"}
    use_synthetic = is_jepa and bool(cfg.get("synthetic", False))

    if use_synthetic:
        _task = str(cfg.granularity)
        _fine_group = int(cfg.get("fine_group", 0))
        num_classes = {"all": 80, "coarse": 40, "fine": 8}[_task]
        _, pretrain_loader_syn, train_loader_syn, test_loader_syn = build_synthetic_crosspt_loaders(
            seq_len=int(cfg.model.get("seq_len", 1000)),
            n_channels=int(cfg.model.get("n_channels", 62)),
            num_subjects=int(cfg.get("num_subjects", 16)),
            samples_per_subject=int(cfg.get("samples_per_subject", 480)),
            batch_size=cfg.batch_size,
            task=_task,
            fine_group=_fine_group,
            num_workers=int(cfg.get("num_workers", 0)),
            seed=int(cfg.get("seed", 42)),
        )
        label_map = None
        train_subset = test_subset = None
        all_labels = de_feat = train_idx = test_idx = None
        is_simple = False
    else:
        # Load all subjects so benchmark splits can select across subjects/stages.
        dataset = EEGImageNetDataset(
            dataset_dir=cfg.dataset_dir,
            subject=-1,
            granularity=cfg.granularity,
        )
        eeg_data = np.stack([sample[0].numpy() for sample in dataset], axis=0)
        de_feat = de_feat_cal(eeg_data, -1, cfg.granularity)
        dataset.add_frequency_feat(de_feat)

        all_labels = np.array([sample[1] for sample in dataset])

        train_idx, test_idx = get_benchmark_split(dataset.data, cfg.metric, cfg.subject)
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)

        # Build label map from the union of train+test labels so both loaders
        # use a consistent mapping to contiguous indices.
        combined_labels = np.concatenate([all_labels[train_idx], all_labels[test_idx]])
        label_map = build_label_map(combined_labels)
        num_classes = len(label_map)

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        is_simple = cfg.model.type == "simple"

    model_obj = model_init(cfg, num_classes, device)

    if cfg.pretrained_model:
        ckpt_path = os.path.join(cfg.output_dir, cfg.pretrained_model)
        if is_jepa:
            load_jepa_checkpoint(model_obj, ckpt_path, device)
        else:
            model_obj.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)

    run_dir = HydraConfig.get().runtime.output_dir

    # ----- JEPA pre-training phase (self-supervised) -----
    if is_jepa and cfg.model.get("pretrain_epochs", 0) > 0 and not cfg.pretrained_model:
        print("=== JEPA pre-training ===")
        model_obj.to(device)
        if use_synthetic:
            pretrain_loader = pretrain_loader_syn
        else:
            dataset.use_frequency_feat = False
            pretrain_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        weight_decay = float(cfg.model.get("weight_decay", 0.05))
        pt_optimizer = AdamW(
            model_obj.parameters(),
            lr=float(cfg.model.optimizer.lr),
            weight_decay=weight_decay,
        )
        pt_scheduler = CosineAnnealingLR(pt_optimizer, T_max=max(1, cfg.model.pretrain_epochs))
        total_steps = max(1, cfg.model.pretrain_epochs * len(pretrain_loader))
        global_step = 0
        ema_start = float(cfg.model.ema_decay)
        ema_end = float(cfg.model.get("ema_decay_end", ema_start))
        epoch_bar = tqdm(range(1, cfg.model.pretrain_epochs + 1), desc="pretrain", unit="ep")
        for ep in epoch_bar:
            model_obj.train()
            epoch_loss = 0.0
            step_bar = tqdm(pretrain_loader, desc=f"ep {ep}", leave=False, unit="step")
            for batch in step_bar:
                inputs = batch[0].to(device)
                pred, target = model_obj.pretrain_forward(inputs)
                loss = jepa_loss(pred, target)
                pt_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                pt_optimizer.step()
                decay = ema_decay_schedule(ema_start, ema_end, global_step, total_steps)
                model_obj.update_target_encoder(ema_decay=decay)
                epoch_loss += loss.item()
                global_step += 1
                step_bar.set_postfix(loss=f"{loss.item():.4f}", ema=f"{decay:.6f}")
            pt_scheduler.step()
            avg = epoch_loss / max(1, len(pretrain_loader))
            epoch_bar.set_postfix(loss=f"{avg:.4f}")
        pt_path = os.path.join(run_dir, f"jepa_pretrained_s{cfg.subject}.pth")
        torch.save({"stage": "pretrain", "model_state": model_obj.state_dict()}, pt_path)
        print(f"  saved pretrained model → {pt_path}")

    if is_simple:
        model_obj.fit(de_feat[train_idx], all_labels[train_idx])
        acc = accuracy_score(all_labels[test_idx], model_obj.predict(de_feat[test_idx]))
        with open(os.path.join(run_dir, "result.txt"), "a", encoding="utf-8") as f:
            f.write(f"{acc}\n")
    elif is_jepa or is_transformer:
        # ----- Decoupled pipeline: encoder → latent space → downstream head -----
        model_obj.to(device)
        linear_probe = bool(cfg.model.get("linear_probe", True))
        downstream_cfg = cfg.model.get("downstream", {"type": "linear"})
        downstream_head = build_jepa_downstream(downstream_cfg, model_obj.embed_dim, num_classes, device)

        # Build raw loaders (real or synthetic)
        if use_synthetic:
            raw_train_loader = train_loader_syn
            raw_test_loader = test_loader_syn
            lmap = None
        else:
            dataset.use_frequency_feat = cfg.model.use_freq
            raw_train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
            raw_test_loader = DataLoader(test_subset, batch_size=cfg.batch_size, shuffle=False)
            lmap = label_map

        if linear_probe:
            # Pre-extract all features once; head trains on TensorDataset (fast)
            print(f"=== {cfg.model.name} fine-tuning: linear probe — extracting features... ===")
            model_obj.freeze_backbone()
            train_feats, train_labels = extract_all_features(model_obj, raw_train_loader, device, lmap)
            test_feats, test_labels = extract_all_features(model_obj, raw_test_loader, device, lmap)
            ft_train_loader = DataLoader(
                TensorDataset(train_feats, train_labels), batch_size=cfg.batch_size, shuffle=True
            )
            ft_test_loader = DataLoader(
                TensorDataset(test_feats, test_labels), batch_size=cfg.batch_size, shuffle=False
            )
            eval_encoder = None   # inputs to jepa_evaluate are already features
            eval_lmap = None      # labels already remapped during extraction
            ft_params = downstream_head.parameters()
        else:
            # End-to-end: backbone stays in the training loop
            print(f"=== {cfg.model.name} fine-tuning: end-to-end ===")
            model_obj.unfreeze_backbone()
            ft_train_loader = raw_train_loader
            ft_test_loader = raw_test_loader
            eval_encoder = model_obj
            eval_lmap = lmap
            ft_params = list(downstream_head.parameters()) + list(model_obj.parameters())

        cls_opt_cfg = cfg.model.get("classifier_optimizer", cfg.model.optimizer)
        ft_optimizer = AdamW(
            ft_params,
            lr=float(cls_opt_cfg.lr),
            weight_decay=float(cls_opt_cfg.get("weight_decay", 0.0)),
        )
        ft_scheduler = CosineAnnealingLR(ft_optimizer, T_max=max(1, cfg.model.epochs))
        criterion = torch.nn.CrossEntropyLoss()
        save_path = os.path.join(run_dir, f"{cfg.model.name}_s{cfg.subject}.pth")
        best_top1 = -math.inf

        epoch_bar = tqdm(range(1, cfg.model.epochs + 1), desc="finetune", unit="ep")
        for epoch in epoch_bar:
            downstream_head.train()
            if linear_probe:
                model_obj.eval()
            else:
                model_obj.train()
            epoch_loss = epoch_top1 = epoch_top5 = epoch_n = 0
            step_bar = tqdm(ft_train_loader, desc=f"ep {epoch}", leave=False, unit="step")
            for batch in step_bar:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                if not linear_probe and lmap is not None:
                    from trainer.metrics import remap_labels
                    labels = remap_labels(labels.cpu(), lmap).to(device)
                feats = model_obj(inputs) if not linear_probe else inputs
                logits = downstream_head(feats)
                loss = criterion(logits, labels)
                ft_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                ft_optimizer.step()
                batch_top1 = topk_correct(logits.detach(), labels, 1)
                epoch_loss += loss.item()
                epoch_top1 += batch_top1
                epoch_top5 += topk_correct(logits.detach(), labels, 5)
                epoch_n += labels.numel()
                step_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    top1=f"{batch_top1 / max(1, labels.numel()):.3f}",
                )
            ft_scheduler.step()
            val_loss, val_top1, val_top5 = jepa_evaluate(
                eval_encoder, downstream_head, ft_test_loader, device, eval_lmap
            )
            epoch_bar.set_postfix(
                tr_loss=f"{epoch_loss / max(1, len(ft_train_loader)):.4f}",
                tr_top1=f"{epoch_top1 / max(1, epoch_n):.3f}",
                val_top1=f"{val_top1:.3f}",
                val_top5=f"{val_top5:.3f}",
            )
            if val_top1 > best_top1:
                best_top1 = val_top1
                torch.save(
                    {
                        "stage": "finetune",
                        "encoder_state": model_obj.state_dict(),
                        "head_state": downstream_head.state_dict(),
                    },
                    save_path,
                )

        test_loss, test_top1, test_top5 = jepa_evaluate(
            eval_encoder, downstream_head, ft_test_loader, device, eval_lmap
        )
        print(f"[eval] test_loss={test_loss:.4f} top1={test_top1:.4f} top5={test_top5:.4f}")
        with open(os.path.join(run_dir, "result.txt"), "a", encoding="utf-8") as f:
            f.write(f"top1={best_top1:.4f}\n")
    else:
        # Other deep models (EEGNet, MLP, RGNN)
        dataset.use_frequency_feat = cfg.model.use_freq
        train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=cfg.batch_size, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = build_optimizer(model_obj.parameters(), cfg.model.optimizer)
        save_path = os.path.join(run_dir, f"{cfg.model.name}_s{cfg.subject}.pth")
        acc, epoch = train_classifier(
            model_obj, train_loader, test_loader, criterion, optimizer,
            cfg.model.epochs, device, label_map, save_path=save_path,
        )
        with open(os.path.join(run_dir, "result.txt"), "a", encoding="utf-8") as f:
            f.write(f"{epoch}: {acc}\n")


if __name__ == "__main__":
    main()
