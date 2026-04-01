import os

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset

from dataset import EEGImageNetDataset
from preprocessing.de_feat_cal import de_feat_cal
from model.eegnet import EEGNet
from model.jepa import EEGJEPA, jepa_loss
from model.mlp import MLP
from model.rgnn import RGNN, get_edge_weight
from model.simple_model import SimpleModel
from trainer import build_label_map, train_classifier
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
    if name == "jepa":
        return EEGJEPA(
            n_channels=62,
            seq_len=400,
            patch_len=cfg.model.patch_len,
            embed_dim=cfg.model.embed_dim,
            enc_depth=cfg.model.enc_depth,
            pred_depth=cfg.model.pred_depth,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            dropout=cfg.model.dropout,
            mask_ratio=cfg.model.mask_ratio,
            ema_decay=cfg.model.ema_decay,
            num_classes=num_classes,
        )
    raise ValueError(f"Unknown model: {name}")


def _set_jepa_linear_probe(model: EEGJEPA, enabled: bool) -> None:
    """Freeze JEPA backbone for linear probe when enabled."""
    backbone_modules = [model.patch_embed, model.context_encoder]
    if enabled:
        for module in backbone_modules:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
        model.cls_token.requires_grad = False
        model.pos_embed.requires_grad = False
    else:
        for module in backbone_modules:
            for p in module.parameters():
                p.requires_grad = True
        model.cls_token.requires_grad = True
        model.pos_embed.requires_grad = True


def _jepa_ema_decay(cfg: DictConfig, step: int, total_steps: int) -> float:
    """Linear warmup of EMA decay from ema_decay to ema_decay_end."""
    start = float(cfg.model.ema_decay)
    end = float(cfg.model.get("ema_decay_end", start))
    if total_steps <= 1:
        return end
    alpha = step / float(total_steps - 1)
    return start + alpha * (end - start)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = get_device()

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
        model_obj.load_state_dict(
            torch.load(os.path.join(cfg.output_dir, cfg.pretrained_model), map_location="cpu")
        )

    run_dir = HydraConfig.get().runtime.output_dir

    # ----- JEPA pre-training phase (self-supervised) -----
    if cfg.model.name.lower() == "jepa" and cfg.model.get("pretrain_epochs", 0) > 0 and not cfg.pretrained_model:
        print("=== JEPA pre-training ===")
        model_obj.to(device)
        dataset.use_frequency_feat = False
        pretrain_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
        pt_optimizer = build_optimizer(model_obj.parameters(), cfg.model.optimizer)
        total_steps = max(1, cfg.model.pretrain_epochs * len(pretrain_loader))
        global_step = 0
        log_interval = max(1, int(cfg.model.get("pretrain_log_interval", 10)))
        for ep in range(cfg.model.pretrain_epochs):
            model_obj.train()
            total_loss = 0.0
            for step, (inputs, _) in enumerate(pretrain_loader, start=1):
                inputs = inputs.to(device)
                pred, target = model_obj.pretrain_forward(inputs)
                loss = jepa_loss(pred, target)
                pt_optimizer.zero_grad()
                loss.backward()
                pt_optimizer.step()
                decay = _jepa_ema_decay(cfg, global_step, total_steps)
                model_obj.update_target_encoder(ema_decay=decay)
                total_loss += loss.item()
                global_step += 1
                if step % log_interval == 0 or step == len(pretrain_loader):
                    print(f"    step {step}/{len(pretrain_loader)} loss={loss.item():.4f} ema={decay:.6f}")
            avg = total_loss / len(pretrain_loader)
            if ep % 10 == 0:
                print(f"  pretrain epoch {ep}: loss={avg:.4f}")
        pt_path = os.path.join(run_dir, f"jepa_pretrained_s{cfg.subject}.pth")
        os.makedirs(os.path.dirname(pt_path), exist_ok=True)
        torch.save(model_obj.state_dict(), pt_path)
        print(f"  saved pretrained model → {pt_path}")

    if is_simple:
        model_obj.fit(de_feat[train_idx], all_labels[train_idx])
        acc = accuracy_score(all_labels[test_idx], model_obj.predict(de_feat[test_idx]))
        with open(os.path.join(run_dir, "result.txt"), "a", encoding="utf-8") as f:
            f.write(f"{acc}\n")
    else:
        if cfg.model.name.lower() == "jepa":
            linear_probe = bool(cfg.model.get("linear_probe", False))
            _set_jepa_linear_probe(model_obj, linear_probe)
            if linear_probe:
                print("=== JEPA fine-tuning mode: linear probe ===")
            else:
                print("=== JEPA fine-tuning mode: end-to-end ===")

        dataset.use_frequency_feat = cfg.model.use_freq
        train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=cfg.batch_size, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss()
        if cfg.model.name.lower() == "jepa" and bool(cfg.model.get("linear_probe", False)):
            cls_opt_cfg = cfg.model.get("classifier_optimizer", cfg.model.optimizer)
            optimizer = build_optimizer(model_obj.classifier.parameters(), cls_opt_cfg)
        else:
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
