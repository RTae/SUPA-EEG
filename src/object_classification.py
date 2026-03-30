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
        for ep in range(cfg.model.pretrain_epochs):
            model_obj.train()
            total_loss = 0.0
            for inputs, _ in pretrain_loader:
                inputs = inputs.to(device)
                pred, target = model_obj.pretrain_forward(inputs)
                loss = jepa_loss(pred, target)
                pt_optimizer.zero_grad()
                loss.backward()
                pt_optimizer.step()
                model_obj.update_target_encoder()
                total_loss += loss.item()
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
