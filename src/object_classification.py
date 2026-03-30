import os

import hydra
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset

from dataset import EEGImageNetDataset
from preprocessing.de_feat_cal import de_feat_cal
from model.eegnet import EEGNet
from model.mlp import MLP
from model.rgnn import RGNN, get_edge_weight
from model.simple_model import SimpleModel
from trainer import build_label_map, train_classifier
from utilities import get_device, build_optimizer

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
    raise ValueError(f"Unknown model: {name}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = get_device()

    dataset = EEGImageNetDataset.from_args(cfg)
    eeg_data = np.stack([sample[0].numpy() for sample in dataset], axis=0)
    de_feat = de_feat_cal(eeg_data, cfg.subject, cfg.granularity)
    dataset.add_frequency_feat(de_feat)

    all_labels = np.array([sample[1] for sample in dataset])
    label_map = build_label_map(all_labels)
    train_idx = np.array([i for i in range(len(dataset)) if i % 50 < 30])
    test_idx = np.array([i for i in range(len(dataset)) if i % 50 >= 30])
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)

    is_simple = cfg.model.type == "simple"
    model_obj = model_init(cfg, len(dataset) // 50, device)

    if cfg.pretrained_model:
        model_obj.load_state_dict(
            torch.load(os.path.join(cfg.output_dir, cfg.pretrained_model), map_location="cpu")
        )

    if is_simple:
        model_obj.fit(de_feat[train_idx], all_labels[train_idx])
        acc = accuracy_score(all_labels[test_idx], model_obj.predict(de_feat[test_idx]))
        with open(os.path.join(cfg.output_dir, "simple.txt"), "a", encoding="utf-8") as f:
            f.write(f"{acc}\n")
    else:
        dataset.use_frequency_feat = cfg.model.use_freq
        train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=cfg.batch_size, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = build_optimizer(model_obj.parameters(), cfg.model.optimizer)
        save_path = os.path.join(cfg.output_dir, f"{cfg.model.name}_s{cfg.subject}.pth")
        acc, epoch = train_classifier(
            model_obj, train_loader, test_loader, criterion, optimizer,
            cfg.model.epochs, device, label_map, save_path=save_path,
        )
        with open(os.path.join(cfg.output_dir, f"{cfg.model.name}.txt"), "a", encoding="utf-8") as f:
            f.write(f"{epoch}: {acc}\n")


if __name__ == "__main__":
    main()
