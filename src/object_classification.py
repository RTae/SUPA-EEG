import os
import json

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset
from loguru import logger

from dataset import EEGImageNetDataset
from model.eegnet import EEGNet
from model.mlp import MLP
from model.rgnn import RGNN, get_edge_weight
from model.semantic import SemanticModel
from model.simple_model import SimpleModel
from preprocessing.de_feat_cal import de_feat_cal
from trainer import build_label_map, train_classifier, train_semantic_classifier
from utilities import build_optimizer, get_benchmark_split, get_device


def _is_semantic_model(model_name: str) -> bool:
    return model_name.lower() == "semantic"


def _load_semantic_neighbors(
    cfg: DictConfig,
    dataset: EEGImageNetDataset,
    label_map: dict[int, int],
    device: torch.device,
) -> dict[int, set[int]] | None:
    default_knn_path = os.path.join(cfg.dataset_dir, f"semantic_knn_{cfg.granularity}.json")
    semantic_knn_path = cfg.model.get("semantic_knn_path", default_knn_path)

    if not semantic_knn_path:
        return None

    if not os.path.exists(semantic_knn_path):
        logger.info(f"[semantic] cache not found at {semantic_knn_path}, building it now...")
        try:
            from preprocessing.semantic_knn import build_semantic_knn

            build_semantic_knn(
                dataset=dataset,
                output_path=semantic_knn_path,
                sd_model=cfg.diffusion.sd_model,
                k_pos=int(cfg.model.get("semantic_knn_k", 4)),
                k_neg=int(cfg.model.get("semantic_neg_k", 4)),
                device=device,
            )
        except Exception as exc:
            logger.error(f"[semantic] failed to build semantic cache: {exc}")
            return None

    with open(semantic_knn_path, encoding="utf-8") as f:
        knn_obj = json.load(f)
    records = knn_obj.get("records", {})
    semantic_neighbors: dict[int, set[int]] = {}
    for orig_str, rec in records.items():
        orig_id = int(orig_str)
        if orig_id not in label_map:
            continue
        remapped = label_map[orig_id]
        neigh = set()
        for n in rec.get("positives", []):
            n = int(n)
            if n in label_map:
                neigh.add(label_map[n])
        if neigh:
            semantic_neighbors[remapped] = neigh
    return semantic_neighbors or None

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
    if _is_semantic_model(name):
        return SemanticModel(cfg, num_classes)
    raise ValueError(f"Unknown model: {name}")


def load_data(cfg: DictConfig, device: torch.device) -> dict:
    dataset = EEGImageNetDataset(
        dataset_dir=cfg.dataset_dir,
        subject=-1,
        granularity=cfg.granularity,
        map_location=device,
    )
    eeg_data = np.stack([sample[0].numpy() for sample in dataset], axis=0)
    de_feat = de_feat_cal(eeg_data, -1, cfg.granularity)
    dataset.add_frequency_feat(de_feat)

    all_labels = np.array([sample[1] for sample in dataset])
    train_idx, test_idx = get_benchmark_split(dataset.data, cfg.metric, cfg.subject)
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    combined_labels = np.concatenate([all_labels[train_idx], all_labels[test_idx]])
    label_map = build_label_map(combined_labels)

    semantic_neighbors = None
    if _is_semantic_model(cfg.model.name):
        semantic_neighbors = _load_semantic_neighbors(cfg, dataset, label_map, device)

    return {
        "num_classes": len(label_map),
        "label_map": label_map,
        "train_subset": Subset(dataset, train_idx),
        "test_subset": Subset(dataset, test_idx),
        "all_labels": all_labels,
        "de_feat": de_feat,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "is_simple": cfg.model.type == "simple",
        "dataset": dataset,
        "semantic_neighbors": semantic_neighbors,
    }


def _train_semantic_model(
    cfg: DictConfig,
    device: torch.device,
    model_obj: SemanticModel,
    data: dict,
    run_dir: str,
) -> dict:
    data["dataset"].use_frequency_feat = False
    train_loader = DataLoader(data["train_subset"], batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(data["test_subset"], batch_size=cfg.batch_size, shuffle=False)

    optimizer = build_optimizer(model_obj.parameters(), cfg.model.optimizer)
    save_path = os.path.join(run_dir, f"{cfg.model.name}_s{cfg.subject}.pth")
    best_top1, best_top5, best_epoch = train_semantic_classifier(
        model_obj,
        train_loader,
        test_loader,
        optimizer,
        cfg.model.epochs,
        device,
        data["label_map"],
        triplet_margin=float(cfg.model.get("triplet_margin", 0.2)),
        ema_decay=float(cfg.model.get("ema_decay", 0.996)),
        semantic_neighbors=data.get("semantic_neighbors"),
        save_path=save_path,
    )

    with open(os.path.join(run_dir, "result.txt"), "a", encoding="utf-8") as f:
        f.write(f"top1={best_top1:.4f}\n")
        f.write(f"top5={best_top5:.4f}\n")
        f.write(f"best_epoch={best_epoch}\n")

    return {"semantic_top1": best_top1, "semantic_top5": best_top5, "semantic_epoch": best_epoch}


def _train_simple_model(model_obj: SimpleModel, data: dict, run_dir: str) -> dict:
    data["dataset"].use_frequency_feat = True
    x_train = data["de_feat"][data["train_idx"]].reshape(len(data["train_idx"]), -1)
    x_test = data["de_feat"][data["test_idx"]].reshape(len(data["test_idx"]), -1)

    y_train = np.array([data["label_map"][int(v)] for v in data["all_labels"][data["train_idx"]]])
    y_test = np.array([data["label_map"][int(v)] for v in data["all_labels"][data["test_idx"]]])

    model_obj.fit(x_train, y_train)
    pred = model_obj.predict(x_test)
    top1 = float(accuracy_score(y_test, pred))

    with open(os.path.join(run_dir, "result.txt"), "a", encoding="utf-8") as f:
        f.write(f"top1={top1:.4f}\n")
    logger.info(f"[simple] test_top1={top1:.4f}")
    return {"simple_top1": top1}

def train_model(cfg: DictConfig, device: torch.device, model_obj: object, data: dict, run_dir: str) -> dict:
    if _is_semantic_model(cfg.model.name):
        return _train_semantic_model(cfg, device, model_obj, data, run_dir)
    if data.get("is_simple", False):
        return _train_simple_model(model_obj, data, run_dir)

    data["dataset"].use_frequency_feat = cfg.model.use_freq
    train_loader = DataLoader(data["train_subset"], batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(data["test_subset"], batch_size=cfg.batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(model_obj.parameters(), cfg.model.optimizer)
    save_path = os.path.join(run_dir, f"{cfg.model.name}_s{cfg.subject}.pth")
    acc, epoch = train_classifier(
        model_obj,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        cfg.model.epochs,
        device,
        data["label_map"],
        save_path=save_path,
    )
    return {"deep_top1": acc, "deep_epoch": epoch}


def evaluate_model(cfg: DictConfig, train_results: dict) -> None:
    model_name = cfg.model.name.lower()
    if _is_semantic_model(model_name):
        logger.info(
            f"[semantic] best_top1={train_results['semantic_top1']:.4f} "
            f"best_top5={train_results['semantic_top5']:.4f} epoch={train_results['semantic_epoch']}"
        )
        return
    if cfg.model.type == "simple":
        logger.info(f"[simple] best_top1={train_results['simple_top1']:.4f}")
        return
    logger.info(f"[deep] best_top1={train_results['deep_top1']:.4f} epoch={train_results['deep_epoch']}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))
    device = get_device()

    data = load_data(cfg, device)
    model_obj = model_init(cfg, data["num_classes"], device)
    run_dir = HydraConfig.get().runtime.output_dir
    train_results = train_model(cfg, device, model_obj, data, run_dir)
    evaluate_model(cfg, train_results)


if __name__ == "__main__":
    main()
