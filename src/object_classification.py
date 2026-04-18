import os

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset
from loguru import logger

from dataset import EEGImageNetDataset, BalancedBatchSampler
from model.eegnet import EEGNet
from model.mlp import MLP
from model.semantic import SemanticModel
from model.simple_model import SimpleModel
from preprocessing.de_feat_cal import de_feat_cal
from trainer import train_classifier, train_semantic_classifier
from utilities import build_optimizer, get_benchmark_split, get_device


def _is_semantic_model(model_name: str) -> bool:
    return model_name.lower() == "semantic"

def _model_init(cfg: DictConfig, num_classes: int, device: torch.device) -> object:
    name = cfg.model.name.lower()
    if cfg.model.type == "simple":
        model_params = OmegaConf.to_container(cfg.model.params, resolve=True)
        return SimpleModel(name, **model_params)
    if name == "eegnet":
        return EEGNet(cfg, num_classes)
    if name == "mlp":
        return MLP(cfg, num_classes)
    if _is_semantic_model(name):
        return SemanticModel(cfg, num_classes)
    raise ValueError(f"Unknown model: {name}")

def _prep_freq_features(dataset: EEGImageNetDataset) -> None:
    logger.info("Calculating frequency-domain features for the dataset...")

    eeg_samples = []
    for feat, _ in dataset:
        if torch.is_tensor(feat):
            eeg_samples.append(feat.detach().cpu().numpy())
        else:
            eeg_samples.append(np.asarray(feat))

    eeg_data = np.stack(eeg_samples, axis=0)
    de_feat = de_feat_cal(eeg_data, dataset.subject, dataset.granularity)
    dataset.add_frequency_feat(de_feat)
    
def _prep_time_features(dataset: EEGImageNetDataset) -> None:
    logger.info("Using raw time-domain features for the dataset.")
    # No additional preparation needed for time-domain features, but this function is here for consistency and future extensibility.
    pass

def load_data(cfg: DictConfig, device: torch.device) -> dict:
    dataset = EEGImageNetDataset(
        dataset_dir=cfg.dataset_dir,
        subject=int(cfg.get("subject", -1)),
        granularity=cfg.granularity,
        map_location=device,
    )
    model_feature_type = str(cfg.model.get("feature_type", "time")).lower()
    
    # Add frequency features to the dataset if required by the model configuration
    if model_feature_type == "freq":
        _prep_freq_features(dataset)
    
    # Add time-domain features to the dataset if required by the model configuration
    if model_feature_type == "time":
        _prep_time_features(dataset)
        
    if model_feature_type not in ["time", "freq"]:
        raise ValueError(f"Unsupported feature type: {model_feature_type}")

    train_idx, test_idx = get_benchmark_split(dataset.data, cfg.metric)
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)

    train_loader = None
    eval_loader = None
    if _is_semantic_model(cfg.model.name):
        samples_per_class = int(cfg.model.get("samples_per_class", 4))
        num_classes_per_batch = max(2, cfg.batch_size // samples_per_class)
        balanced_sampler = BalancedBatchSampler(
            train_subset,
            num_classes_per_batch=num_classes_per_batch,
            samples_per_class=samples_per_class,
        )
        train_loader = DataLoader(train_subset, batch_sampler=balanced_sampler)
        eval_loader = DataLoader(test_subset, batch_size=cfg.batch_size, shuffle=False)

    return {
        "num_classes": len(dataset.label_to_index),
        "train_subset": train_subset,
        "test_subset": test_subset,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "is_simple": cfg.model.type == "simple",
        "dataset": dataset,
        "train_loader": train_loader,
        "eval_loader": eval_loader,
    }


def _train_semantic_model(
    cfg: DictConfig,
    device: torch.device,
    model_obj: SemanticModel,
    data: dict,
    run_dir: str,
) -> dict:
    optimizer = build_optimizer(model_obj.parameters(), cfg.model.optimizer)
    save_path = os.path.join(run_dir, f"{cfg.model.name}.pth")
    best_top1, best_top5, best_epoch = train_semantic_classifier(
        model_obj,
        data["train_loader"],
        data["eval_loader"],
        optimizer,
        cfg.model.epochs,
        device,
        triplet_margin=float(cfg.model.get("triplet_margin", 0.2)),
        ema_decay=float(cfg.model.get("ema_decay", 0.996)),
        save_path=save_path,
    )

    with open(os.path.join(run_dir, "result.txt"), "a", encoding="utf-8") as f:
        f.write(f"top1={best_top1:.4f}\n")
        f.write(f"top5={best_top5:.4f}\n")
        f.write(f"best_epoch={best_epoch}\n")

    return {"top1": best_top1, "top5": best_top5, "epoch": best_epoch}

def _train_simple_model(model_obj: SimpleModel, data: dict, run_dir: str) -> dict:
    freq_feat = data["dataset"].frequency_feat
    if freq_feat is None:
        raise ValueError("Frequency features are required for this model but were not prepared in load_data().")

    x_train = freq_feat[data["train_idx"]].reshape(len(data["train_idx"]), -1)
    x_test = freq_feat[data["test_idx"]].reshape(len(data["test_idx"]), -1)
    all_labels = np.array([sample[1] for sample in data["dataset"]])
    y_train = all_labels[data["train_idx"]]
    y_test = all_labels[data["test_idx"]]

    model_obj.fit(x_train, y_train)
    pred = model_obj.predict(x_test)
    top1 = float(accuracy_score(y_test, pred))

    with open(os.path.join(run_dir, "result.txt"), "a", encoding="utf-8") as f:
        f.write(f"top1={top1:.4f}\n")
        
    return {"top1": top1}

def _train_nn_model(
    cfg: DictConfig,
    model_obj: torch.nn.Module,
    data: dict,
    device: torch.device,
    run_dir: str,
):
    train_loader = DataLoader(data["train_subset"], batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(data["test_subset"], batch_size=cfg.batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(model_obj.parameters(), cfg.model.optimizer)
    save_path = os.path.join(run_dir, f"{cfg.model.name}.pth")
    best_top1, best_top5, best_epoch = train_classifier(
        model_obj,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        cfg.model.epochs,
        device,
        save_path=save_path,
    )
    with open(os.path.join(run_dir, "result.txt"), "a", encoding="utf-8") as f:
        f.write(f"top1={best_top1:.4f}\n")
        f.write(f"top5={best_top5:.4f}\n")
        f.write(f"epoch={best_epoch}\n")

    return {"top1": best_top1, "top5": best_top5, "epoch": best_epoch}

def train_model(cfg: DictConfig, device: torch.device, model_obj: object, data: dict, run_dir: str) -> dict:
    if _is_semantic_model(cfg.model.name):
        return _train_semantic_model(cfg, device, model_obj, data, run_dir)
    if data.get("is_simple", False):
        return _train_simple_model(model_obj, data, run_dir)
    return _train_nn_model(cfg, model_obj, data, device, run_dir)

def evaluate_model(cfg: DictConfig, train_results: dict) -> None:
    model_name = cfg.model.name.lower()
    if _is_semantic_model(model_name):
        logger.info(
            f"[semantic] best_top1={train_results['top1']:.4f} "
            f"best_top5={train_results['top5']:.4f} epoch={train_results['epoch']}"
        )
        return
    if cfg.model.type == "simple":
        logger.info(f"[simple] best_top1={train_results['top1']:.4f}")
        return
    logger.info(
        f"[deep] best_top1={train_results['top1']:.4f} "
        f"best_top5={train_results['top5']:.4f} epoch={train_results['epoch']}"
    )


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
    device = get_device()

    data = load_data(cfg, device)
    model_obj = _model_init(cfg, data["num_classes"], device)
    run_dir = HydraConfig.get().runtime.output_dir
    train_results = train_model(cfg, device, model_obj, data, run_dir)
    evaluate_model(cfg, train_results)


if __name__ == "__main__":
    main()
