import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import EEGImageNetDataset
from model.eegnet import EEGNet
from model.mlp import MLP
from model.rgnn import RGNN, get_edge_weight
from model.semantic_triplet import SemanticJEPATripletModel
from model.simple_model import SimpleModel
from preprocessing.de_feat_cal import de_feat_cal
from trainer import build_label_map, remap_labels, topk_correct, train_classifier
from utilities import build_optimizer, get_benchmark_split, get_device

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
    if name == "semantic_triplet":
        return SemanticJEPATripletModel(cfg, num_classes)
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
    }


def _batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    if embeddings.shape[0] < 2:
        return embeddings.new_tensor(0.0)

    dist_mat = torch.cdist(embeddings, embeddings, p=2)
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)

    pos_mask = same_label & ~eye
    neg_mask = ~same_label

    has_pos = pos_mask.any(dim=1)
    has_neg = neg_mask.any(dim=1)
    valid = has_pos & has_neg
    if not valid.any():
        return embeddings.new_tensor(0.0)

    hardest_pos = (dist_mat * pos_mask.float()).max(dim=1).values
    max_dist = dist_mat.max().detach() + 1.0
    hardest_neg = dist_mat.masked_fill(~neg_mask, max_dist).min(dim=1).values
    loss = F.relu(hardest_pos - hardest_neg + margin)
    return loss[valid].mean()


@torch.no_grad()
def _evaluate_semantic(
    model_obj: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    label_map: dict[int, int],
) -> tuple[float, float, float]:
    model_obj.eval()
    total = 0
    top1_correct = 0
    top5_correct = 0
    total_loss = 0.0

    for inputs, labels in dataloader:
        labels = remap_labels(labels, label_map)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_obj(inputs)
        logits = outputs["logits"]
        total_loss += criterion(logits, labels).item()
        total += labels.shape[0]
        top1_correct += topk_correct(logits, labels, 1)
        top5_correct += topk_correct(logits, labels, 5)

    denom = max(total, 1)
    return top1_correct / denom, top5_correct / denom, total_loss / max(len(dataloader), 1)

def _train_semantic_model(
    cfg: DictConfig,
    device: torch.device,
    model_obj: SemanticJEPATripletModel,
    data: dict,
    run_dir: str,
) -> dict:
    data["dataset"].use_frequency_feat = False
    train_loader = DataLoader(data["train_subset"], batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(data["test_subset"], batch_size=cfg.batch_size, shuffle=False)

    model_obj = model_obj.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(model_obj.parameters(), cfg.model.optimizer)
    save_path = os.path.join(run_dir, f"{cfg.model.name}_s{cfg.subject}.pth")

    best_top1 = 0.0
    best_top5 = 0.0
    best_epoch = -1

    epoch_bar = tqdm(range(cfg.model.epochs), desc="semantic-train", unit="ep")
    for epoch in epoch_bar:
        model_obj.train()
        running_total = 0.0
        running_ce = 0.0
        running_triplet = 0.0
        running_jepa = 0.0

        for inputs, labels in train_loader:
            labels = remap_labels(labels, data["label_map"])
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model_obj(inputs)

            ce_loss = criterion(outputs["logits"], labels)
            triplet_loss = _batch_hard_triplet_loss(
                outputs["embedding"], labels, margin=float(cfg.model.get("triplet_margin", 0.2))
            )
            jepa_loss = F.smooth_l1_loss(outputs["jepa_pred"], outputs["jepa_target"].detach())

            total_loss = (
                float(cfg.model.get("ce_weight", 1.0)) * ce_loss
                + float(cfg.model.get("triplet_weight", 0.5)) * triplet_loss
                + float(cfg.model.get("jepa_weight", 0.5)) * jepa_loss
            )
            total_loss.backward()
            optimizer.step()
            model_obj.update_target_encoder(float(cfg.model.get("ema_decay", 0.996)))

            running_total += total_loss.item()
            running_ce += ce_loss.item()
            running_triplet += triplet_loss.item()
            running_jepa += jepa_loss.item()

        top1, top5, val_loss = _evaluate_semantic(model_obj, test_loader, criterion, device, data["label_map"])
        epoch_bar.set_postfix(
            tr_total=f"{running_total / max(1, len(train_loader)):.4f}",
            tr_ce=f"{running_ce / max(1, len(train_loader)):.4f}",
            tr_tri=f"{running_triplet / max(1, len(train_loader)):.4f}",
            tr_jepa=f"{running_jepa / max(1, len(train_loader)):.4f}",
            val_top1=f"{top1:.3f}",
            val_top5=f"{top5:.3f}",
            val_loss=f"{val_loss:.4f}",
        )

        if top1 > best_top1:
            best_top1 = top1
            best_top5 = top5
            best_epoch = epoch
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model_obj.state_dict(), save_path)

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
    print(f"[simple] test_top1={top1:.4f}")
    return {"simple_top1": top1}

def _train_nn_model(
    model_obj: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    label_map: dict[int, int],
    save_path: str,
) -> dict:
    best_top1 = 0.0
    best_epoch = -1

    epoch_bar = tqdm(range(num_epochs), desc="nn-train", unit="ep")
    for epoch in epoch_bar:
        model_obj.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            labels = remap_labels(labels, label_map)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model_obj(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        top1, _, val_loss = _evaluate_semantic(model_obj, test_loader, criterion, device, label_map)
        epoch_bar.set_postfix(
            tr_loss=f"{running_loss / max(1, len(train_loader)):.4f}",
            val_top1=f"{top1:.3f}",
            val_loss=f"{val_loss:.4f}",
        )

        if top1 > best_top1:
            best_top1 = top1
            best_epoch = epoch
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model_obj.state_dict(), save_path)

    with open(os.path.join(run_dir, "result.txt"), "a", encoding="utf-8") as f:
        f.write(f"top1={best_top1:.4f}\n")
        f.write(f"best_epoch={best_epoch}\n")

    return {"deep_top1": best_top1, "deep_epoch": best_epoch}

def train_model(cfg: DictConfig, device: torch.device, model_obj: object, data: dict, run_dir: str) -> dict:
    if cfg.model.name.lower() in "semantic_model":
        return _train_semantic_model(cfg, device, model_obj, data, run_dir)
    if cfg.model.name.lower() == "eegnet":
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
    if model_name in "semantic_model":
        print(
            f"[semantic] best_top1={train_results['semantic_top1']:.4f} "
            f"best_top5={train_results['semantic_top5']:.4f} epoch={train_results['semantic_epoch']}"
        )
        return
    if cfg.model.type == "simple":
        print(f"[simple] best_top1={train_results['simple_top1']:.4f}")
        return
    print(f"[deep] best_top1={train_results['deep_top1']:.4f} epoch={train_results['deep_epoch']}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    device = get_device()

    data = load_data(cfg, device)
    model_obj = model_init(cfg, data["num_classes"], device)
    run_dir = HydraConfig.get().runtime.output_dir
    train_results = train_model(cfg, device, model_obj, data, run_dir)
    evaluate_model(cfg, train_results)


if __name__ == "__main__":
    main()
