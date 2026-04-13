"""Label mapping and evaluation helpers."""

import numpy as np
import torch
from torch.utils.data import DataLoader


def build_label_map(all_labels: np.ndarray) -> dict[int, int]:
    """Map original label ids to contiguous 0-based indices."""
    unique = torch.from_numpy(all_labels).unique()
    return {orig.item(): new for new, orig in enumerate(unique)}


def remap_labels(labels: torch.Tensor, label_map: dict[int, int]) -> torch.Tensor:
    return torch.tensor([label_map[l.item()] for l in labels])


def topk_correct(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    """Return the count of samples whose true label is among the top-k predictions."""
    k = min(k, logits.shape[-1])
    return int(logits.topk(k, dim=1).indices.eq(labels.unsqueeze(1)).any(dim=1).sum().item())


def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    """Compute batch-hard triplet loss with hardest positive/negative mining."""
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
    return torch.relu(hardest_pos - hardest_neg + margin)[valid].mean()


def resolve_clip_targets(
    labels: tuple[str, ...] | list[str],
    embeddings: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    return torch.stack([embeddings[n] for n in labels]).squeeze().to(device)


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    label_map: dict[int, int],
) -> tuple[float, float, float]:
    """Return (top1_acc, top5_acc, avg_loss) on the given dataloader."""
    model.eval()
    top1_correct = top5_correct = total = 0
    total_loss = 0.0
    for inputs, labels in dataloader:
        labels = remap_labels(labels, label_map)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        total_loss += criterion(outputs, labels).item()
        total += len(labels)
        top1_correct += topk_correct(outputs, labels, 1)
        top5_correct += topk_correct(outputs, labels, 5)
    denom = max(total, 1)
    return top1_correct / denom, top5_correct / denom, total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate_semantic_classifier(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    label_map: dict[int, int],
) -> tuple[float, float, float]:
    """Evaluate models that return dict outputs containing a logits tensor."""
    model.eval()
    top1_correct = top5_correct = total = 0
    total_loss = 0.0

    for inputs, labels in dataloader:
        labels = remap_labels(labels, label_map)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        logits = outputs["logits"]
        total_loss += criterion(logits, labels).item()
        total += len(labels)
        top1_correct += topk_correct(logits, labels, 1)
        top5_correct += topk_correct(logits, labels, 5)

    denom = max(total, 1)
    return top1_correct / denom, top5_correct / denom, total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate_generator(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    clip_embeddings: dict[str, torch.Tensor],
) -> float:
    """Return average test loss for the embedding-regression model."""
    model.eval()
    total_loss = sum(
        criterion(
            model(inputs.to(device)),
            resolve_clip_targets(labels, clip_embeddings, device),
        ).item()
        for inputs, labels in dataloader
    )
    return total_loss / len(dataloader)
