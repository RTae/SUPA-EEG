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
        k5 = min(5, outputs.shape[-1])
        top5_idx = outputs.topk(k5, dim=1).indices          # (B, k5)
        top1_correct += int(top5_idx[:, :1].eq(labels.unsqueeze(1)).any(dim=1).sum().item())
        top5_correct += int(top5_idx.eq(labels.unsqueeze(1)).any(dim=1).sum().item())
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
