"""Label mapping and evaluation helpers."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score
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
) -> tuple[float, float]:
    """Return (accuracy, avg_loss) on the given dataloader."""
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    for inputs, labels in dataloader:
        labels = remap_labels(labels, label_map)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        total_loss += criterion(outputs, labels).item()
        predicted = outputs.argmax(dim=1)
        total += len(labels)
        correct += accuracy_score(labels.cpu(), predicted.cpu(), normalize=False)
    return correct / total, total_loss / len(dataloader)


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
