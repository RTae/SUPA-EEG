"""Evaluation helpers."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


from .loss import batch_hard_triplet_loss


def topk_correct(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    """Return the count of samples whose true label is among the top-k predictions."""
    k = min(k, logits.shape[-1])
    return int(logits.topk(k, dim=1).indices.eq(labels.unsqueeze(1)).any(dim=1).sum().item())


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
) -> tuple[float, float, float]:
    """Return (top1_acc, top5_acc, avg_loss) on the given dataloader."""
    model.eval()
    top1_correct = top5_correct = total = 0
    total_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        total_loss += criterion(outputs, labels).item()
        total += len(labels)
        top1_correct += topk_correct(outputs, labels, 1)
        top5_correct += topk_correct(outputs, labels, 5)
    denom = max(total, 1)
    return top1_correct / denom, top5_correct / denom, total_loss / max(len(dataloader), 1)


@torch.no_grad()
def _collect_semantic_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_embeddings: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_embeddings.append(F.normalize(outputs["embedding"], dim=1).cpu())
        all_labels.append(labels.cpu())

    if not all_embeddings:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def evaluate_semantic_embeddings(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    triplet_margin: float,
) -> tuple[float, float, float]:
    """Evaluate semantic embeddings using eval-set class prototypes and triplet loss."""
    embeddings, labels = _collect_semantic_embeddings(model, eval_loader, device)

    if embeddings.numel() == 0:
        return 0.0, 0.0, 0.0

    prototype_labels = torch.unique(labels, sorted=True)
    prototypes = []
    for class_id in prototype_labels.tolist():
        class_embeddings = embeddings[labels == class_id]
        prototype = F.normalize(class_embeddings.mean(dim=0, keepdim=True), dim=1)
        prototypes.append(prototype.squeeze(0))
    prototype_matrix = torch.stack(prototypes, dim=0)

    similarities = embeddings @ prototype_matrix.t()
    label_to_proto = {int(class_id): idx for idx, class_id in enumerate(prototype_labels.tolist())}
    mapped_labels = torch.tensor([label_to_proto[int(l.item())] for l in labels], dtype=torch.long)

    total = len(mapped_labels)
    top1 = topk_correct(similarities, mapped_labels, 1) / max(total, 1)
    top5 = topk_correct(similarities, mapped_labels, 5) / max(total, 1)
    val_loss = float(batch_hard_triplet_loss(embeddings, labels, triplet_margin).item())

    return top1, top5, val_loss


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
