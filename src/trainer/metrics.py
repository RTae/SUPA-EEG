"""Label mapping and evaluation helpers."""

import numpy as np
import torch
import torch.nn.functional as F
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


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Compute batch-hard triplet loss with hardest positive/negative mining."""
    if embeddings.shape[0] < 2:
        return embeddings.new_tensor(0.0)

    if not embeddings.isfinite().all():
        n_nan = (~embeddings.isfinite()).any(dim=1).sum().item()
        import logging
        logging.warning(f"[triplet] {n_nan}/{embeddings.shape[0]} embeddings contain NaN/Inf — returning 0")
        return embeddings.new_tensor(0.0)

    # Use matmul-based distance (MPS-compatible). Embeddings are L2-normalised,
    # so ||a - b||^2 = 2 - 2*(a·b^T), clamped to avoid sqrt of negatives.
    sim = embeddings @ embeddings.t()
    dist_mat = (2.0 - 2.0 * sim).clamp(min=0.0).sqrt()
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)

    pos_mask = same_label & ~eye
    neg_mask = ~same_label

    has_pos = pos_mask.any(dim=1)
    has_neg = neg_mask.any(dim=1)
    valid = has_pos & has_neg
    if not valid.any():
        import logging
        logging.warning(f"[triplet] no valid triplets in batch — labels={labels.tolist()}")
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
def _collect_semantic_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label_map: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_embeddings: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for inputs, labels in dataloader:
        labels = remap_labels(labels, label_map)
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
    label_map: dict[int, int],
    triplet_margin: float,
) -> tuple[float, float, float]:
    """Evaluate semantic embeddings using eval-set class prototypes and triplet loss."""
    embeddings, labels = _collect_semantic_embeddings(model, eval_loader, device, label_map)

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
