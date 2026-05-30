"""Evaluation helpers."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from torch.utils.data import DataLoader

from .loss import batch_hard_triplet_loss


def topk_correct(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    """Return the count of samples whose true label is among the top-k predictions."""
    k = min(k, logits.shape[-1])
    return int(logits.topk(k, dim=1).indices.eq(labels.unsqueeze(1)).any(dim=1).sum().item())


def retrieve_topk(similarity_matrix: np.ndarray, k: int) -> tuple[int, int]:
    """Count top-k and top-1 correct retrievals using the SAMGA paper's diagonal convention.

    This implements the paper-exact EEG-to-image retrieval metric: the matrix must be
    square N×N, where entry (i, i) is the similarity between EEG query i and its
    corresponding image embedding i. This requires a separate image embedding gallery
    (e.g., CLIP features) aligned to the EEG embeddings by concept index.

    NOT suitable for prototype/label-based retrieval — use _label_retrieval_counts for that.

    Args:
        similarity_matrix: Square (N, N) cosine-similarity matrix. Entry (i, i) is correct.
        k: Rank threshold (e.g., 5 for Top-5).

    Returns:
        (top_k_count, top_1_count): number of queries whose correct item ranks <= k or == 1.
    """
    if similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError(
            f"retrieve_topk expects a square N×N similarity matrix "
            f"(diagonal = correct EEG–image pair); got shape {similarity_matrix.shape}. "
            "For label-based prototype retrieval use _label_retrieval_counts instead."
        )
    sorted_indices = np.argsort(-similarity_matrix, axis=1)
    rankings = np.argsort(sorted_indices, axis=1)
    diagonal_ranks = np.diag(rankings) + 1  # 1-indexed rank of each correct pair
    count_k = int((diagonal_ranks <= k).sum())
    count_1 = int((diagonal_ranks == 1).sum())
    return count_k, count_1


def retrieve_all(
    eeg_features: np.ndarray,
    image_features: np.ndarray,
) -> tuple[int, int, int]:
    """Paper-style zero-shot EEG-to-image retrieval evaluation (SAMGA protocol).

    Each row i of eeg_features is an EEG concept embedding; the corresponding row i
    of image_features is that concept's image embedding (e.g., from CLIP). The correct
    retrieval for query i is always column i — the diagonal-match convention.

    Both arrays must be ℓ2-normalised and share the same concept ordering.
    This is a forward-looking API for when image/CLIP embeddings become available;
    our current eval loop uses _label_retrieval_counts (prototype/class-based).

    Args:
        eeg_features:   (N, D) ℓ2-normalised EEG embeddings, one per concept.
        image_features: (N, D) ℓ2-normalised image embeddings, same concept order.

    Returns:
        (top5_count, top1_count, total)
    """
    similarity_matrix = sk_cosine_similarity(eeg_features, image_features)
    count_5, count_1 = retrieve_topk(similarity_matrix, 5)
    return count_5, count_1, eeg_features.shape[0]


def _label_retrieval_counts(
    sim_matrix: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> tuple[int, int]:
    """Label-indexed retrieval counts on a rectangular similarity matrix.

    This is our current evaluation metric for the SemanticModel: each EEG trial is
    ranked against all class prototypes (N_trials × N_classes matrix), and we count
    how often the correct class lands in the top-k or top-1 position.

    This differs from retrieve_topk / retrieve_all (paper-style diagonal matching):
    - The matrix is rectangular, not square.
    - Correctness is determined by labels[i], not the diagonal.
    - No separate image embedding gallery is required.

    Args:
        sim_matrix: (N_trials, N_classes) cosine-similarity matrix.
        labels:     (N_trials,) integer array; labels[i] is the column index of the
                    correct class for trial i, in [0, N_classes).
        k:          Rank threshold (e.g., 5 for Top-5).

    Returns:
        (count_k, count_1): trials whose correct class ranks <= k, and exactly == 1.
    """
    N = sim_matrix.shape[0]
    sorted_indices = np.argsort(-sim_matrix, axis=1)
    rankings = np.argsort(sorted_indices, axis=1)
    correct_ranks = rankings[np.arange(N), labels] + 1  # 1-indexed
    return int((correct_ranks <= k).sum()), int((correct_ranks == 1).sum())


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
    """Evaluate semantic embeddings via retrieval against eval-set class prototypes.

    Follows the paper's retrieval protocol: cosine similarity between each test
    trial's EEG embedding and all class prototypes, then top-1 / top-5 accuracy
    measure how often the correct class is ranked first / in the top-5.
    """
    embeddings, labels = _collect_semantic_embeddings(model, eval_loader, device)

    if embeddings.numel() == 0:
        return 0.0, 0.0, 0.0

    prototype_labels = torch.unique(labels, sorted=True)
    prototypes = []
    for class_id in prototype_labels.tolist():
        class_embeddings = embeddings[labels == class_id]
        prototype = F.normalize(class_embeddings.mean(dim=0, keepdim=True), dim=1)
        prototypes.append(prototype.squeeze(0))
    prototype_matrix = torch.stack(prototypes, dim=0)  # N_classes × D

    # N_trials × N_classes cosine similarity (embeddings are already ℓ2-normalised)
    eeg_np = embeddings.numpy()
    proto_np = prototype_matrix.numpy()
    sim_matrix = sk_cosine_similarity(eeg_np, proto_np)

    label_to_proto = {int(c): i for i, c in enumerate(prototype_labels.tolist())}
    mapped_labels = np.array([label_to_proto[int(l.item())] for l in labels])

    count_5, count_1 = _label_retrieval_counts(sim_matrix, mapped_labels, k=5)
    N = sim_matrix.shape[0]
    top1 = count_1 / max(N, 1)
    top5 = count_5 / max(N, 1)
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
