import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

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

    Each row i of eeg_features is an EEG concept embedding (mean over test trials,
    ℓ2-normalised); the corresponding row i of image_features is that concept's CLIP
    image embedding. The correct retrieval for query i is always column i —
    the diagonal-match convention.

    Used by ``utilities.evaluate`` for SUPAEEG evaluation against the pre-extracted
    CLIP feature gallery.

    Args:
        eeg_features:   (N, D) ℓ2-normalised EEG embeddings, one per concept.
        image_features: (N, D) ℓ2-normalised image embeddings, same concept order.

    Returns:
        (top5_count, top1_count, total)
    """
    similarity_matrix = sk_cosine_similarity(eeg_features, image_features)
    count_5, count_1 = retrieve_topk(similarity_matrix, 5)
    return count_5, count_1, eeg_features.shape[0]