import logging

import torch
import torch.nn as nn


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
        logging.warning(f"[triplet] {n_nan}/{embeddings.shape[0]} embeddings contain NaN/Inf — returning 0")
        return embeddings.new_tensor(0.0)

    # Use squared Euclidean distance on L2-normalised embeddings.
    # Avoid sqrt here: d/dx sqrt(x) explodes at x=0 and can destabilise training
    # when positives collapse to identical embeddings early in optimisation.
    sim = (embeddings @ embeddings.t()).clamp(min=-1.0, max=1.0)
    dist_mat = (2.0 - 2.0 * sim).clamp(min=0.0)

    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)

    pos_mask = same_label & ~eye
    neg_mask = ~same_label

    has_pos = pos_mask.any(dim=1)
    has_neg = neg_mask.any(dim=1)
    valid = has_pos & has_neg
    if not valid.any():
        logging.warning(f"[triplet] no valid triplets in batch — labels={labels.tolist()}")
        return embeddings.new_tensor(0.0)

    max_dist = dist_mat.max().detach() + 1.0
    hardest_pos_idx = dist_mat.masked_fill(~pos_mask, -1.0).argmax(dim=1)
    hardest_neg_idx = dist_mat.masked_fill(~neg_mask, max_dist).argmin(dim=1)

    valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
    anchors = embeddings[valid_idx]
    positives = embeddings[hardest_pos_idx[valid_idx]]
    negatives = embeddings[hardest_neg_idx[valid_idx]]

    triplet_criterion = nn.TripletMarginLoss(margin=margin, p=2, reduction="mean")
    return triplet_criterion(anchors, positives, negatives)