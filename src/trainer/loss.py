from __future__ import annotations

import torch
import torch.nn.functional as F


def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances between rows of x and y."""
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True).T
    return (x_norm + y_norm - 2.0 * (x @ y.T)).clamp_min_(0.0)


def mmd_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0),
) -> torch.Tensor:
    """Unbiased multi-kernel RBF MMD.

    Args:
        x, y:   (batch, D) l2-normalised tensors
        sigmas: RBF bandwidth values

    Returns:
        Scalar MMD^2 estimate (clamped >= 0).
    """
    if x.shape[0] < 2:
        return torch.tensor(0.0, device=x.device)
    sigmas_t = torch.tensor(sigmas, device=x.device, dtype=x.dtype)

    def rbf(d2: torch.Tensor) -> torch.Tensor:
        return sum(torch.exp(-d2 / (2 * s ** 2)) for s in sigmas_t) / len(sigmas_t)

    B   = x.shape[0]
    Kxx = rbf(_pairwise_sq_dists(x, x))
    Kyy = rbf(_pairwise_sq_dists(y, y))
    Kxy = rbf(_pairwise_sq_dists(x, y))
    sxx = (Kxx.sum() - Kxx.diag().sum()) / (B * (B - 1))
    syy = (Kyy.sum() - Kyy.diag().sum()) / (B * (B - 1))
    return (sxx + syy - 2.0 * Kxy.mean()).clamp_min(0.0)


def get_mmd_weight(
    epoch: int,
    stage1_epochs: int,
    mmd_start: float = 0.9,
    mmd_end: float = 0.5,
) -> float:
    """Linearly decay MMD weight from mmd_start to mmd_end over stage 1."""
    t = (epoch - 1) / max(stage1_epochs - 1, 1)
    return mmd_start + (mmd_end - mmd_start) * t


def info_nce_loss(
    zE: torch.Tensor,
    zI: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric InfoNCE loss with learnable logit scale.

    Args:
        zE:          EEG embeddings (batch, D), l2-normalised
        zI:          Image embeddings (batch, D), l2-normalised
        logit_scale: Learnable log-scale parameter

    Returns:
        Scalar loss tensor.
    """
    scale  = torch.exp(logit_scale)
    sim    = zE @ zI.T * scale
    labels = torch.arange(len(zE), device=zE.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


def compute_loss(
    zE: torch.Tensor,
    zI: torch.Tensor,
    logit_scale: torch.Tensor,
    epoch: int,
    stage1_epochs: int,
    mmd_start: float = 0.9,
    mmd_end: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Two-stage SUPAEEG training objective.

    Stage 1 (epoch <= stage1_epochs):
        mmd_w * MMD_RBF(zE, zI) + (1 - mmd_w) * InfoNCE(zE, zI)
        mmd_w decays linearly from mmd_start to mmd_end.
    Stage 2 (epoch > stage1_epochs):
        InfoNCE(zE, zI) only

    Args:
        zE, zI:        l2-normalised embeddings each (batch, D)
        logit_scale:   Learnable log-scale parameter
        epoch:         Current training epoch (1-indexed)
        stage1_epochs: Number of stage-1 epochs
        mmd_start:     Initial MMD weight
        mmd_end:       Final MMD weight at end of stage 1

    Returns:
        (total_loss, components_dict)
    """
    infonce = info_nce_loss(zE, zI, logit_scale)
    if epoch <= stage1_epochs:
        mmd_w = get_mmd_weight(epoch, stage1_epochs, mmd_start, mmd_end)
        mmd   = mmd_rbf(zE, zI)
        total = mmd_w * mmd + (1 - mmd_w) * infonce
        return total, {
            "total":      total.item(),
            "infonce":    infonce.item(),
            "mmd":        mmd.item(),
            "mmd_weight": mmd_w,
        }
    return infonce, {
        "total":      infonce.item(),
        "infonce":    infonce.item(),
        "mmd":        0.0,
        "mmd_weight": 0.0,
    }
