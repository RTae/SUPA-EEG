from __future__ import annotations

import torch
import torch.nn.functional as F

def info_nce_loss(zk: torch.Tensor, Sk: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE contrastive loss between EEG and visual embeddings.

    Args:
        zk:  EEG embeddings, shape ``(batch, D)``.
        Sk:  Visual feature targets, shape ``(batch, D)``.
        tau: Temperature scaling factor.  Default: 0.07.

    Returns:
        Scalar loss tensor.
    """
    zk = F.normalize(zk, dim=1)
    Sk = F.normalize(Sk, dim=1)
    logits = zk @ Sk.T / tau
    labels = torch.arange(len(zk), device=zk.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def sigreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    z3: torch.Tensor,
) -> torch.Tensor:
    """Gaussian regulariser to prevent representation collapse.

    Penalises embeddings whose per-dimension statistics deviate from N(0,1),
    encouraging the encoder to use all embedding dimensions.

    Args:
        z1, z2, z3: Scale embeddings, each ``(batch, D)``.

    Returns:
        Scalar loss tensor (sum of per-scale KL divergences to N(0,1)).
    """
    total = 0.0
    for zk in (z1, z2, z3):
        mean_k = zk.mean(dim=0)
        std_k = zk.std(dim=0) + 1e-8
        total = total + 0.5 * (mean_k.pow(2) + std_k.pow(2) - std_k.pow(2).log() - 1).sum()
    return total  # type: ignore[return-value]


def l1_sparsity_loss(scale_encoder: "MultiScaleEncoder") -> torch.Tensor:  # type: ignore[name-defined]
    """L1 sparsity penalty on channel attention weights.

    Encourages each scale to rely on a sparse subset of EEG channels.

    Args:
        scale_encoder: The MultiScaleEncoder whose ``last_alpha`` attributes to penalise.

    Returns:
        Scalar loss tensor.
    """
    return (
        scale_encoder.channel_attn_1.last_alpha.abs().sum()  # type: ignore[union-attr]
        + scale_encoder.channel_attn_2.last_alpha.abs().sum()  # type: ignore[union-attr]
        + scale_encoder.channel_attn_3.last_alpha.abs().sum()  # type: ignore[union-attr]
    )


def compute_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    z3: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    scale_encoder: "MultiScaleEncoder",  # type: ignore[name-defined]
    lambda_reg: float = 0.1,
    beta_l1: float = 0.01,
    tau: float = 0.07,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Full SUPAEEG training objective.

    Combines symmetric InfoNCE (per depth scale) with a Gaussian regulariser
    and an L1 channel-sparsity penalty.

    Args:
        z1, z2, z3:    EEG scale embeddings, each ``(batch, 768)``.
        S1, S2, S3:    CLIP visual targets, each ``(batch, 768)``.
        scale_encoder: MultiScaleEncoder holding the channel attention weights.
        lambda_reg:    Weight for the Gaussian regulariser.  Default: 0.1.
        beta_l1:       Weight for the L1 sparsity penalty.  Default: 0.01.
        tau:           InfoNCE temperature.  Default: 0.07.

    Returns:
        ``(total_loss, components)`` where ``components`` contains
        ``'total'``, ``'infonce'``, ``'sigreg'``, ``'l1'`` as floats.
    """
    infonce = info_nce_loss(z1, S1, tau) + info_nce_loss(z2, S2, tau) + info_nce_loss(z3, S3, tau)
    sigreg = sigreg_loss(z1, z2, z3)
    l1 = l1_sparsity_loss(scale_encoder)
    total = infonce + lambda_reg * sigreg + beta_l1 * l1
    return total, {
        "total": total.item(),
        "infonce": infonce.item(),
        "sigreg": sigreg.item(),
        "l1": l1.item(),
    }
