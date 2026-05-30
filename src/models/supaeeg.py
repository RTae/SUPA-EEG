"""SUPAEEG: Scale-Unified Parieto-occipital Architecture

Full model definition, loss functions, and supporting classes.

Architecture overview
---------------------
EEG signal (batch, 17, 100)
  │
  ├─ ChannelAttention (scale-1)  ─┐
  ├─ ChannelAttention (scale-2)   ├─ EEGNetEncoder (shared weights)
  └─ ChannelAttention (scale-3)  ─┘
           │
  Shared TransformerEncoder
           │
  Scale-specific projection heads (256 → 768)
           │
  z1, z2, z3  (batch, 768)  ──► InfoNCE vs CLIP S1/S2/S3 targets

Training targets S1/S2/S3 come from a frozen VisualFeatureLookup;
gradients never flow through that lookup.

At inference, embed() concatenates and ℓ2-normalises [z1, z2, z3]
to produce a (batch, 2304) descriptor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders.eegnet_encoder import EEGNetEncoder
from src.encoders.visual_encoder import VisualFeatureLookup


# ---------------------------------------------------------------------------
# ChannelAttention
# ---------------------------------------------------------------------------


class ChannelAttention(nn.Module):
    """Per-scale soft channel weighting via a scale prototype vector.

    Given a prototype vector ``ck`` for a particular depth scale, this module
    produces a soft attention mask over EEG channels and multiplies it into
    the raw EEG signal.  The resulting weighted EEG emphasises the channels
    most relevant to that depth scale.

    Args:
        n_channels: Number of EEG channels.  Default: 17.
        d_model:    Dimensionality of the scale prototype vector.  Default: 256.
    """

    def __init__(self, n_channels: int = 17, d_model: int = 256) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, n_channels)
        # last_alpha is set during forward and exposed for L1 sparsity loss.
        self.last_alpha: torch.Tensor | None = None

    def forward(self, eeg: torch.Tensor, ck: nn.Parameter) -> torch.Tensor:
        """Apply channel attention to the EEG signal.

        Args:
            eeg: Raw EEG tensor of shape ``(batch, n_channels, n_timepoints)``.
            ck:  Scale prototype vector, shape ``(1, 1, d_model)``.

        Returns:
            Channel-weighted EEG tensor of shape ``(batch, n_channels, n_timepoints)``.
        """
        # ck: (1, 1, 256) → squeeze → (256,)
        # linear → (n_channels,) → softmax → (n_channels,)
        alpha = F.softmax(self.linear(ck.squeeze()), dim=0)
        self.last_alpha = alpha  # expose for L1 sparsity loss

        # (batch, n_channels, n_timepoints) * (n_channels, 1) → broadcast
        return eeg * alpha.unsqueeze(-1)


# ---------------------------------------------------------------------------
# MultiScaleEncoder
# ---------------------------------------------------------------------------


class MultiScaleEncoder(nn.Module):
    """Shared transformer encoder with three scale prototype vectors.

    Three learned prototype vectors (c1, c2, c3) each condition a separate
    ChannelAttention module, then the weighted EEG is tokenised by a shared
    EEGNetEncoder.  Each token sequence is prefixed with its scale prototype
    and passed through a single shared TransformerEncoder.  The output at
    position 0 (the prototype token, BERT CLS convention) is projected to
    visual feature space (768-d).

    All three branches share the same TransformerEncoder weights, forcing the
    encoder to learn a representation that is simultaneously useful for all
    three depth scales.

    Args:
        d_model:        Token embedding dimension.  Default: 256.
        nhead:          Number of attention heads.  Default: 8.
        num_layers:     Number of transformer encoder layers.  Default: 4.
        dim_feedforward: Feed-forward hidden size.  Default: 512.
        dropout:        Dropout probability.  Default: 0.1.
        d_visual:       Dimensionality of the visual feature targets.  Default: 768.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        d_visual: int = 768,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Learned scale prototype vectors (NOT I-JEPA features)
        # Each is supervised indirectly through the InfoNCE gradient.
        # ------------------------------------------------------------------
        self.c1 = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.c2 = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.c3 = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ------------------------------------------------------------------
        # Per-scale channel attention modules
        # ------------------------------------------------------------------
        self.channel_attn_1 = ChannelAttention(n_channels=17, d_model=d_model)
        self.channel_attn_2 = ChannelAttention(n_channels=17, d_model=d_model)
        self.channel_attn_3 = ChannelAttention(n_channels=17, d_model=d_model)

        # ------------------------------------------------------------------
        # Shared transformer encoder (weights used by all three branches)
        # ------------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ------------------------------------------------------------------
        # Scale-specific projection heads: d_model → d_visual
        # ------------------------------------------------------------------
        self.proj1 = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Linear(512, d_visual),
        )
        self.proj2 = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Linear(512, d_visual),
        )
        self.proj3 = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Linear(512, d_visual),
        )

    def forward(
        self,
        eeg_raw: torch.Tensor,
        eegnet: EEGNetEncoder,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a batch of EEG signals into three scale embeddings.

        Args:
            eeg_raw: Raw EEG tensor of shape ``(batch, 17, 100)``.
            eegnet:  Shared EEGNetEncoder instance.

        Returns:
            Tuple ``(z1, z2, z3)`` each of shape ``(batch, 768)``.
        """
        batch = eeg_raw.shape[0]

        results = []
        for channel_attn, ck, proj in (
            (self.channel_attn_1, self.c1, self.proj1),
            (self.channel_attn_2, self.c2, self.proj2),
            (self.channel_attn_3, self.c3, self.proj3),
        ):
            # Apply scale-specific channel attention: (batch, 17, 100)
            weighted = channel_attn(eeg_raw, ck)

            # Tokenise: (batch, N_t, 256)
            tokens = eegnet(weighted)

            # Prepend scale prototype as CLS token: (batch, 1, 256)
            prefix = ck.expand(batch, 1, 256)

            # Concatenate CLS + tokens: (batch, 1+N_t, 256)
            seq = torch.cat([prefix, tokens], dim=1)

            # Shared transformer: (batch, 1+N_t, 256)
            out = self.transformer(seq)

            # Take CLS output at position 0 and project to visual space
            z = proj(out[:, 0, :])  # (batch, 768)
            results.append(z)

        z1, z2, z3 = results
        return z1, z2, z3


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


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
    loss_e2i = F.cross_entropy(logits, labels)
    loss_i2e = F.cross_entropy(logits.T, labels)
    return (loss_e2i + loss_i2e) / 2


def sigreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    z3: torch.Tensor,
) -> torch.Tensor:
    """Gaussian regulariser to prevent representation collapse.

    Penalises embeddings whose per-dimension statistics deviate far from a
    standard normal, encouraging the encoder to use all embedding dimensions.

    Args:
        z1: Scale-1 embeddings, shape ``(batch, D)``.
        z2: Scale-2 embeddings, shape ``(batch, D)``.
        z3: Scale-3 embeddings, shape ``(batch, D)``.

    Returns:
        Scalar loss tensor (sum of per-scale KL divergences to N(0,1)).
    """
    total = 0.0
    for zk in (z1, z2, z3):
        mean_k = zk.mean(dim=0)
        std_k = zk.std(dim=0) + 1e-8
        kl_k = 0.5 * (
            mean_k.pow(2) + std_k.pow(2) - std_k.pow(2).log() - 1
        ).sum()
        total = total + kl_k
    return total  # type: ignore[return-value]


def l1_sparsity_loss(scale_encoder: MultiScaleEncoder) -> torch.Tensor:
    """L1 sparsity penalty on channel attention weights.

    Encourages each scale to rely on a sparse subset of EEG channels.
    The ``last_alpha`` attribute of each ChannelAttention module is populated
    on every forward pass.

    Args:
        scale_encoder: The MultiScaleEncoder whose attention weights to penalise.

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
    scale_encoder: MultiScaleEncoder,
    lambda_reg: float = 0.1,
    beta_l1: float = 0.01,
    tau: float = 0.07,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Full SUPAEEG training objective.

    Combines symmetric InfoNCE contrastive loss (one per depth scale) with
    a Gaussian regulariser and an L1 channel-sparsity penalty.

    Args:
        z1, z2, z3:     EEG scale embeddings, each ``(batch, 768)``.
        S1, S2, S3:     CLIP visual targets, each ``(batch, 768)``.
        scale_encoder:  MultiScaleEncoder holding the attention weights.
        lambda_reg:     Weight for the Gaussian regulariser.  Default: 0.1.
        beta_l1:        Weight for the L1 sparsity penalty.  Default: 0.01.
        tau:            InfoNCE temperature.  Default: 0.07.

    Returns:
        Tuple ``(total_loss, components_dict)`` where ``components_dict``
        contains ``'total'``, ``'infonce'``, ``'sigreg'``, ``'l1'`` scalars.
    """
    infonce = (
        info_nce_loss(z1, S1, tau)
        + info_nce_loss(z2, S2, tau)
        + info_nce_loss(z3, S3, tau)
    )
    sigreg = sigreg_loss(z1, z2, z3)
    l1 = l1_sparsity_loss(scale_encoder)
    total = infonce + lambda_reg * sigreg + beta_l1 * l1
    components = {
        "total": total.item(),
        "infonce": infonce.item(),
        "sigreg": sigreg.item(),
        "l1": l1.item(),
    }
    return total, components


# ---------------------------------------------------------------------------
# SUPAEEG — full model
# ---------------------------------------------------------------------------


class SUPAEEG(nn.Module):
    """SUPAEEG: Scale-conditioned Unified Predictive Alignment for EEG.

    Full model combining EEGNetEncoder + MultiScaleEncoder.

    The model maps raw EEG trials to three depth-scale embeddings (z1, z2, z3)
    that are trained to align with frozen CLIP features (S1/S2/S3) via a
    symmetric InfoNCE loss.

    Key design decisions
    --------------------
    - At test time only the learned EEG encoder is required.
    - No image encoder or generative model is needed at inference.
    - I-JEPA/CLIP features are used only as fixed training targets.
    - Gradient does **not** flow into VisualFeatureLookup at any point.
    - c1/c2/c3 are nn.Parameters (not I-JEPA features).

    Args:
        feature_lookup: Pre-loaded VisualFeatureLookup (never trained, no grad).
        d_model:        Token embedding dimension.  Default: 256.
        nhead:          Transformer attention heads.  Default: 8.
        num_layers:     Transformer encoder depth.  Default: 4.
        dim_feedforward: Feed-forward hidden size.  Default: 512.
        dropout:        Dropout probability.  Default: 0.1.
        device:         Target device for tensor placement.
    """

    def __init__(
        self,
        feature_lookup: VisualFeatureLookup,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()

        self.device = torch.device(device) if isinstance(device, str) else device
        self.feature_lookup = feature_lookup  # not an nn.Module, no grad

        self.eeg_encoder = EEGNetEncoder(n_channels=17, n_timepoints=100)
        self.scale_encoder = MultiScaleEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            d_visual=768,
        )

    def forward(
        self,
        eeg: torch.Tensor,
        image_concepts: list[str],
        image_files: list[str],
        return_loss: bool = True,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Forward pass with optional loss computation.

        Args:
            eeg:            EEG tensor ``(batch, 17, 100)``.
            image_concepts: List of concept folder names, length ``batch``.
            image_files:    List of image filenames, length ``batch``.
            return_loss:    If ``True``, retrieve visual targets and compute
                            the full training loss.

        Returns:
            If ``return_loss=True``:
                ``(z1, z2, z3, loss, components)``
            Otherwise:
                ``(z1, z2, z3)``
        """
        z1, z2, z3 = self.scale_encoder(eeg, self.eeg_encoder)

        if return_loss:
            # Retrieve fixed visual targets — no gradient flows here
            S1, S2, S3 = self.feature_lookup.retrieve_batch(image_concepts, image_files)
            S1 = S1.to(self.device)
            S2 = S2.to(self.device)
            S3 = S3.to(self.device)
            loss, components = compute_loss(
                z1, z2, z3,
                S1, S2, S3,
                self.scale_encoder,
            )
            return z1, z2, z3, loss, components

        return z1, z2, z3

    @torch.no_grad()
    def embed(self, eeg: torch.Tensor) -> torch.Tensor:
        """Inference-only multi-scale EEG embedding.

        Concatenates the three scale embeddings and returns an ℓ2-normalised
        descriptor suitable for retrieval evaluation.

        Args:
            eeg: EEG tensor ``(batch, 17, 100)``.

        Returns:
            ℓ2-normalised embedding of shape ``(batch, 2304)``.
        """
        z1, z2, z3 = self.scale_encoder(eeg, self.eeg_encoder)
        z = torch.cat([z1, z2, z3], dim=1)  # (batch, 2304)
        return F.normalize(z, dim=1)
