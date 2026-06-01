from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders.eegnet_encoder import EEGNetEncoder


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
# SUPAEEG — full model
# ---------------------------------------------------------------------------


class SUPAEEG(nn.Module):
    """SUPAEEG: Scale-conditioned Unified Predictive Alignment for EEG.

    Full model combining EEGNetEncoder + MultiScaleEncoder.

    The model maps raw EEG trials to three depth-scale embeddings (z1, z2, z3)
    that are trained to align with frozen CLIP features (S1/S2/S3) via a
    symmetric InfoNCE loss (computed externally in the training loop).

    Key design decisions
    --------------------
    - At test time only the learned EEG encoder is required.
    - No image encoder or generative model is needed at inference.
    - CLIP features are used only as fixed training targets, passed in by the
      training loop — this model never touches the feature lookup directly.
    - c1/c2/c3 are nn.Parameters (not I-JEPA features).

    Args:
        d_model:         Token embedding dimension.  Default: 256.
        nhead:           Transformer attention heads.  Default: 8.
        num_layers:      Transformer encoder depth.  Default: 4.
        dim_feedforward: Feed-forward hidden size.  Default: 512.
        dropout:         Dropout probability.  Default: 0.1.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a batch of EEG trials into three scale embeddings.

        Args:
            eeg: EEG tensor ``(batch, 17, 100)``.

        Returns:
            ``(z1, z2, z3)`` each of shape ``(batch, 768)``.
        """
        return self.scale_encoder(eeg, self.eeg_encoder)

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
