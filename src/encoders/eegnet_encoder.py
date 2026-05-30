"""EEGNet-style spatiotemporal tokenizer for EEG signals.

Implements a depthwise-separable convolutional backbone that converts raw EEG
channel × time tensors into a sequence of spatiotemporal tokens ready for
downstream transformer processing.

Reference architecture:
    Lawhern et al. (2018) "EEGNet: A Compact Convolutional Neural Network for
    EEG-based Brain–Computer Interfaces"
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGNetEncoder(nn.Module):
    """EEGNet-style spatiotemporal tokenizer for EEG signals.

    Converts raw multi-channel EEG into a sequence of rich spatiotemporal
    tokens via temporal convolution → depthwise convolution → separable
    convolution, then projects each token to a fixed ``d_model``-dimensional
    embedding with a learnable positional encoding.

    Args:
        n_channels:   Number of EEG channels.  Default: 17.
        n_timepoints: Number of time samples per trial.  Default: 100.
        dropout:      Dropout probability applied after the final pooling.
                      Default: 0.25.

    Input:
        ``(batch, n_channels, n_timepoints)``

    Output:
        ``(batch, N_t, 256)`` spatiotemporal tokens, where ``N_t`` is the
        temporal dimension after average pooling (computed automatically).
    """

    def __init__(
        self,
        n_channels: int = 17,
        n_timepoints: int = 100,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Temporal convolution block
        # ------------------------------------------------------------------
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(1, 25),
            padding=(0, 12),
            bias=False,
        )
        self.temporal_bn = nn.BatchNorm2d(16)
        self.temporal_elu = nn.ELU()

        # ------------------------------------------------------------------
        # Depthwise (spatial) convolution block
        # ------------------------------------------------------------------
        self.depthwise_conv = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(n_channels, 1),
            groups=16,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(32)
        self.depthwise_elu = nn.ELU()

        # ------------------------------------------------------------------
        # Separable convolution block
        # ------------------------------------------------------------------
        self.separable_conv = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(1, 15),
            padding=(0, 7),
            bias=False,
        )
        self.separable_bn = nn.BatchNorm2d(32)
        self.separable_elu = nn.ELU()
        self.avgpool = nn.AvgPool2d((1, 8))
        self.dropout = nn.Dropout(dropout)

        # ------------------------------------------------------------------
        # Compute N_t (temporal token count) via a dummy forward pass
        # ------------------------------------------------------------------
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            x = self.temporal_elu(self.temporal_bn(self.temporal_conv(dummy)))
            x = self.depthwise_elu(self.depthwise_bn(self.depthwise_conv(x)))
            x = self.separable_elu(self.separable_bn(self.separable_conv(x)))
            x = self.avgpool(x)
            # shape: (1, 32, 1, N_t)
            self.n_t: int = x.shape[-1]

        # ------------------------------------------------------------------
        # Token projection and positional encoding
        # ------------------------------------------------------------------
        self.proj = nn.Linear(32, 256)
        # Learnable positional encoding: shape (1, N_t, 256)
        self.pos_enc = nn.Parameter(torch.randn(1, self.n_t, 256) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map raw EEG to spatiotemporal token sequence.

        Args:
            x: EEG tensor of shape ``(batch, n_channels, n_timepoints)``.

        Returns:
            Token tensor of shape ``(batch, N_t, 256)``.
        """
        # (batch, n_channels, n_timepoints) -> (batch, 1, n_channels, n_timepoints)
        x = x.unsqueeze(1)

        # Temporal conv block
        x = self.temporal_elu(self.temporal_bn(self.temporal_conv(x)))

        # Depthwise conv block
        x = self.depthwise_elu(self.depthwise_bn(self.depthwise_conv(x)))

        # Separable conv block
        x = self.separable_elu(self.separable_bn(self.separable_conv(x)))
        x = self.dropout(self.avgpool(x))
        # x: (batch, 32, 1, N_t)

        # Reshape to token sequence: (batch, N_t, 32)
        x = x.squeeze(2).permute(0, 2, 1)

        # Project to d_model=256 and add positional encoding
        x = self.proj(x) + self.pos_enc  # (batch, N_t, 256)
        return x
