from __future__ import annotations

import torch.nn as nn


class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return x + self.f(x)


class EEGNetEncoder(nn.Module):
    """EEGProject-style MLP encoder.

    Keeps the class name EEGNetEncoder for interface compatibility
    with the rest of the codebase.

    Input:  (batch, n_channels, n_timepoints)
    Output: (batch, feature_dim)

    Args:
        n_channels:   int = 17
        n_timepoints: int = 100
        feature_dim:  int = 1024
        dropout:      float = 0.3
    """
    def __init__(self, n_channels=17, n_timepoints=100,
                 feature_dim=1024, dropout=0.3):
        super().__init__()

        self.input_dim = n_channels * n_timepoints   # 17 * 100 = 1700

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, feature_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.Dropout(dropout),
            )),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x):
        # x: (batch, n_channels, n_timepoints)
        x = x.reshape(x.shape[0], self.input_dim)   # flatten
        return self.model(x)                          # (batch, feature_dim)
