import torch
import torch.nn as nn


class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return x + self.f(x)


class TemporalPatchEncoder(nn.Module):
    """Split EEG into temporal windows and encode each independently.

    Each window captures a different temporal phase of visual processing.
    Early windows align to shallow image layers (local features).
    Late windows align to deep image layers (semantic features).

    Args:
        n_channels:    int   = 17    EEG channels
        n_timepoints:  int   = 100   total timepoints
        n_windows:     int   = 4     number of temporal patches
        feature_dim:   int   = 512   output dim per window
        dropout:       float = 0.3
    """

    def __init__(
        self,
        n_channels: int = 17,
        n_timepoints: int = 100,
        n_windows: int = 4,
        feature_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        assert n_timepoints % n_windows == 0, (
            f"n_timepoints {n_timepoints} must be divisible by n_windows {n_windows}"
        )

        self.n_windows = n_windows
        self.window_size = n_timepoints // n_windows   # 25
        self.window_dim = n_channels * self.window_size   # 17 * 25 = 425

        # one MLP encoder per window — separate weights
        # so each window can specialise on its temporal phase
        self.window_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.window_dim, feature_dim),
                ResidualAdd(nn.Sequential(
                    nn.GELU(),
                    nn.Linear(feature_dim, feature_dim),
                    nn.Dropout(dropout),
                )),
                nn.LayerNorm(feature_dim),
            )
            for _ in range(n_windows)
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Encode each temporal window independently.

        Args:
            x: (batch, n_channels, n_timepoints)

        Returns:
            List of n_windows tensors, each (batch, feature_dim)
        """
        # split along time axis: (batch, C, T) → list of (batch, C, T/W)
        windows = x.split(self.window_size, dim=2)

        embeddings = []
        for window, encoder in zip(windows, self.window_encoders):
            # flatten: (batch, C, T/W) → (batch, C*T/W)
            flat = window.reshape(window.shape[0], self.window_dim)
            embeddings.append(encoder(flat))   # (batch, feature_dim)

        return embeddings   # list of n_windows × (batch, feature_dim)


# Keep EEGNetEncoder name for interface compatibility
EEGNetEncoder = TemporalPatchEncoder
