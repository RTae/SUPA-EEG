from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders.eegnet_encoder import EEGNetEncoder


class SUPAEEG(nn.Module):
    """SUPAEEG with EEGProject backbone.

    Architecture:
        EEG (batch, 17, 100)
          down
        EEGNetEncoder  MLP  -> (batch, 1024)
          down
        |- proj1 Linear(1024->512->768) -> z1  aligned to S1 local
        |- proj2 Linear(1024->512->768) -> z2  aligned to S2 spatial
        '- proj3 Linear(1024->512->768) -> z3  aligned to S3 semantic

    At inference: embed() -> concat(z1,z2,z3) -> l2-normalise -> (batch, 2304)

    Args:
        n_channels:   int   = 17
        n_timepoints: int   = 100
        feature_dim:  int   = 1024   EEGProject hidden dim
        d_visual:     int   = 768    CLIP feature dim
        dropout:      float = 0.3
    """

    def __init__(self, n_channels=17, n_timepoints=100,
                 feature_dim=1024, d_visual=768, dropout=0.3):
        super().__init__()

        self.eeg_encoder = EEGNetEncoder(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            feature_dim=feature_dim,
            dropout=dropout,
        )

        self.proj1 = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.GELU(), nn.Linear(512, d_visual)
        )
        self.proj2 = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.GELU(), nn.Linear(512, d_visual)
        )
        self.proj3 = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.GELU(), nn.Linear(512, d_visual)
        )

    def forward(self, eeg):
        # eeg: (batch, 17, 100)
        z = self.eeg_encoder(eeg)   # (batch, 1024)
        z1 = self.proj1(z)          # (batch, 768)
        z2 = self.proj2(z)          # (batch, 768)
        z3 = self.proj3(z)          # (batch, 768)
        return z1, z2, z3

    @torch.no_grad()
    def embed(self, eeg):
        """Inference only. Returns l2-normalised (batch, 2304) descriptor."""
        z1, z2, z3 = self.forward(eeg)
        z = torch.cat([z1, z2, z3], dim=1)
        return F.normalize(z, dim=1)


if __name__ == "__main__":
    model = SUPAEEG()
    eeg = torch.randn(4, 17, 100)
    z1, z2, z3 = model(eeg)
    assert z1.shape == (4, 768)
    assert z2.shape == (4, 768)
    assert z3.shape == (4, 768)
    emb = model.embed(eeg)
    assert emb.shape == (4, 2304)
    print("All shapes correct")
