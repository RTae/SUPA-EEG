import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders.eegnet_encoder import EEGNetEncoder


class SUPAEEG(nn.Module):
    """SUPAEEG: EEGProject + shared encoder alignment.

    Architecture:
        EEG (batch, 17, 100)
          eeg_encoder  -> (batch, 1024)
          eeg_projector Linear(1024, 512)
          share_encoder Linear(512, 512)   <- shared with image side
          l2-normalize  -> zE (batch, 512)

        image_layers (batch, 5, 3200)
          .float().mean(dim=1) -> (batch, 3200)
          img_pre_projector Linear(3200, 1024)
          img_projector     Linear(1024, 512)
          share_encoder     Linear(512, 512)   <- SAME nn.Module as EEG
          l2-normalize      -> zI (batch, 512)

    Args:
        n_channels:      int   = 17
        n_timepoints:    int   = 100
        eeg_feature_dim: int   = 1024
        image_input_dim: int   = 3200
        image_mid_dim:   int   = 1024
        feature_dim:     int   = 512
        dropout:         float = 0.3
    """

    def __init__(self, n_channels=17, n_timepoints=100,
                 eeg_feature_dim=1024, image_input_dim=3200,
                 image_mid_dim=1024, feature_dim=512, dropout=0.3):
        super().__init__()
        self.eeg_encoder       = EEGNetEncoder(n_channels, n_timepoints,
                                               eeg_feature_dim, dropout)
        self.eeg_projector     = nn.Linear(eeg_feature_dim, feature_dim)
        self.img_pre_projector = nn.Linear(image_input_dim, image_mid_dim)
        self.img_projector     = nn.Linear(image_mid_dim, feature_dim)
        self.share_encoder     = nn.Linear(feature_dim, feature_dim)
        self.logit_scale       = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        )

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        x = self.eeg_encoder(eeg)
        x = self.eeg_projector(x)
        x = self.share_encoder(x)
        return F.normalize(x, dim=1)   # (batch, 512)

    def encode_image(self, image_layers: torch.Tensor) -> torch.Tensor:
        x = image_layers.float().mean(dim=1)
        x = self.img_pre_projector(x)
        x = self.img_projector(x)
        x = self.share_encoder(x)
        return F.normalize(x, dim=1)   # (batch, 512)

    def forward(
        self, eeg: torch.Tensor, image_layers: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_eeg(eeg), self.encode_image(image_layers)

    @torch.no_grad()
    def embed(self, eeg: torch.Tensor) -> torch.Tensor:
        """Inference only. Returns l2-normalised (batch, 512) descriptor."""
        return self.encode_eeg(eeg)


if __name__ == "__main__":
    model = SUPAEEG()
    zE, zI = model(torch.randn(4, 17, 100), torch.randn(4, 5, 3200))
    assert zE.shape == (4, 512)
    assert zI.shape == (4, 512)
    assert model.embed(torch.randn(4, 17, 100)).shape == (4, 512)
    print("All shapes correct")
