import torch
import torch.nn as nn

from src.encoders.eegnet_encoder import EEGNetEncoder


class ConvPoolEncoder(nn.Module):
    """Small convolutional EEG encoder used by the bounded ablations."""

    def __init__(
        self,
        n_channels: int,
        n_timepoints: int,
        feature_dim: int,
        dropout: float,
        temporal_kernel: int = 15,
        use_transformer: bool = False,
        use_attention_pool: bool = False,
    ) -> None:
        super().__init__()
        kernel = min(temporal_kernel, n_timepoints if n_timepoints % 2 else n_timepoints - 1)
        kernel = max(kernel, 3)
        self.temporal = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, 96, 1, bias=False),
            nn.BatchNorm1d(96),
            nn.ELU(),
            nn.AvgPool1d(4, ceil_mode=True),
            nn.Dropout(dropout),
        )
        self.transformer = None
        if use_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model=96,
                nhead=4,
                dim_feedforward=192,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.attention_pool = nn.Linear(96, 1) if use_attention_pool else None
        self.project = nn.Sequential(nn.Linear(96, feature_dim), nn.LayerNorm(feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x).transpose(1, 2)
        if self.transformer is not None:
            x = self.transformer(x)
        if self.attention_pool is None:
            x = x.mean(dim=1)
        else:
            weights = self.attention_pool(x).softmax(dim=1)
            x = (x * weights).sum(dim=1)
        return self.project(x)


class EEGNetConvEncoder(nn.Module):
    """Compact EEGNet-style temporal/depthwise encoder."""

    def __init__(self, n_channels: int, feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, (1, 31), padding=(0, 15), bias=False),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4), ceil_mode=True),
            nn.Dropout(dropout),
            nn.Conv2d(32, 32, (1, 15), padding=(0, 7), groups=32, bias=False),
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.project = nn.Sequential(nn.Flatten(), nn.Linear(64, feature_dim), nn.LayerNorm(feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.features(x.unsqueeze(1)))


class TemporalCompressionEncoder(nn.Module):
    """Consume full-rate EEG, then compress time before the baseline projector."""

    def __init__(
        self,
        n_channels: int,
        n_timepoints: int,
        feature_dim: int,
        dropout: float,
        target_timepoints: int,
    ) -> None:
        super().__init__()
        if target_timepoints <= 0 or target_timepoints > n_timepoints:
            raise ValueError("target_timepoints must be in (0, n_timepoints]")
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, 15, padding=7, groups=n_channels, bias=False),
            nn.BatchNorm1d(n_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(target_timepoints),
        )
        self.projector = EEGNetEncoder(n_channels, target_timepoints, feature_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.stem(x))


def build_eeg_encoder(
    encoder_type: str,
    n_channels: int,
    n_timepoints: int,
    feature_dim: int,
    dropout: float,
    temporal_compression: int = 0,
) -> nn.Module:
    if temporal_compression:
        return TemporalCompressionEncoder(
            n_channels, n_timepoints, feature_dim, dropout, temporal_compression
        )
    if encoder_type == "eegproject":
        return EEGNetEncoder(n_channels, n_timepoints, feature_dim, dropout)
    if encoder_type == "eegnet":
        return EEGNetConvEncoder(n_channels, feature_dim, dropout)
    if encoder_type == "tsconv":
        return ConvPoolEncoder(n_channels, n_timepoints, feature_dim, dropout, temporal_kernel=31)
    if encoder_type == "eegconformer":
        return ConvPoolEncoder(
            n_channels, n_timepoints, feature_dim, dropout,
            temporal_kernel=15, use_transformer=True,
        )
    if encoder_type == "atm":
        return ConvPoolEncoder(
            n_channels, n_timepoints, feature_dim, dropout,
            temporal_kernel=15, use_attention_pool=True,
        )
    raise ValueError(
        "eeg_encoder_type must be one of "
        "{'eegproject', 'eegnet', 'tsconv', 'eegconformer', 'atm'}"
    )
