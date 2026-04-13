import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticModel(nn.Module):
    """Semantic EEG encoder with a switchable backbone for metric learning.

    Supports three backbones via ``model.backbone``: ``transformer``, ``jepa``,
    and ``nn``. See README for architecture diagrams.
    """

    def __init__(self, cfg, num_classes: int) -> None:
        _ = num_classes
        super().__init__()
        model_cfg = cfg.model

        self.backbone = str(model_cfg.get("backbone", "transformer")).lower()
        n_channels = int(model_cfg.get("n_channels", 62))
        seq_len = int(model_cfg.get("seq_len", 400))
        patch_len = int(model_cfg.get("patch_len", 20))
        embed_dim = int(model_cfg.get("embed_dim", 256))
        num_heads = int(model_cfg.get("num_heads", 8))
        depth = int(model_cfg.get("depth", 4))
        dropout = float(model_cfg.get("dropout", 0.1))
        proj_dim = int(model_cfg.get("projection_dim", 128))
        nn_hidden_dim = int(model_cfg.get("nn_hidden_dim", embed_dim))

        if self.backbone in {"transformer", "jepa"} and seq_len % patch_len != 0:
            raise ValueError("model.seq_len must be divisible by model.patch_len")

        self.patch_len = patch_len
        self.embed_dim = nn_hidden_dim if self.backbone == "nn" else embed_dim

        self.patch_embed = None
        self.cls_token = None
        self.pos_embed = None
        self.pos_drop = None
        self.online_encoder = None
        self.target_encoder = None

        if self.backbone in {"transformer", "jepa"}:
            num_patches = seq_len // patch_len
            self.patch_embed = nn.Conv1d(
                in_channels=n_channels,
                out_channels=embed_dim,
                kernel_size=patch_len,
                stride=patch_len,
                bias=False,
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.pos_drop = nn.Dropout(dropout)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.online_encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
            if self.backbone == "jepa":
                self.target_encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
                self._init_target_encoder()
        elif self.backbone == "nn":
            self.online_encoder = nn.Sequential(
                nn.Conv1d(n_channels, nn_hidden_dim, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(nn_hidden_dim),
                nn.GELU(),
                nn.Conv1d(nn_hidden_dim, nn_hidden_dim, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(nn_hidden_dim),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
        else:
            raise ValueError("model.backbone must be one of: 'jepa', 'transformer', 'nn'")

        self.projection_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, proj_dim),
        )

    def _init_target_encoder(self) -> None:
        if self.target_encoder is None or self.online_encoder is None:
            return
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def update_target_encoder(self, ema_decay: float) -> None:
        if self.target_encoder is None or self.online_encoder is None:
            return
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_encoder.parameters(), self.online_encoder.parameters(), strict=True
            ):
                target_param.data.mul_(ema_decay).add_(online_param.data, alpha=1.0 - ema_decay)

    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        if self.patch_embed is None or self.cls_token is None or self.pos_embed is None or self.pos_drop is None:
            raise RuntimeError("Token encoding is only available for transformer-style backbones")
        tokens = self.patch_embed(x).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)
        return tokens

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone in {"transformer", "jepa"}:
            tokens = self._encode_tokens(x)
            encoded = self.online_encoder(tokens)
            return encoded[:, 0]

        encoded = self.online_encoder(x).squeeze(-1)
        return encoded

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode(x)
        embedding = F.normalize(self.projection_head(features), dim=1)
        return {
            "features": features,
            "embedding": embedding,
        }
