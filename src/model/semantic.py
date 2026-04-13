import torch
import torch.nn as nn


class SemanticModel(nn.Module):
    """
    
    The online branch is optimized with supervised objectives (cross-entropy +
    triplet), while the target branch is updated by EMA for JEPA consistency.
    """

    def __init__(self, cfg, num_classes: int) -> None:
        super().__init__()
        model_cfg = cfg.model

        n_channels = int(model_cfg.get("n_channels", 62))
        seq_len = int(model_cfg.get("seq_len", 400))
        patch_len = int(model_cfg.get("patch_len", 20))
        embed_dim = int(model_cfg.get("embed_dim", 256))
        num_heads = int(model_cfg.get("num_heads", 8))
        depth = int(model_cfg.get("depth", 4))
        dropout = float(model_cfg.get("dropout", 0.1))
        proj_dim = int(model_cfg.get("projection_dim", 128))

        if seq_len % patch_len != 0:
            raise ValueError("model.seq_len must be divisible by model.patch_len")

        num_patches = seq_len // patch_len
        self.patch_len = patch_len
        self.embed_dim = embed_dim

        # Backbone patchifier: (B, C, T) -> (B, P, D)
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

        self.projection_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

        # JEPA predictor and EMA target encoder.
        self.predictor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.target_encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self._init_target_encoder()

    def _init_target_encoder(self) -> None:
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def update_target_encoder(self, ema_decay: float) -> None:
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_encoder.parameters(), self.online_encoder.parameters(), strict=True
            ):
                target_param.data.mul_(ema_decay).add_(online_param.data, alpha=1.0 - ema_decay)

    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        tokens = self.patch_embed(x).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)
        return tokens

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._encode_tokens(x)
        encoded = self.online_encoder(tokens)
        return encoded[:, 0]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self._encode_tokens(x)
        online_encoded = self.online_encoder(tokens)
        cls_online = online_encoded[:, 0]

        with torch.no_grad():
            target_encoded = self.target_encoder(tokens)
            cls_target = target_encoded[:, 0]

        logits = self.classifier(cls_online)
        embedding = self.projection_head(cls_online)
        jepa_pred = self.predictor(cls_online)
        return {
            "logits": logits,
            "embedding": embedding,
            "jepa_pred": jepa_pred,
            "jepa_target": cls_target,
        }
