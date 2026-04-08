"""EEG Transformer encoder for the decoupled encoder → head pipeline.

Architecture:
  (B, C, T)
    → PatchEmbed (Conv1d)           → (B, n_patches, embed_dim)
    → prepend CLS token             → (B, 1+n_patches, embed_dim)
    → learned positional embedding
    → TransformerEncoder            → (B, 1+n_patches, embed_dim)
    → CLS token [:, 0]              → (B, embed_dim)   ← latent space

A downstream classification head (linear or MLP) is built separately via
``build_jepa_downstream`` and trained on top of these frozen or unfrozen features,
mirroring the JEPA decoupled pipeline:

  data → EEGTransformer (encoder) → latent space → downstream head → predict
"""

import torch
import torch.nn as nn

from model.jepa import PatchEmbed, TransformerEncoder


class EEGTransformer(nn.Module):
    def __init__(self, cfg, n_channels: int = 62, seq_len: int = 400):
        super().__init__()
        patch_len  = int(cfg.model.patch_len)
        embed_dim  = int(cfg.model.embed_dim)
        depth      = int(cfg.model.depth)
        num_heads  = int(cfg.model.num_heads)
        mlp_ratio  = float(cfg.model.mlp_ratio)
        dropout    = float(cfg.model.dropout)
        n_channels = int(cfg.model.get("n_channels", n_channels))
        seq_len    = int(cfg.model.get("seq_len", seq_len))
        n_patches  = seq_len // patch_len

        self.embed_dim   = embed_dim
        self.patch_embed = PatchEmbed(n_channels, patch_len, embed_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.encoder     = TransformerEncoder(embed_dim, depth, num_heads, mlp_ratio, dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return CLS-token feature vector (B, embed_dim) for downstream use."""
        B = x.size(0)
        tokens = self.patch_embed(x)                              # (B, n_patches, D)
        cls    = self.cls_token.expand(B, -1, -1)                 # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)                  # (B, 1+n_patches, D)
        tokens = tokens + self.pos_embed
        tokens = self.encoder(tokens)                             # (B, 1+n_patches, D)
        return tokens[:, 0].contiguous()                          # (B, embed_dim)

    def freeze_backbone(self):
        """Freeze the encoder for linear probing."""
        for module in (self.patch_embed, self.encoder):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze the encoder for end-to-end fine-tuning."""
        for module in (self.patch_embed, self.encoder):
            for p in module.parameters():
                p.requires_grad = True
        self.cls_token.requires_grad = True
        self.pos_embed.requires_grad = True
