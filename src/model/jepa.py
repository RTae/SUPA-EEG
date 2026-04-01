"""EEG-JEPA: Joint Embedding Predictive Architecture for EEG signals.

Architecture overview:
  1. Patchify: split the EEG time-series into non-overlapping temporal patches.
  2. Context encoder (ViT-style): encode *visible* patches.
  3. Target encoder (EMA of context encoder): encode *masked* target patches.
  4. Predictor: predict target representations from context representations.
  5. Classifier head: linear probe on the [CLS] token for downstream classification.

Pre-training loss: MSE between predicted and target-encoder representations.
Fine-tuning: freeze encoders, train classifier head (or end-to-end).
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Project each temporal patch of shape (channels, patch_len) to a token."""

    def __init__(self, n_channels: int, patch_len: int, embed_dim: int):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Conv1d(n_channels, embed_dim, kernel_size=patch_len, stride=patch_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)  →  (B, embed_dim, n_patches)  →  (B, n_patches, embed_dim)
        return self.proj(x).transpose(1, 2).contiguous()


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(*[self.norm1(x)] * 3)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# JEPA
# ---------------------------------------------------------------------------

class EEGJEPA(nn.Module):
    """Joint Embedding Predictive Architecture for EEG.

    Args:
        n_channels:   Number of EEG channels (default 62).
        seq_len:      Temporal length of the input (default 400 for a 40–440 window).
        patch_len:    Length of each temporal patch (seq_len must be divisible).
        embed_dim:    Transformer embedding dimension.
        enc_depth:    Number of transformer blocks in the context/target encoder.
        pred_depth:   Number of transformer blocks in the predictor.
        num_heads:    Attention heads.
        mlp_ratio:    MLP expansion ratio inside transformer blocks.
        dropout:      Dropout rate.
        mask_ratio:   Fraction of patches masked for pre-training.
        ema_decay:    Exponential moving average decay for the target encoder.
        num_classes:  Number of downstream classes (classifier head).
    """

    def __init__(
        self,
        n_channels: int = 62,
        seq_len: int = 400,
        patch_len: int = 40,
        embed_dim: int = 128,
        enc_depth: int = 4,
        pred_depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        mask_ratio: float = 0.5,
        ema_decay: float = 0.996,
        num_classes: int = 40,
    ):
        super().__init__()
        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"
        self.n_patches = seq_len // patch_len
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay

        # ---------- patch embedding (shared weights for context & target) ----------
        self.patch_embed = PatchEmbed(n_channels, patch_len, embed_dim)

        # ---------- positional embedding ----------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # +1 for [CLS]
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))

        # ---------- context encoder ----------
        self.context_encoder = TransformerEncoder(embed_dim, enc_depth, num_heads, mlp_ratio, dropout)

        # ---------- target encoder (EMA copy — no grad) ----------
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.eval()

        # ---------- predictor ----------
        self.predictor = TransformerEncoder(embed_dim, pred_depth, num_heads, mlp_ratio, dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ---------- classifier head ----------
        self.classifier = nn.Linear(embed_dim, num_classes)

        self._init_weights()
        self._sync_target_encoder()

    def train(self, mode: bool = True):
        """Keep target encoder in eval mode while the rest of model may train."""
        super().train(mode)
        self.target_encoder.eval()
        return self

    # ------------------------------------------------------------------
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _sync_target_encoder(self):
        """Hard-sync target encoder with context encoder."""
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_target_encoder(self, ema_decay: float | None = None):
        """Momentum update: target_encoder ← ema_decay * target + (1-ema) * context."""
        decay = self.ema_decay if ema_decay is None else ema_decay
        for tp, cp in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            tp.mul_(decay).add_(cp, alpha=1.0 - decay)

    # ------------------------------------------------------------------
    # Masking helpers
    # ------------------------------------------------------------------
    def _random_mask(self, batch_size: int, device: torch.device):
        """Return (context_indices, target_indices) per sample — same mask across batch for simplicity."""
        del batch_size
        n_mask = max(1, int(self.n_patches * self.mask_ratio))
        perm = torch.randperm(self.n_patches, device=device)
        target_idx = perm[:n_mask]
        context_idx = perm[n_mask:]
        return context_idx.sort().values, target_idx.sort().values

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------
    def _embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → patch tokens (B, n_patches, D)."""
        return self.patch_embed(x).contiguous()

    def encode_context(self, tokens: torch.Tensor, context_idx: torch.Tensor) -> torch.Tensor:
        """Encode only the visible (context) patches + [CLS]."""
        B, _, D = tokens.shape
        ctx = tokens[:, context_idx]                                  # (B, n_ctx, D)
        ctx = ctx + self.pos_embed[:, context_idx + 1]                # +1 to skip CLS pos
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1]
        ctx = torch.cat([cls, ctx], dim=1)                            # (B, 1+n_ctx, D)
        return self.context_encoder(ctx)

    @torch.no_grad()
    def encode_target(self, tokens: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        """Encode the masked (target) patches with the EMA encoder — no grad."""
        self.target_encoder.eval()
        tgt = tokens[:, target_idx].contiguous() + self.pos_embed[:, target_idx + 1]
        return self.target_encoder(tgt)

    def predict_targets(self, context_out: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        """From context encoder output, predict representations at target positions."""
        B, _, D = context_out.shape
        n_tgt = target_idx.shape[0]
        mask_tokens = self.mask_token.expand(B, n_tgt, -1) + self.pos_embed[:, target_idx + 1]
        inp = torch.cat([context_out, mask_tokens], dim=1)            # (B, 1+n_ctx+n_tgt, D)
        out = self.predictor(inp)
        # Return only the mask-token positions (last n_tgt tokens).
        return out[:, -n_tgt:]

    # ------------------------------------------------------------------
    def pretrain_forward(self, x: torch.Tensor):
        """Pre-training forward pass. Returns (prediction, target) in latent space.

        Args:
            x: (B, C, T) raw EEG input.
        Returns:
            pred: (B, n_target, D) predicted target representations.
            target: (B, n_target, D) actual target representations (stop-grad).
        """
        tokens = self._embed_patches(x)
        context_idx, target_idx = self._random_mask(x.size(0), x.device)

        ctx_out = self.encode_context(tokens, context_idx)
        tgt_out = self.encode_target(tokens, target_idx)
        pred = self.predict_targets(ctx_out, target_idx)

        return pred, tgt_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Classification forward pass — encode all patches, classify from [CLS].

        Args:
            x: (B, C, T) raw EEG input.
        Returns:
            cls_features: (B, embed_dim).
        """
        tokens = self._embed_patches(x)
        all_idx = torch.arange(self.n_patches, device=x.device)
        ctx_out = self.encode_context(tokens, all_idx)
        cls_token = ctx_out[:, 0].contiguous()
        return cls_token

    def freeze_backbone(self):
        """Freeze patch embed + context encoder for linear probing."""
        for module in (self.patch_embed, self.context_encoder):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze patch embed + context encoder for end-to-end fine-tuning."""
        for module in (self.patch_embed, self.context_encoder):
            for p in module.parameters():
                p.requires_grad = True
        self.cls_token.requires_grad = True
        self.pos_embed.requires_grad = True


# ---------------------------------------------------------------------------
# Pre-training loss
# ---------------------------------------------------------------------------

def jepa_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 loss between predicted and target representations."""
    # Some backends/operators may produce non-contiguous tensors after indexing.
    # Flatten with reshape (safe for non-contiguous layouts) before loss.
    pred_flat = pred.reshape(-1, pred.shape[-1]).contiguous()
    target_flat = target.reshape(-1, target.shape[-1]).contiguous()
    return F.smooth_l1_loss(pred_flat, target_flat)
