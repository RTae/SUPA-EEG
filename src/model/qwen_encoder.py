"""Qwen-based EEG encoder for the decoupled encoder → head pipeline.

Design
------
Qwen2 is a causal (decoder-only) transformer.  To obtain a meaningful
pooled representation we append a learnable aggregation token **at the end**
of the sequence.  With causal attention the last token attends to every
preceding patch token, acting as a full-context summary vector.

  (B, C, T)
    → PatchEmbed (Conv1d)           → (B, n_patches, patch_embed_dim)
    → input_proj (Linear)           → (B, n_patches, qwen_hidden_dim)
    → append AGG token              → (B, n_patches+1, qwen_hidden_dim)
    → Qwen2 transformer body        → (B, n_patches+1, qwen_hidden_dim)
    → AGG token [:, -1]             → (B, qwen_hidden_dim)   ← latent space

A downstream head (linear or MLP) is built separately via
``build_jepa_downstream`` and trained on top, mirroring the JEPA / EEG
Transformer decoupled pipeline.

  data → QwenEEGEncoder → latent space → downstream head → predict
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from model.jepa import PatchEmbed


class QwenEEGEncoder(nn.Module):
    def __init__(self, cfg, n_channels: int = 62, seq_len: int = 400):
        super().__init__()
        pretrained_name = str(cfg.model.pretrained_name)
        patch_len       = int(cfg.model.patch_len)
        patch_embed_dim = int(cfg.model.patch_embed_dim)
        n_channels      = int(cfg.model.get("n_channels", n_channels))
        seq_len         = int(cfg.model.get("seq_len", seq_len))

        # ── Qwen backbone (transformer body only, no LM head) ──────────────
        qwen_cfg = AutoConfig.from_pretrained(pretrained_name)
        self.backbone = AutoModel.from_pretrained(pretrained_name)
        hidden_dim = qwen_cfg.hidden_size
        self.embed_dim = hidden_dim  # downstream head reads this

        # ── EEG front-end ──────────────────────────────────────────────────
        self.patch_embed = PatchEmbed(n_channels, patch_len, patch_embed_dim)
        self.input_proj  = nn.Linear(patch_embed_dim, hidden_dim)

        # Aggregation token (appended last so causal attn gives full context)
        self.agg_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.agg_token, std=0.02)

        # Lightweight projection to align input_proj weights
        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return aggregation-token feature (B, hidden_dim) for downstream use."""
        B = x.size(0)
        tokens = self.patch_embed(x)                              # (B, n_patches, patch_embed_dim)
        tokens = self.input_proj(tokens)                          # (B, n_patches, hidden_dim)
        agg    = self.agg_token.expand(B, -1, -1)                 # (B, 1, hidden_dim)
        tokens = torch.cat([tokens, agg], dim=1)                  # (B, n_patches+1, hidden_dim)

        # Pass as inputs_embeds to bypass Qwen's token embedding table.
        # attention_mask = all ones → no padding mask (causal mask applied internally).
        attn_mask = torch.ones(B, tokens.size(1), dtype=torch.long, device=x.device)
        out = self.backbone(inputs_embeds=tokens, attention_mask=attn_mask)
        return out.last_hidden_state[:, -1].contiguous()          # (B, hidden_dim)

    # ------------------------------------------------------------------
    def freeze_backbone(self):
        """Freeze Qwen weights; keep patch_embed, input_proj, agg_token trainable."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze Qwen weights for end-to-end fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True
