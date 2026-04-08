"""LLM-based EEG encoder for the decoupled encoder → head pipeline.

Design
------
A pre-trained causal LLM (e.g. Qwen2.5-0.5B, or any HuggingFace AutoModel)
is used as a frozen (or fine-tunable) sequence encoder.  EEG time-series are
split into temporal patches, projected into the LLM's hidden dimension, then
passed through the transformer body.  To obtain a pooled representation we
append a learnable aggregation token **at the end** of the sequence.  With
causal attention the last token attends to every preceding patch token,
acting as a full-context summary vector.

  (B, C, T)
    → PatchEmbed (Conv1d)           → (B, n_patches, patch_embed_dim)
    → input_proj (Linear)           → (B, n_patches, llm_hidden_dim)
    → append AGG token              → (B, n_patches+1, llm_hidden_dim)
    → LLM transformer body          → (B, n_patches+1, llm_hidden_dim)
    → AGG token [:, -1]             → (B, llm_hidden_dim)   ← latent space

A downstream head (linear or MLP) is built separately via
``build_jepa_downstream`` and trained on top, mirroring the JEPA / EEG
Transformer decoupled pipeline.

  data → LLMEEGEncoder → latent space → downstream head → predict

Swapping the backbone only requires changing ``model.pretrained_name`` in the
Hydra config — no code changes needed.

LoRA mode
---------
When ``model.lora: true``, the backbone is wrapped with ``peft.get_peft_model``
using a ``LoraConfig``.  Only the injected low-rank matrices (≈ 0.5–1 % of
params) plus the EEG front-end train.  The rest of the backbone stays frozen.
This gives better adaptation than a fully frozen backbone at a fraction of the
cost of full fine-tuning.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from model.jepa import PatchEmbed

_PEFT_AVAILABLE = False
try:
    from peft import LoraConfig, TaskType, get_peft_model
    _PEFT_AVAILABLE = True
except ImportError:
    pass


class LLMEEGEncoder(nn.Module):
    def __init__(self, cfg, n_channels: int = 62, seq_len: int = 400):
        super().__init__()
        pretrained_name = str(cfg.model.pretrained_name)
        patch_len       = int(cfg.model.patch_len)
        patch_embed_dim = int(cfg.model.patch_embed_dim)
        n_channels      = int(cfg.model.get("n_channels", n_channels))
        seq_len         = int(cfg.model.get("seq_len", seq_len))

        # ── LLM backbone (transformer body only, no LM head) ───────────────
        llm_cfg = AutoConfig.from_pretrained(pretrained_name)
        backbone = AutoModel.from_pretrained(pretrained_name)
        hidden_dim = llm_cfg.hidden_size
        self.embed_dim = hidden_dim  # downstream head reads this

        # ── Optional LoRA wrapping ──────────────────────────────────────────
        self.lora_enabled = bool(cfg.model.get("lora", False))
        if self.lora_enabled:
            if not _PEFT_AVAILABLE:
                raise ImportError("peft is required for LoRA mode: uv add peft")
            lora_r       = int(cfg.model.get("lora_r", 8))
            lora_alpha   = int(cfg.model.get("lora_alpha", 16))
            lora_dropout = float(cfg.model.get("lora_dropout", 0.05))
            target_modules = list(cfg.model.get("lora_target_modules", ["q_proj", "v_proj"]))
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            backbone = get_peft_model(backbone, lora_cfg)
            backbone.print_trainable_parameters()
        self.backbone = backbone

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

        # Pass as inputs_embeds to bypass the LLM's token embedding table.
        # attention_mask = all ones → no padding mask (causal mask applied internally).
        attn_mask = torch.ones(B, tokens.size(1), dtype=torch.long, device=x.device)
        out = self.backbone(inputs_embeds=tokens, attention_mask=attn_mask)
        return out.last_hidden_state[:, -1].contiguous()          # (B, hidden_dim)

    # ------------------------------------------------------------------
    def freeze_backbone(self):
        """Freeze LLM weights; keep patch_embed, input_proj, agg_token trainable.

        In LoRA mode peft already froze non-LoRA params at init — this is a no-op
        for those weights so LoRA matrices stay trainable.
        """
        for name, p in self.backbone.named_parameters():
            if self.lora_enabled and "lora_" in name:
                continue  # keep LoRA matrices trainable
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze LLM weights for end-to-end fine-tuning.

        In LoRA mode this unfreezes only LoRA matrices (peft keeps base weights frozen
        unless you call ``merge_and_unload`` first).
        """
        if self.lora_enabled:
            for name, p in self.backbone.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True

    def frontend_parameters(self):
        """Return only the trainable front-end parameters (patch_embed, input_proj, agg_token).

        Used when the LLM backbone is frozen but the EEG adapter should still
        receive gradients (fine_tune_llm=false, linear_probe=false).
        """
        return (
            list(self.patch_embed.parameters())
            + list(self.input_proj.parameters())
            + [self.agg_token]
        )
