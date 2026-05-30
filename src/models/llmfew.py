"""LLMFew: LLM-enhanced framework for few-shot multivariate time series classification.


Input shape expected: (B, M, L) — same time-domain layout the rest of the project
uses for EEGNet (62 channels x 400 timesteps by default).
"""

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# Register Ascend NPU so `.to("npu")` works when this module is imported
# in isolation (e.g. inside a notebook or an ad-hoc script).
try:  # pragma: no cover - environment-dependent
    import torch_npu  # noqa: F401
except ImportError:
    pass


# -----------------------------------------------------------------------------
# Causal-convolution building blocks
# -----------------------------------------------------------------------------
class CausalConv1d(nn.Module):
    """1D causal convolution with weight normalization.

    Left-pads the input so the output length equals the input length and no
    information from the future leaks into the current time step.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self._left_pad = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self._left_pad, 0))
        return self.conv(x)


class CausalLayer(nn.Module):
    """CausalConv -> Norm (via weight_norm inside the conv) -> LeakyReLU (paper Eq. 2)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.conv(x), negative_slope=0.01)


class ConvBlock(nn.Module):
    """ConvBlock = two stacked CausalLayers sharing the same dilation (paper Eq. 2).

    The residual connection (+ H(d-1)) is added outside this block (in PTCEnc)
    to keep the block definition identical to the paper.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer1 = CausalLayer(channels, channels, kernel_size, dilation)
        self.layer2 = CausalLayer(channels, channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(self.layer1(x))
        y = self.dropout(self.layer2(y))
        return y


class PTCEnc(nn.Module):
    """Patch-wise Temporal Convolution Encoder.

    Input:  (B, in_dim, NP) — each patch position is a token with `in_dim`
            features (we set in_dim = M * P after patching).
    Output: (B, NP, d_llm) — NP tokens aligned with the LLM embedding dimension.
    """

    def __init__(
        self,
        in_dim: int,
        channel_size: int,
        depth: int,
        kernel_size: int,
        d_llm: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # 1x1 projection to lift (M*P) -> channel_size before the dilated stack.
        self.input_proj = nn.Conv1d(in_dim, channel_size, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                ConvBlock(channel_size, kernel_size, dilation=2 ** d, dropout=dropout)
                for d in range(depth)
            ]
        )
        # "Sampling Conv" (paper) that aligns the encoder output with the LLM embedding dim.
        self.sampling_conv = nn.Conv1d(channel_size, d_llm, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            # Paper Eq. 2: H(d) = ConvBlock(H(d-1)) + H(d-1)
            h = block(h) + h
        h = self.sampling_conv(h)           # (B, d_llm, NP)
        return h.transpose(1, 2).contiguous()  # (B, NP, d_llm)


# -----------------------------------------------------------------------------
# Patching helper
# -----------------------------------------------------------------------------
def _pad_and_patch(x: torch.Tensor, patch_length: int, stride: int) -> torch.Tensor:
    """Replicate-pad the tail by `stride` values, then unfold along time.

    Input :  (B, M, L)
    Output:  (B, M, P, NP) where NP = (L - P) // S + 2 (paper Eq. 1).
    """
    last = x[..., -1:].expand(-1, -1, stride)
    x = torch.cat([x, last], dim=-1)  # (B, M, L + S)
    patches = x.unfold(dimension=-1, size=patch_length, step=stride)  # (B, M, NP, P)
    return patches.transpose(-1, -2).contiguous()  # (B, M, P, NP)


# -----------------------------------------------------------------------------
# Default LoRA target modules per LLM family
# -----------------------------------------------------------------------------
_DEFAULT_LORA_TARGETS = {
    "gpt2": ["c_attn"],                          # combined Q/K/V projection
    "qwen": ["q_proj", "k_proj", "v_proj"],
    "llama": ["q_proj", "k_proj", "v_proj"],
    "phi": ["qkv_proj"],                         # Phi-3 uses a fused qkv_proj
    "default": ["q_proj", "k_proj", "v_proj"],
}


def _resolve_lora_targets(llm_name: str, override: Iterable[str] | None) -> List[str]:
    if override:
        return list(override)
    name = llm_name.lower()
    for key, targets in _DEFAULT_LORA_TARGETS.items():
        if key in name:
            return list(targets)
    return list(_DEFAULT_LORA_TARGETS["default"])


# -----------------------------------------------------------------------------
# LLMFew model
# -----------------------------------------------------------------------------
class LLMFew(nn.Module):
    """Full LLMFew stack: Patching -> PTCEnc -> LLM(LoRA) -> classification head."""

    def __init__(self, cfg, num_classes: int) -> None:
        super().__init__()
        m_cfg = cfg.model

        # --- Shape / patching hyperparameters (paper Appendix C defaults) ------
        self.num_channels = int(m_cfg.get("num_channels", 62))
        self.sequence_length = int(m_cfg.get("sequence_length", 400))
        self.patch_length = int(m_cfg.get("patch_length", 32))
        self.stride = int(m_cfg.get("stride", 16))
        self.num_patches = (self.sequence_length - self.patch_length) // self.stride + 2

        # --- LLM backbone ------------------------------------------------------
        # Imported lazily so users without `transformers` can still import the
        # module for shape-only testing.
        from transformers import AutoConfig, AutoModel

        llm_name = str(m_cfg.get("llm_name", "gpt2"))
        llm_config = AutoConfig.from_pretrained(llm_name)
        self.d_llm = getattr(llm_config, "hidden_size", None) or getattr(llm_config, "n_embd")
        base_llm = AutoModel.from_pretrained(llm_name)

        # --- LoRA fine-tuning on attention Q/K/V -------------------------------
        from peft import LoraConfig, TaskType, get_peft_model

        target_modules = _resolve_lora_targets(
            llm_name, m_cfg.get("lora_target_modules", None)
        )
        lora_config = LoraConfig(
            r=int(m_cfg.get("lora_r", 8)),
            lora_alpha=int(m_cfg.get("lora_alpha", 32)),
            lora_dropout=float(m_cfg.get("lora_dropout", 0.1)),
            bias="none",
            target_modules=target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.llm = get_peft_model(base_llm, lora_config)

        # --- PTCEnc: takes (B, M*P, NP) -> (B, NP, d_llm) ----------------------
        self.ptcenc = PTCEnc(
            in_dim=self.num_channels * self.patch_length,
            channel_size=int(m_cfg.get("channel_size", 256)),
            depth=int(m_cfg.get("depth", 3)),
            kernel_size=int(m_cfg.get("kernel_size", 3)),
            d_llm=self.d_llm,
            dropout=float(m_cfg.get("encoder_dropout", 0.1)),
        )

        # --- Classification head (paper Eq. 4) ---------------------------------
        self.flat_dim = self.num_patches * self.d_llm
        self.classifier = nn.Linear(self.flat_dim, num_classes)
        self.output_norm = nn.LayerNorm(num_classes)

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, 1, M, L) or (B, M, L).
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() != 3:
            raise ValueError(f"LLMFew expects (B, M, L), got shape {tuple(x.shape)}")

        # 1) Patching -> (B, M, P, NP)
        xp = _pad_and_patch(x, self.patch_length, self.stride)
        b, m, p, np_ = xp.shape

        # 2) Flatten channels with patch length so each of NP positions becomes a
        #    (M*P)-dim vector that PTCEnc will project to the LLM embedding dim.
        h = xp.reshape(b, m * p, np_)

        # 3) PTCEnc -> (B, NP, d_llm)
        he = self.ptcenc(h)

        # 4) LLM decoder with LoRA -> (B, NP, d_llm)
        llm_out = self.llm(inputs_embeds=he)
        hd = llm_out.last_hidden_state

        # 5) Fuse with skip connection then flatten + classify.
        fused = F.relu(he + hd)
        logits = self.classifier(fused.reshape(b, -1))
        return self.output_norm(logits)  # Cross-entropy expects logits.

    # --------------------------------------------------------- utility helpers
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
