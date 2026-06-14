import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders.eeg_ablation_encoders import build_eeg_encoder

_SHARE_ENCODER_TYPES = {"linear", "none", "separate", "transformer", "tokenized_cls", "jepa"}


class TransformerShareEncoder(nn.Module):
    """Shared encoder: treat the 512-d vector as a single token through a Transformer."""

    def __init__(self, feature_dim: int, n_layers: int = 2, nhead: int = 8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=nhead,
            dim_feedforward=feature_dim * 2,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.unsqueeze(1)).squeeze(1)


class TokenizedCLSEncoder(nn.Module):
    """Shared encoder: split 512-d into sub-tokens → ViT-style CLS pooling.

    The input vector is divided into n_tokens equal chunks, each projected
    to feature_dim. Learnable positional embeddings are added before the
    Transformer so token position carries information.
    """

    def __init__(self, feature_dim: int, n_tokens: int = 8, n_layers: int = 2, nhead: int = 8):
        super().__init__()
        if feature_dim % n_tokens != 0:
            raise ValueError(f"feature_dim ({feature_dim}) must be divisible by n_tokens ({n_tokens})")
        self.n_tokens = n_tokens
        token_dim = feature_dim // n_tokens
        self.token_proj = nn.Linear(token_dim, feature_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        # +1 for the CLS token position
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens + 1, feature_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=nhead,
            dim_feedforward=feature_dim * 2,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        tokens = x.view(batch, self.n_tokens, -1)       # (B, n_tokens, token_dim)
        tokens = self.token_proj(tokens)                # (B, n_tokens, feature_dim)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)        # (B, n_tokens+1, feature_dim)
        tokens = tokens + self.pos_embed                # add positional embeddings
        out = self.encoder(tokens)                      # (B, n_tokens+1, feature_dim)
        return self.out_proj(out[:, 0])                 # CLS → (B, feature_dim)


class JEPAStyleEncoder(nn.Module):
    """JEPA-inspired shared encoder (Joint-Embedding Predictive Architecture).

    Splits the 512-d vector into n_tokens chunks and applies two stages:

    1. Context encoder (Transformer): processes all token positions, but
       during training a fraction (mask_ratio) of tokens are replaced with
       a learned mask token, hiding their content from the context encoder.

    2. Predictor (lighter Transformer): takes the context encoder output and
       predicts/refines representations for ALL positions — including the
       masked ones — entirely in latent/embedding space (not input space).
       This is the key JEPA distinction from MAE.

    At inference: mask_ratio=0, all tokens visible, predictor just refines.
    """

    def __init__(
        self,
        feature_dim: int,
        n_tokens: int = 8,
        context_layers: int = 2,
        predictor_layers: int = 2,
        nhead: int = 8,
        mask_ratio: float = 0.25,
    ):
        super().__init__()
        if feature_dim % n_tokens != 0:
            raise ValueError(f"feature_dim ({feature_dim}) must be divisible by n_tokens ({n_tokens})")
        self.n_tokens = n_tokens
        self.mask_ratio = mask_ratio

        token_dim = feature_dim // n_tokens
        self.token_proj = nn.Linear(token_dim, feature_dim)

        self.cls_token  = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.pos_embed  = nn.Parameter(torch.zeros(1, n_tokens + 1, feature_dim))
        nn.init.normal_(self.cls_token,  std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.pos_embed,  std=0.02)

        # Context encoder — full capacity
        context_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=nhead,
            dim_feedforward=feature_dim * 2,
            dropout=0.1, batch_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(context_layer, num_layers=context_layers)

        # Predictor — narrower feedforward (predicts in latent space, not input space)
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=nhead,
            dim_feedforward=feature_dim,      # narrower than context encoder
            dropout=0.1, batch_first=True,
        )
        self.predictor = nn.TransformerEncoder(predictor_layer, num_layers=predictor_layers)

        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        tokens = x.view(batch, self.n_tokens, -1)   # (B, n_tokens, token_dim)
        tokens = self.token_proj(tokens)             # (B, n_tokens, feature_dim)

        # Replace a random subset of tokens with the mask token (training only)
        if self.training and self.mask_ratio > 0:
            mask = torch.rand(batch, self.n_tokens, device=x.device) < self.mask_ratio
            mask_tokens = self.mask_token.expand(batch, self.n_tokens, -1)
            tokens = torch.where(mask.unsqueeze(-1), mask_tokens, tokens)

        cls    = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)     # (B, n_tokens+1, feature_dim)
        tokens = tokens + self.pos_embed

        # Stage 1: context encoder extracts representations
        ctx = self.context_encoder(tokens)           # (B, n_tokens+1, feature_dim)

        # Stage 2: predictor refines all positions in latent space
        out = self.predictor(ctx)                    # (B, n_tokens+1, feature_dim)

        return self.out_proj(out[:, 0])              # CLS → (B, feature_dim)


def _build_share_encoder(encoder_type: str, feature_dim: int) -> nn.Module:
    if encoder_type == "linear":
        return nn.Linear(feature_dim, feature_dim)
    elif encoder_type in ("none", "separate"):
        return nn.Linear(feature_dim, feature_dim)
    elif encoder_type == "transformer":
        return TransformerShareEncoder(feature_dim)
    elif encoder_type == "tokenized_cls":
        return TokenizedCLSEncoder(feature_dim)
    elif encoder_type == "jepa":
        return JEPAStyleEncoder(feature_dim)
    else:
        raise ValueError(f"share_encoder_type must be one of {_SHARE_ENCODER_TYPES}, got {encoder_type!r}")


class SubjectAwareRouter(nn.Module):
    """Subject-aware blending weights for 5 InternViT layer features.

    Produces a (batch, n_layers) softmax weight vector that blends
    the pre-extracted InternViT layer features into one combined
    image representation.

    Components:
        global_logits:  shared prior over layers, init [-2,-1,0,-1,-2]
                        centered at layer 28 (index 2 of [20,24,28,32,36])
        subject_bias:   per-subject deviation from global prior
                        Embedding(n_subjects, n_layers), init zeros

    Training:
        weights = softmax((global_logits + subject_bias[subject_id]
                  * subject_dropout_mask) * layer_dropout_mask / temperature)

    Inference:
        weights = softmax(global_logits / temperature)
        subject_bias never consulted - global prior only

    Args:
        n_subjects:           int   = 10   total subjects (always 10)
        n_layers:             int   = 5    number of visual layers
        temperature:          float = 1.0  softmax temperature
        subject_dropout_rate: float = 0.3  prob of zeroing subject bias
                                            forces model to learn global prior
        layer_dropout_rate:   float = 0.1  prob of zeroing each layer logit
                                            prevents over-concentration
    """

    def __init__(
        self,
        n_subjects: int = 10,
        n_layers: int = 5,
        temperature: float = 1.0,
        subject_dropout_rate: float = 0.3,
        layer_dropout_rate: float = 0.1,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not (0.0 <= subject_dropout_rate <= 1.0):
            raise ValueError(
                f"subject_dropout_rate must be in [0, 1], got {subject_dropout_rate}"
            )
        if not (0.0 <= layer_dropout_rate <= 1.0):
            raise ValueError(
                f"layer_dropout_rate must be in [0, 1], got {layer_dropout_rate}"
            )

        self.temperature = float(temperature)
        self.subject_dropout_rate = float(subject_dropout_rate)
        self.layer_dropout_rate = float(layer_dropout_rate)

        init_logits = torch.zeros(n_layers, dtype=torch.float32)
        if n_layers == 5:
            init_logits = torch.tensor(
                [-2.0, -1.0, 0.0, -1.0, -2.0],
                dtype=torch.float32,
            )
        self.global_logits = nn.Parameter(init_logits)

        self.subject_bias = nn.Embedding(n_subjects, n_layers)
        nn.init.zeros_(self.subject_bias.weight)

    def forward(
        self,
        subject_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute blending weights.

        Args:
            subject_ids: (batch,) int64 0-indexed subject IDs
                         Pass None at inference to use global prior only
                         subject_ids is also ignored when self.training=False

        Returns:
            weights: (batch, n_layers) softmax weights summing to 1.0
        """
        batch_size = subject_ids.shape[0] if subject_ids is not None else 1
        logits = self.global_logits.unsqueeze(0).expand(batch_size, -1).clone()

        if self.training and subject_ids is not None:
            bias = self.subject_bias(subject_ids)

            s_mask = (
                torch.rand(batch_size, 1, device=bias.device)
                > self.subject_dropout_rate
            ).float()
            bias = bias * s_mask

            logits = logits + bias

            l_mask = (
                torch.rand_like(logits)
                > self.layer_dropout_rate
            ).float()
            logits = logits * l_mask

        return F.softmax(logits / self.temperature, dim=1)


class SUPAEEG(nn.Module):
    """SUPAEEG: EEGProject + shared encoder alignment

    Architecture:
        EEG (batch, 17, 100)
          eeg_encoder  -> (batch, 1024)   temporal CNN
          eeg_projector Linear(1024, 512)
          share_encoder Linear(512, 512)  <- shared with image
          l2-normalize  -> zE (batch, 512)

        image_layers (batch, 5, 3200)
          router weights -> weighted mean -> (batch, 3200)
          img_pre_projector Linear(3200, 1024)
          img_projector     Linear(1024, 512)
          share_encoder     Linear(512, 512)  <- SAME nn.Module as EEG
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
                 image_mid_dim=1024, feature_dim=512, dropout=0.3,
                 n_subjects=10, n_layers=5, router_temperature=1.0,
                 subject_dropout_rate=0.3, layer_dropout_rate=0.1,
                 share_encoder_type="linear", eeg_encoder_type="eegproject",
                 image_layer_mode="router", image_layer_index=0,
                 temporal_compression=0):
        super().__init__()
        if share_encoder_type not in _SHARE_ENCODER_TYPES:
            raise ValueError(f"share_encoder_type must be one of {_SHARE_ENCODER_TYPES}, got {share_encoder_type!r}")
        self.share_encoder_type = share_encoder_type
        self.eeg_encoder_type = eeg_encoder_type
        self.image_layer_mode = image_layer_mode
        self.image_layer_index = image_layer_index
        self.eeg_encoder = build_eeg_encoder(
            eeg_encoder_type, n_channels, n_timepoints, eeg_feature_dim,
            dropout, temporal_compression,
        )
        self.eeg_projector     = nn.Linear(eeg_feature_dim, feature_dim)
        self.img_pre_projector = nn.Linear(image_input_dim, image_mid_dim)
        self.img_projector     = nn.Linear(image_mid_dim, feature_dim)
        # Build share encoder(s).
        # "linear" / "transformer" / "jepa": one shared module used by both paths.
        # "separate": two independent modules, no weight sharing.
        # "none": Identity — both paths go directly to l2-normalize.
        if share_encoder_type == "none":
            enc = nn.Identity()
            self.eeg_share_encoder = enc
            self.img_share_encoder = enc
        elif share_encoder_type == "separate":
            self.eeg_share_encoder = _build_share_encoder(share_encoder_type, feature_dim)
            self.img_share_encoder = _build_share_encoder(share_encoder_type, feature_dim)
        else:  # linear, transformer, jepa — shared weights
            enc = _build_share_encoder(share_encoder_type, feature_dim)
            self.eeg_share_encoder = enc
            self.img_share_encoder = enc
        self.logit_scale       = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        )
        self.router            = SubjectAwareRouter(
            n_subjects=n_subjects,
            n_layers=n_layers,
            temperature=router_temperature,
            subject_dropout_rate=subject_dropout_rate,
            layer_dropout_rate=layer_dropout_rate,
        )
        if image_layer_mode not in {"router", "uniform", "single"}:
            raise ValueError("image_layer_mode must be router, uniform, or single")
        if image_layer_mode == "single" and not 0 <= image_layer_index < n_layers:
            raise ValueError(
                f"image_layer_index must be in [0, {n_layers}), got {image_layer_index}"
            )

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        x = self.eeg_encoder(eeg)          # (batch, 1024)
        x = self.eeg_projector(x)          # (batch, 512)
        x = self.eeg_share_encoder(x)      # (batch, 512)
        return F.normalize(x, dim=1)

    def encode_image(
        self,
        image_layers: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode image layers with subject-aware blending.

        Args:
            image_layers: (batch, n_layers, 3200) float16 or float32
            subject_ids:  (batch,) int64 0-indexed, or None

        Returns:
            (batch, 512) l2-normalised
        """
        if self.image_layer_mode == "router":
            weights = self.router(subject_ids)
        elif self.image_layer_mode == "uniform":
            weights = image_layers.new_full(
                (image_layers.shape[0], image_layers.shape[1]),
                1.0 / image_layers.shape[1],
            )
        else:
            weights = image_layers.new_zeros(
                (image_layers.shape[0], image_layers.shape[1])
            )
            weights[:, self.image_layer_index] = 1.0
        x = (image_layers.float() * weights.unsqueeze(-1)).sum(dim=1)
        x = self.img_pre_projector(x)
        x = self.img_projector(x)
        x = self.img_share_encoder(x)      # (batch, 512)
        return F.normalize(x, dim=1)

    def forward(
        self,
        eeg: torch.Tensor,
        image_layers: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_eeg(eeg), self.encode_image(image_layers, subject_ids)

    @torch.no_grad()
    def embed(self, eeg: torch.Tensor) -> torch.Tensor:
        """Inference only. Returns l2-normalised (batch, 512) descriptor."""
        return self.encode_eeg(eeg)


if __name__ == "__main__":
    model = SUPAEEG()
    eeg  = torch.randn(4, 17, 100)
    imgs = torch.randn(4, 5, 3200)
    sids = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    model.train()
    zE, zI = model(eeg, imgs, sids)
    assert zE.shape == (4, 512), f"got {zE.shape}"
    assert zI.shape == (4, 512), f"got {zI.shape}"

    model.eval()
    with torch.no_grad():
        emb = model.embed(eeg)
        assert emb.shape == (4, 512), f"got {emb.shape}"

        zI_no_sid = model.encode_image(imgs, subject_ids=None)
        assert zI_no_sid.shape == (4, 512)

    from src.encoders.eeg_augmentation import smooth_eeg
    smoothed = smooth_eeg(eeg, p=1.0)
    assert smoothed.shape == eeg.shape
    assert not torch.allclose(smoothed, eeg)

    smoothed_zero = smooth_eeg(eeg, p=0.0)
    assert torch.allclose(smoothed_zero, eeg)

    print("All assertions passed")
