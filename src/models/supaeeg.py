import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders.eegnet_encoder import TemporalPatchEncoder


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
    """SUPAEEG with temporal patching and aligned layer routing.

    Each temporal window of EEG is aligned to a corresponding
    group of InternViT layers, exploiting the known temporal
    structure of visual EEG responses.

    Args:
        n_channels:           int   = 17
        n_timepoints:         int   = 100
        n_windows:            int   = 4      temporal patches
        feature_dim:          int   = 512    per-window embedding dim
        image_input_dim:      int   = 3200   InternViT feature dim
        image_mid_dim:        int   = 1024   image projector intermediate
        n_subjects:           int   = 10
        n_layers:             int   = 5
        router_temperature:   float = 1.0
        subject_dropout_rate: float = 0.3
        layer_dropout_rate:   float = 0.1
        dropout:              float = 0.3
    """

    # Layer-group assignment for 5 InternViT layers [20, 24, 28, 32, 36].
    # Indices refer to position in the 5-layer stack (0=layer20 … 4=layer36).
    # Overlap at layer 28 (index 2) is intentional — it is the most informative.
    LAYER_GROUPS: list[list[int]] = [
        [0],       # window 0 (0-25 ms)   early visual  → layer 20
        [1, 2],    # window 1 (25-50 ms)  mid           → layers 24, 28
        [2, 3],    # window 2 (50-75 ms)  late-mid      → layers 28, 32
        [3, 4],    # window 3 (75-100 ms) semantic      → layers 32, 36
    ]

    def __init__(
        self,
        n_channels: int = 17,
        n_timepoints: int = 100,
        n_windows: int = 4,
        feature_dim: int = 512,
        image_input_dim: int = 3200,
        image_mid_dim: int = 1024,
        n_subjects: int = 10,
        n_layers: int = 5,
        router_temperature: float = 1.0,
        subject_dropout_rate: float = 0.3,
        layer_dropout_rate: float = 0.1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_windows = n_windows
        self.feature_dim = feature_dim

        # EEG encoder: n_windows separate MLPs
        self.eeg_encoder = TemporalPatchEncoder(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_windows=n_windows,
            feature_dim=feature_dim,
            dropout=dropout,
        )

        # per-window EEG projectors (separate weights — each window specialises)
        self.eeg_projectors = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim)
            for _ in range(n_windows)
        ])

        # shared image pre-projector
        self.img_pre_projector = nn.Linear(image_input_dim, image_mid_dim)

        # per-window image projectors
        self.img_projectors = nn.ModuleList([
            nn.Linear(image_mid_dim, feature_dim)
            for _ in range(n_windows)
        ])

        # SHARED encoder — same nn.Linear for both EEG and image, all windows
        self.share_encoder = nn.Linear(feature_dim, feature_dim)

        # subject-aware router for image layer blending
        self.router = SubjectAwareRouter(
            n_subjects=n_subjects,
            n_layers=n_layers,
            temperature=router_temperature,
            subject_dropout_rate=subject_dropout_rate,
            layer_dropout_rate=layer_dropout_rate,
        )

        # learnable temperature
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        )

    def encode_eeg_windows(
        self,
        eeg: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Encode EEG into per-window l2-normalised embeddings.

        Args:
            eeg: (batch, n_channels, n_timepoints)

        Returns:
            List of n_windows tensors each (batch, feature_dim), l2-normalised.
        """
        window_feats = self.eeg_encoder(eeg)   # list of (batch, feature_dim)
        zE_list = []
        for feat, proj in zip(window_feats, self.eeg_projectors):
            z = proj(feat)                      # (batch, feature_dim)
            z = self.share_encoder(z)           # (batch, feature_dim)
            zE_list.append(F.normalize(z, dim=1))
        return zE_list

    def encode_image_windows(
        self,
        image_layers: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Encode image into per-window embeddings aligned to layer groups.

        Args:
            image_layers: (batch, n_layers, 3200)
            subject_ids:  (batch,) int64 0-indexed, or None

        Returns:
            List of n_windows tensors each (batch, feature_dim), l2-normalised.
        """
        # router weights over all 5 layers: (batch, 5)
        weights = self.router(subject_ids)

        zI_list = []
        for k, (layer_indices, img_proj) in enumerate(
            zip(self.LAYER_GROUPS, self.img_projectors)
        ):
            group_layers  = image_layers[:, layer_indices, :]   # (batch, g, 3200)
            group_weights = weights[:, layer_indices]            # (batch, g)
            # renormalise weights within group
            group_weights = group_weights / group_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            # weighted mean over assigned layers: (batch, 3200)
            x = (group_layers.float() * group_weights.unsqueeze(-1)).sum(dim=1)
            x = self.img_pre_projector(x)   # (batch, image_mid_dim)
            x = img_proj(x)                 # (batch, feature_dim)
            x = self.share_encoder(x)       # (batch, feature_dim)
            zI_list.append(F.normalize(x, dim=1))

        return zI_list

    def forward(
        self,
        eeg: torch.Tensor,
        image_layers: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Returns:
            zE_list: list of n_windows (batch, feature_dim) EEG embeddings
            zI_list: list of n_windows (batch, feature_dim) image embeddings
        """
        return (
            self.encode_eeg_windows(eeg),
            self.encode_image_windows(image_layers, subject_ids),
        )

    @torch.no_grad()
    def embed(self, eeg: torch.Tensor) -> torch.Tensor:
        """Inference: concat all window embeddings → l2-norm → (batch, n_windows * feature_dim)."""
        zE_list = self.encode_eeg_windows(eeg)
        z = torch.cat(zE_list, dim=1)   # (batch, n_windows * feature_dim)
        return F.normalize(z, dim=1)

    @torch.no_grad()
    def encode_image_for_eval(
        self,
        image_layers: torch.Tensor,
    ) -> torch.Tensor:
        """Inference: concat all window image embeddings → l2-norm → (batch, n_windows * feature_dim)."""
        zI_list = self.encode_image_windows(image_layers, subject_ids=None)
        z = torch.cat(zI_list, dim=1)
        return F.normalize(z, dim=1)


if __name__ == "__main__":
    model = SUPAEEG()
    eeg  = torch.randn(4, 17, 100)
    imgs = torch.randn(4, 5, 3200)
    sids = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    # training
    model.train()
    zE_list, zI_list = model(eeg, imgs, sids)
    assert len(zE_list) == 4
    assert len(zI_list) == 4
    for k in range(4):
        assert zE_list[k].shape == (4, 512), f"zE[{k}] shape {zE_list[k].shape}"
        assert zI_list[k].shape == (4, 512), f"zI[{k}] shape {zI_list[k].shape}"

    # inference embed
    model.eval()
    with torch.no_grad():
        emb = model.embed(eeg)
        assert emb.shape == (4, 2048), f"embed shape {emb.shape}"
        img_emb = model.encode_image_for_eval(imgs)
        assert img_emb.shape == (4, 2048), f"img_emb shape {img_emb.shape}"

    # smooth augmentation
    from src.encoders.eeg_augmentation import smooth_eeg
    smoothed = smooth_eeg(eeg, p=1.0)
    assert smoothed.shape == eeg.shape
    assert not torch.allclose(smoothed, eeg)   # should be different

    smoothed_zero = smooth_eeg(eeg, p=0.0)
    assert torch.allclose(smoothed_zero, eeg)  # p=0 should be identity

    print("All assertions passed")
