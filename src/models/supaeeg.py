import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders.eegnet_encoder import EEGNetEncoder


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
    """SUPAEEG: temporal CNN EEG encoder + shared linear alignment.

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
                 subject_dropout_rate=0.3, layer_dropout_rate=0.1):
        super().__init__()
        self.eeg_encoder       = EEGNetEncoder(n_channels, n_timepoints,
                                               eeg_feature_dim, dropout)
        self.eeg_projector     = nn.Linear(eeg_feature_dim, feature_dim)
        self.img_pre_projector = nn.Linear(image_input_dim, image_mid_dim)
        self.img_projector     = nn.Linear(image_mid_dim, feature_dim)
        self.share_encoder     = nn.Linear(feature_dim, feature_dim)
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

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        x = self.eeg_encoder(eeg)      # (batch, 1024)
        x = self.eeg_projector(x)      # (batch, 512)
        x = self.share_encoder(x)      # (batch, 512)
        return F.normalize(x, dim=1)   # (batch, 512)

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
        weights = self.router(subject_ids)
        x = (image_layers.float() * weights.unsqueeze(-1)).sum(dim=1)
        x = self.img_pre_projector(x)
        x = self.img_projector(x)
        x = self.share_encoder(x)
        return F.normalize(x, dim=1)   # (batch, 512)

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
