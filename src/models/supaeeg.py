import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders.eegnet_encoder import EEGNetEncoder


class GradientReversalFunction(torch.autograd.Function):
    """Reverses gradients during backward pass.
    Forward pass: identity (no change to values)
    Backward pass: multiply gradient by -lambda_grl
    This trains the encoder to REMOVE subject information.
    """

    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.save_for_backward(torch.tensor(lambda_grl))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_grl = ctx.saved_tensors[0].item()
        return -lambda_grl * grad_output, None


class GradientReversalLayer(nn.Module):
    """Apply gradient reversal with scheduled lambda.

    lambda_grl controls reversal strength:
      0.0 = no reversal (GRL off)
      1.0 = full reversal (standard GRL)

    Scheduled from 0 → lambda_max over training to
    stabilise early training before adversarial pressure builds.

    Args:
        lambda_max: float = 1.0  maximum reversal strength
    """

    def __init__(self, lambda_max: float = 1.0):
        super().__init__()
        self.lambda_max = lambda_max
        self._lambda = 0.0   # current value, updated by set_lambda()

    def set_lambda(self, progress: float) -> None:
        """Update lambda based on training progress in [0, 1].
        Uses standard GRL schedule: 2/(1+exp(-10*p)) - 1
        progress=0 → lambda=0, progress=1 → lambda=lambda_max
        """
        self._lambda = self.lambda_max * (
            2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self._lambda)


class SubjectClassifier(nn.Module):
    """2-layer MLP predicting subject ID from EEG features.
    Trained to predict subject; encoder trained to fool it via GRL.
    Discarded entirely at inference.

    Args:
        in_features:  int = 512   input feature dimension
        n_subjects:   int = 10    number of training subjects
        hidden_dim:   int = 256   hidden layer size
    """

    def __init__(self, in_features=512, n_subjects=10, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_subjects),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_features) → (batch, n_subjects) logits"""
        return self.net(x)


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
    """SUPAEEG: EEGProject + shared encoder alignment.

    Architecture:
        EEG (batch, 17, 100)
          eeg_encoder  -> (batch, 1024)
          eeg_projector Linear(1024, 512)
          share_encoder Linear(512, 512)   <- shared with image side
          l2-normalize  -> zE (batch, 512)

        image_layers (batch, 5, 3200)
          .float().mean(dim=1) -> (batch, 3200)
          img_pre_projector Linear(3200, 1024)
          img_projector     Linear(1024, 512)
          share_encoder     Linear(512, 512)   <- SAME nn.Module as EEG
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
                 use_grl: bool = True, lambda_grl_max: float = 1.0,
                 grl_hidden_dim: int = 256):
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

        # Instance normalisation — removes per-trial amplitude scaling
        # (subject-specific skull thickness / electrode impedance artefacts)
        # Uses (batch, 1, feature_dim) shape so each sample's features are
        # normalised independently over the feature/length dimension.
        self.instance_norm = nn.InstanceNorm1d(1)

        # Adversarial subject invariance
        self.use_grl = use_grl
        if use_grl:
            self.grl      = GradientReversalLayer(lambda_max=lambda_grl_max)
            self.subj_clf = SubjectClassifier(
                in_features=feature_dim,
                n_subjects=n_subjects,
                hidden_dim=grl_hidden_dim,
            )

    def encode_eeg(
        self,
        eeg: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode EEG into shared embedding space.

        Args:
            eeg:         (batch, 17, 100)
            subject_ids: (batch,) int64 0-indexed, or None

        Returns:
            zE:          (batch, 512) l2-normalised EEG embedding
            subj_logits: (batch, n_subjects) or None if not training/no GRL
        """
        x = self.eeg_encoder(eeg)          # (batch, 1024)
        x = self.eeg_projector(x)          # (batch, 512)
        x = self.instance_norm(x.unsqueeze(1)).squeeze(1)  # (batch, 512) normalise amplitude

        # GRL branch: adversarial subject prediction
        subj_logits = None
        if self.use_grl and self.training and subject_ids is not None:
            x_grl       = self.grl(x)           # reversed gradients
            subj_logits = self.subj_clf(x_grl)  # (batch, n_subjects)

        x = self.share_encoder(x)          # (batch, 512)
        return F.normalize(x, dim=1), subj_logits

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            zE:          (batch, 512)
            zI:          (batch, 512)
            subj_logits: (batch, n_subjects) or None
        """
        zE, subj_logits = self.encode_eeg(eeg, subject_ids)
        zI = self.encode_image(image_layers, subject_ids)
        return zE, zI, subj_logits

    @torch.no_grad()
    def embed(self, eeg: torch.Tensor) -> torch.Tensor:
        """Inference only. Returns l2-normalised (batch, 512) descriptor."""
        zE, _ = self.encode_eeg(eeg, subject_ids=None)
        return zE


if __name__ == "__main__":
    model = SUPAEEG()
    eeg = torch.randn(4, 17, 100)
    imgs = torch.randn(4, 5, 3200)
    sids = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    model.train()
    zE, zI, subj_logits = model(eeg, imgs, sids)
    assert zE.shape == (4, 512), f"got {zE.shape}"
    assert zI.shape == (4, 512), f"got {zI.shape}"
    assert subj_logits is not None and subj_logits.shape == (4, 10)

    model.eval()
    with torch.no_grad():
        emb = model.embed(eeg)
        assert emb.shape == (4, 512), f"got {emb.shape}"

        zI_no_sid = model.encode_image(imgs, subject_ids=None)
        assert zI_no_sid.shape == (4, 512)

        zI_with_sid = model.encode_image(imgs, subject_ids=sids)
        assert zI_with_sid.shape == (4, 512)
        assert torch.allclose(zI_no_sid, zI_with_sid, atol=1e-5), (
            "inference must use global prior regardless of subject_ids"
        )

    print("Phase 2 all assertions passed")
