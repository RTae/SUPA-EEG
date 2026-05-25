"""
Visual Feature Encoder for THINGS-EEG offline feature extraction.

Wraps a frozen CLIP or I-JEPA ViT encoder and extracts multi-scale features
at three depth levels via forward hooks:

    S1 – early layers  → local edges & colours
    S2 – mid layers    → spatial structure
    S3 – late layers   → high-level object semantics

The extracted features form a fixed lookup table (concept, image_file) → {S1, S2, S3}
that is loaded at training time by VisualFeatureLookup.

Architecture reference:
    - CLIP  : openai/clip-vit-base-patch32  (12 transformer layers)
    - I-JEPA: facebook/ijepa_vith14_1k      (32 transformer layers)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from loguru import logger

# ---------------------------------------------------------------------------
# Per-encoder configuration: which HuggingFace checkpoint to use by default,
# how many transformer layers it has, and which layer indices correspond to
# the three depth scales S1 / S2 / S3.
# ---------------------------------------------------------------------------
_ENCODER_CONFIGS: dict[str, dict] = {
    "clip": {
        "default_model": "openai/clip-vit-base-patch32",
        "num_layers": 12,
        # Divide 12 layers into thirds: [0-3] local, [4-7] spatial, [8-11] semantic
        "s1_layer": 3,
        "s2_layer": 7,
        "s3_layer": 11,
    },
    "ijepa": {
        "default_model": "facebook/ijepa_vith14_1k",
        "num_layers": 32,
        # Divide 32 layers into thirds: [0-10] local, [11-21] spatial, [22-31] semantic
        "s1_layer": 10,
        "s2_layer": 21,
        "s3_layer": 31,
    },
}


class VisualEncoder(nn.Module):
    """Frozen CLIP or I-JEPA visual encoder with multi-scale depth tapping.

    All model parameters are frozen at construction time.  The encoder is
    strictly used for inference: calling ``forward`` registers temporary hooks
    on layers S1 / S2 / S3, runs a single forward pass, removes the hooks,
    and returns the three CLS-token embeddings.

    Args:
        encoder_type: ``"clip"`` or ``"ijepa"``.
        model_name:   HuggingFace model identifier.  Defaults to the
                      canonical checkpoint for the chosen encoder type.
        device:       Target device for the model weights.
    """

    def __init__(
        self,
        encoder_type: Literal["clip", "ijepa"] = "clip",
        model_name: str | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()

        if encoder_type not in _ENCODER_CONFIGS:
            raise ValueError(
                f"encoder_type must be one of {list(_ENCODER_CONFIGS.keys())}, "
                f"got '{encoder_type}'"
            )

        self.encoder_type = encoder_type
        cfg = _ENCODER_CONFIGS[encoder_type]
        self._model_name = model_name or cfg["default_model"]
        self.device = torch.device(device) if isinstance(device, str) else device

        self._s1_idx: int = cfg["s1_layer"]
        self._s2_idx: int = cfg["s2_layer"]
        self._s3_idx: int = cfg["s3_layer"]

        self._hooks: list = []
        self._intermediate: dict[str, torch.Tensor] = {}

        logger.info(
            f"Loading {encoder_type.upper()} encoder from '{self._model_name}' "
            f"on device={self.device} | "
            f"S1=layer{self._s1_idx}, S2=layer{self._s2_idx}, S3=layer{self._s3_idx}"
        )

        self._build_model()

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self) -> None:
        """Instantiate model + processor and freeze all parameters."""
        if self.encoder_type == "clip":
            from transformers import CLIPImageProcessor, CLIPVisionModel

            self.processor = CLIPImageProcessor.from_pretrained(self._model_name)
            self.model = CLIPVisionModel.from_pretrained(self._model_name)
            # Transformer layers live at vision_model.encoder.layers
            self._transformer_layers = self.model.vision_model.encoder.layers

        else:  # ijepa
            from transformers import AutoImageProcessor, AutoModel

            self.processor = AutoImageProcessor.from_pretrained(self._model_name)
            self.model = AutoModel.from_pretrained(self._model_name)
            # Standard ViT layout used by I-JEPA
            self._transformer_layers = self.model.encoder.layer

        # Freeze everything – this encoder is never trained
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.model.to(self.device)

        num_layers = len(self._transformer_layers)
        logger.info(
            f"{self.encoder_type.upper()} loaded | "
            f"total transformer layers: {num_layers} | "
            f"parameters: {sum(p.numel() for p in self.model.parameters()):,} (all frozen)"
        )

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Attach forward hooks to the S1, S2, S3 layers."""
        self._remove_hooks()

        def _make_hook(scale_key: str):
            def _hook(module: nn.Module, input, output) -> None:  # noqa: ARG001
                # Layer output is usually (hidden_states,) or just hidden_states.
                hidden = output[0] if isinstance(output, tuple) else output
                # CLS token is always the first sequence position in ViT variants.
                self._intermediate[scale_key] = hidden[:, 0, :].detach()

            return _hook

        layers = self._transformer_layers
        self._hooks.append(layers[self._s1_idx].register_forward_hook(_make_hook("S1")))
        self._hooks.append(layers[self._s2_idx].register_forward_hook(_make_hook("S2")))
        self._hooks.append(layers[self._s3_idx].register_forward_hook(_make_hook("S3")))

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Preprocessing helper
    # ------------------------------------------------------------------

    def preprocess(self, images: list) -> torch.Tensor:
        """Run the HuggingFace processor on a list of PIL images.

        Returns:
            ``pixel_values`` tensor on ``self.device``, ready for ``forward``.
        """
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract multi-scale CLS-token features from a batch of images.

        Args:
            pixel_values: Float tensor ``(B, C, H, W)`` as returned by
                          ``self.preprocess``.

        Returns:
            Dictionary with keys ``"S1"``, ``"S2"``, ``"S3"``, each a
            CPU tensor of shape ``(B, D)`` where *D* is the hidden dimension
            of the corresponding encoder layer.
        """
        self._intermediate.clear()
        self._register_hooks()

        try:
            self.model(pixel_values=pixel_values)
        finally:
            # Always clean up hooks, even if the forward pass raises.
            self._remove_hooks()

        if len(self._intermediate) != 3:  # pragma: no cover
            raise RuntimeError(
                f"Expected 3 intermediate features (S1/S2/S3), "
                f"got {list(self._intermediate.keys())}. "
                "Check that the layer indices are within the model depth."
            )

        return {k: v.cpu() for k, v in self._intermediate.items()}


# ---------------------------------------------------------------------------
# Lookup table – loaded once at training time, never updated
# ---------------------------------------------------------------------------


class VisualFeatureLookup:
    """Read-only lookup table of pre-extracted multi-scale image features.

    The table maps ``(concept, image_file)`` string pairs to a dict with
    keys ``"S1"``, ``"S2"``, ``"S3"`` (1-D feature tensors).

    Usage::

        lookup = VisualFeatureLookup("data/visual_features_clip.pt")
        s1, s2, s3 = lookup.retrieve("00001_aardvark", "aardvark_01b.jpg")

    Args:
        path:   Path to the ``.pt`` file produced by
                ``scripts/extract_visual_features.py``.
        device: Device to move tensors to on retrieval.
    """

    def __init__(
        self,
        path: str | Path,
        device: str | torch.device = "cpu",
    ) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Visual features file not found: {path}")

        self.device = torch.device(device) if isinstance(device, str) else device

        logger.info(f"Loading visual feature lookup table from '{path}'...")
        payload: dict = torch.load(path, map_location="cpu", weights_only=True)

        # Expected payload keys: "features", "encoder_type", "model_name", "layer_indices"
        self._table: dict[tuple[str, str], dict[str, torch.Tensor]] = payload["features"]
        self.encoder_type: str = payload.get("encoder_type", "unknown")
        self.model_name: str = payload.get("model_name", "unknown")
        self.layer_indices: dict[str, int] = payload.get("layer_indices", {})

        logger.info(
            f"Lookup table loaded | "
            f"encoder={self.encoder_type} ({self.model_name}) | "
            f"entries={len(self._table):,} | "
            f"layer indices={self.layer_indices}"
        )

    def retrieve(
        self, concept: str, image_file: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (S1, S2, S3) feature tensors for a single image.

        Args:
            concept:    Top-level folder name, e.g. ``"00001_aardvark"``.
            image_file: Image filename,         e.g. ``"aardvark_01b.jpg"``.

        Returns:
            Tuple ``(S1, S2, S3)`` of 1-D tensors on ``self.device``.
        """
        key = (concept, image_file)
        if key not in self._table:
            raise KeyError(
                f"No visual features found for ({concept!r}, {image_file!r}). "
                "Make sure the lookup table was built from the same dataset split."
            )
        entry = self._table[key]
        s1 = entry["S1"].to(self.device)
        s2 = entry["S2"].to(self.device)
        s3 = entry["S3"].to(self.device)
        return s1, s2, s3

    def retrieve_batch(
        self,
        concepts: list[str],
        image_files: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return stacked (S1, S2, S3) tensors for a batch of images.

        Args:
            concepts:    List of concept folder names, length *B*.
            image_files: List of image filenames,      length *B*.

        Returns:
            Tuple ``(S1, S2, S3)`` each of shape ``(B, D)`` on ``self.device``.
        """
        if len(concepts) != len(image_files):
            raise ValueError("concepts and image_files must have the same length.")

        s1_list, s2_list, s3_list = [], [], []
        for concept, image_file in zip(concepts, image_files):
            s1, s2, s3 = self.retrieve(concept, image_file)
            s1_list.append(s1)
            s2_list.append(s2)
            s3_list.append(s3)

        return (
            torch.stack(s1_list, dim=0),
            torch.stack(s2_list, dim=0),
            torch.stack(s3_list, dim=0),
        )

    def __len__(self) -> int:
        return len(self._table)

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._table
