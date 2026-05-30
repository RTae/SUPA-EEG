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
from typing import Literal, Sequence

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
        encoder_layers_path: str | None = None,
    ) -> None:
        """
        Args:
            encoder_type:        ``"clip"`` or ``"ijepa"``.
            model_name:          HuggingFace model identifier.
            device:              Target device for the model weights.
            encoder_layers_path: Optional dot-separated path to the transformer
                                 layer list inside the model object, e.g.
                                 ``"encoder.layer"`` or ``"ijepa.encoder.layer"``.
                                 Only needed if auto-discovery fails for an unusual
                                 checkpoint.  Example::

                                     VisualEncoder("ijepa",
                                                   encoder_layers_path="encoder.layer")
        """
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
        self._encoder_layers_path = encoder_layers_path

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

    @staticmethod
    def _as_layer_sequence(candidate: object) -> Sequence[nn.Module] | None:
        """Normalize common HuggingFace encoder containers to an indexable sequence."""
        if isinstance(candidate, (nn.ModuleList, list, tuple)) and len(candidate) > 0:
            return candidate

        if not isinstance(candidate, nn.Module):
            return None

        children = list(candidate.named_children())
        if not children:
            return None

        # Some encoder modules wrap the actual blocks in a single ModuleList child.
        if len(children) == 1 and isinstance(children[0][1], nn.ModuleList):
            layer_list = children[0][1]
            return layer_list if len(layer_list) > 0 else None

        # Other I-JEPA versions expose blocks directly as .0, .1, ... children.
        if all(name.isdigit() for name, _ in children):
            return [child for _, child in children]

        return None

    @staticmethod
    def _find_transformer_layers(model: nn.Module) -> Sequence[nn.Module]:
        """Locate the list of transformer encoder layers inside *model*.

        Different HuggingFace model classes store their layers under different
        attribute paths.  This method tries every known layout in order and
        returns the first one that resolves to a non-empty layer sequence.

        If nothing matches, it prints a two-level attribute tree of the model
        so you can identify the correct path and open an issue / PR.

        Known layouts tried (in order):
            model.encoder                 – IJepaModel variants with direct .0/.1 blocks
            model.encoder.layer           – IJepaModel, BertModel, ViTModel
            model.encoder.layers          – CLIPEncoder (fallback)
            model.vision_model.encoder.layers – CLIPVisionModel (fallback)
            model.ijepa.encoder.layer     – IJepaForMaskedImageModeling wrapper
            model.vit.encoder.layer       – ViTForImageClassification wrapper
            model.model.encoder.layer     – generic ForX wrapper
            model.model.encoder.layers    – generic ForX wrapper (layers variant)
        """
        _CANDIDATES: list[tuple[str, object]] = [
            # Some custom checkpoints expose the encoder as the layer container.
            ("model.encoder",                     lambda m: m.encoder),
            ("model.encoder.layer",               lambda m: m.encoder.layer),
            ("model.encoder.layers",              lambda m: m.encoder.layers),
            ("model.vision_model.encoder.layers", lambda m: m.vision_model.encoder.layers),
            ("model.ijepa.encoder.layer",         lambda m: m.ijepa.encoder.layer),
            ("model.vit.encoder.layer",           lambda m: m.vit.encoder.layer),
            ("model.model.encoder.layer",         lambda m: m.model.encoder.layer),
            ("model.model.encoder.layers",        lambda m: m.model.encoder.layers),
        ]

        for path, getter in _CANDIDATES:
            try:
                layers = VisualEncoder._as_layer_sequence(getter(model))
                if layers is not None:
                    logger.debug(f"Transformer layers found at: {path}")
                    return layers
            except AttributeError:
                continue

        # Nothing matched – build a readable attribute tree to help debugging.
        def _tree(module: nn.Module, depth: int = 0, max_depth: int = 2) -> list[str]:
            lines = []
            for name, child in module.named_children():
                lines.append("  " * depth + f".{name}  [{type(child).__name__}]")
                if depth < max_depth:
                    lines.extend(_tree(child, depth + 1, max_depth))
            return lines

        tree_str = "\n".join(_tree(model))
        raise AttributeError(
            f"\n\nCould not locate transformer encoder layers in "
            f"{type(model).__name__}.\n\n"
            f"Model attribute tree (depth ≤ 2):\n{tree_str}\n\n"
            f"Look for a ModuleList of encoder/layer blocks in the tree above, "
            f"then pass the correct dot-path via the encoder_layers_path= argument.\n"
        )

    def _build_model(self) -> None:
        """Instantiate model + processor and freeze all parameters."""
        if self.encoder_type == "clip":
            from transformers import CLIPImageProcessor, CLIPVisionModel

            self.processor = CLIPImageProcessor.from_pretrained(self._model_name)
            self.model = CLIPVisionModel.from_pretrained(self._model_name)

        else:  # ijepa
            from transformers import AutoImageProcessor, AutoModel

            self.processor = AutoImageProcessor.from_pretrained(self._model_name)
            self.model = AutoModel.from_pretrained(self._model_name)

        if self._encoder_layers_path:
            # User supplied an explicit path – resolve it via attribute traversal.
            obj = self.model
            for attr in self._encoder_layers_path.split("."):
                obj = getattr(obj, attr)
            layers = self._as_layer_sequence(obj)
            if layers is None:
                raise AttributeError(
                    f"encoder_layers_path='{self._encoder_layers_path}' resolved to "
                    f"{type(obj).__name__}, but it is not an indexable layer sequence."
                )
            self._transformer_layers = layers
            logger.info(f"Using manually specified layer path: '{self._encoder_layers_path}'")
        else:
            # CLIP and I-JEPA layouts both vary across transformers versions.
            self._transformer_layers = self._find_transformer_layers(self.model)

        # Freeze everything – this encoder is never trained
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.model.to(self.device)

        num_layers = len(self._transformer_layers)

        # If the actual depth differs from the config default, recompute the
        # S1/S2/S3 indices to keep the ~25 / ~65 / ~98 % depth ratios.
        cfg_num_layers = _ENCODER_CONFIGS[self.encoder_type]["num_layers"]
        if num_layers != cfg_num_layers:
            logger.warning(
                f"Actual layer count ({num_layers}) differs from config default "
                f"({cfg_num_layers}). Recomputing S1/S2/S3 indices proportionally."
            )
            self._s1_idx = round(num_layers * 0.25) - 1
            self._s2_idx = round(num_layers * 0.65) - 1
            self._s3_idx = num_layers - 1

        logger.info(
            f"{self.encoder_type.upper()} loaded | "
            f"total transformer layers: {num_layers} | "
            f"S1=layer{self._s1_idx}, S2=layer{self._s2_idx}, S3=layer{self._s3_idx} | "
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
                self._intermediate[scale_key] = self._pool_hidden(hidden).detach()

            return _hook

        layers = self._transformer_layers
        self._hooks.append(layers[self._s1_idx].register_forward_hook(_make_hook("S1")))
        self._hooks.append(layers[self._s2_idx].register_forward_hook(_make_hook("S2")))
        self._hooks.append(layers[self._s3_idx].register_forward_hook(_make_hook("S3")))

    def _pool_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Convert a ViT layer sequence output into one vector per image.

        CLIP prepends a CLS token, so the first token is the global image
        representation. I-JEPA exposes patch tokens only, so mean pooling is
        the correct fixed-size image representation for the lookup table.
        """
        if hidden.ndim != 3:
            raise RuntimeError(
                f"Expected transformer hidden states with shape (B, tokens, D), "
                f"got {tuple(hidden.shape)}."
            )
        if self.encoder_type == "clip":
            return hidden[:, 0, :]
        return hidden.mean(dim=1)

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

def validate_features(lookup: VisualFeatureLookup) -> None:
    """Validate that the feature bank has at least one entry with correct shapes.

    Args:
        lookup: The lookup table to validate.

    Raises:
        ValueError: If the table is empty or if feature shapes are wrong.
    """
    if len(lookup) == 0:
        raise ValueError("Visual feature bank is empty.")

    # Check the first entry
    first_key = next(iter(lookup._table))  # type: ignore[attr-defined]
    entry = lookup._table[first_key]  # type: ignore[attr-defined]
    for scale in ("S1", "S2", "S3"):
        if scale not in entry:
            raise ValueError(f"Feature entry missing key '{scale}'.")
        shape = tuple(entry[scale].shape)
        if shape != (768,):
            raise ValueError(
                f"Expected {scale} shape (768,), got {shape}."
            )


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


