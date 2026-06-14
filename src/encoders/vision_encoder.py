from __future__ import annotations

import os

import numpy as np
import torch
from transformers import AutoModel, CLIPImageProcessor
from loguru import logger
from src.encoders.utilities import extract_directory

OUTPUT_DIM = 3200   # InternViT-6B hidden dim (architectural constant)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_internvit(model_name: str, device: torch.device):
    """Load frozen InternViT encoder.

    Applies two runtime patches to work around incompatibilities between the
    InternViT custom model and the installed transformers version — without
    modifying any library or cached files:

    1. **meta-tensor error**: ``InternVisionEncoder.__init__`` calls
       ``torch.linspace(...).item()``. When ``device_map`` is set, transformers
       wraps ``__init__`` in ``init_empty_weights()`` making every tensor a meta
       tensor. Fix: patch ``torch.linspace`` temporarily to always run on CPU.

    2. **all_tied_weights_keys missing**: newer transformers calls
       ``model.all_tied_weights_keys`` in ``_finalize_model_loading``, but
       custom remote models don't define it. Fix: add it to ``PreTrainedModel``
       once if absent.

    Args:
        model_name: HuggingFace model ID, e.g. ``OpenGVLab/InternViT-6B-448px-V1-5``
        device:     Target torch device.

    Returns:
        ``(processor, model)`` tuple — processor is a ``CLIPImageProcessor``,
        model is frozen bfloat16 on ``device``.
    """
    from transformers import PreTrainedModel

    # Patch 1: provide all_tied_weights_keys on PreTrainedModel if absent.
    # Newer transformers calls this in _finalize_model_loading; custom remote
    # models loaded via trust_remote_code (like InternViT) don't define it.
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = property(
            lambda self: {k: None for k in (getattr(self, "_tied_weights_keys", None) or [])}
        )

    # Patch 2: make torch.linspace always produce a CPU tensor so that
    # InternVisionEncoder.__init__'s .item() call works even when transformers
    # wraps __init__ in init_empty_weights() (meta-device context) via device_map.
    _orig_linspace = torch.linspace
    def _cpu_linspace(*args, **kwargs):
        kwargs.setdefault("device", "cpu")
        return _orig_linspace(*args, **kwargs)
    torch.linspace = _cpu_linspace

    try:
        logger.info(f"Loading InternViT from {model_name}...")
        processor = CLIPImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map={"":  str(device)},
            torch_dtype=torch.bfloat16,
        )
    finally:
        torch.linspace = _orig_linspace  # always restore

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    logger.info(
        f"InternViT loaded on {device} as bfloat16. "
        f"Parameters: {sum(p.numel() for p in model.parameters()):,} (all frozen)"
    )
    return processor, model

# ---------------------------------------------------------------------------
# Feature lookup (used at training / evaluation time)
# ---------------------------------------------------------------------------

class InternViTFeatureLookup:
    """InternViT feature store keyed by (concept, image_file).

    Loads a single ``internvit_features.npy`` produced by
    ``scripts/extract_internvit_features.py``::

        {(concept_str, img_file_str): np.ndarray(n_layers, 3200) float16}

    Args:
        feature_path: path to ``internvit_features.npy``
    """

    def __init__(self, feature_path: str) -> None:
        if feature_path.endswith(".pt"):
            payload = torch.load(feature_path, map_location="cpu", weights_only=True)
            table = payload.get("features", payload)
            self.features = {
                key: np.stack(
                    [
                        np.asarray(value["S1"], dtype=np.float32),
                        np.asarray(value["S2"], dtype=np.float32),
                        np.asarray(value["S3"], dtype=np.float32),
                    ]
                )
                for key, value in table.items()
            }
        else:
            self.features = np.load(feature_path, allow_pickle=True).item()

    def retrieve_batch(
        self,
        concepts: list[str],
        image_files: list[str],
    ) -> torch.Tensor:
        """Return ``(batch, n_layers, d)`` float32 tensor (d depends on feature bank)."""
        arrs = [
            np.asarray(self.features[(c, f)], dtype=np.float32)
            for c, f in zip(concepts, image_files)
        ]
        return torch.from_numpy(np.stack(arrs))

    def __len__(self) -> int:
        return len(self.features)


# ---------------------------------------------------------------------------
# Feature guard 
# ---------------------------------------------------------------------------
def ensure_internvit_features(
    internvit_dir: str,
    layer_ids: list[int],
    train_img_dir: str,
    test_img_dir: str,
    model_name: str = "OpenGVLab/InternViT-6B-448px-V1-5",
    device: str = "cpu",
    batch_size: int = 8,
) -> None:
    """Check if InternViT features exist; extract if missing.

    Called automatically at the start of training (no-op if the file is present).

    Args:
        internvit_dir:  Directory for the output ``.npy`` file.
        layer_ids:      e.g. ``[20, 24, 28, 32, 36]``
        train_img_dir:  ``data/things_eeg/training_images``
        test_img_dir:   ``data/things_eeg/test_images``
        model_name:     HuggingFace model ID.
        device:         Device for InternViT extraction.
        batch_size:     Images per batch during extraction.
    """
    out_path = os.path.join(internvit_dir, "internvit_features.npy")

    if os.path.isfile(out_path):
        logger.info(f"InternViT features found at {out_path}. Skipping extraction.")
        return

    logger.warning(
        f"InternViT feature file not found: {out_path}\n"
        "Running offline extraction — this may take 10-30 minutes."
    )

    _device = torch.device(device)
    processor, model = load_internvit(model_name, _device)

    features: dict = {}
    for image_dir in (train_img_dir, test_img_dir):
        if not os.path.isdir(image_dir):
            logger.warning(f"Image directory not found, skipping: {image_dir}")
            continue
        features.update(
            extract_directory(image_dir, model, processor, layer_ids, _device, batch_size)
        )

    os.makedirs(internvit_dir, exist_ok=True)
    np.save(out_path, features)

    if not os.path.isfile(out_path):
        raise FileNotFoundError(f"Extraction completed but file still missing: {out_path}")
    logger.info(f"InternViT features saved to {out_path} ({len(features):,} images).")
