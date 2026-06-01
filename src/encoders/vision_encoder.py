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

    Loads on CPU with ``torch_dtype=bfloat16`` (avoids 24 GB float32 copy),
    then moves to ``device``.  ``device_map`` is intentionally omitted —
    it triggers ``caching_allocator_warmup`` which calls
    ``model.all_tied_weights_keys`` that InternViT doesn't define.
    The cached ``modeling_intern_vit.py`` has been patched separately so that
    ``torch.linspace(..., device='cpu')`` is used in ``InternVisionEncoder.__init__``
    to avoid meta-tensor errors.

    Args:
        model_name: HuggingFace model ID, e.g. ``OpenGVLab/InternViT-6B-448px-V1-5``
        device:     Target torch device.

    Returns:
        ``(processor, model)`` tuple — processor is a ``CLIPImageProcessor``,
        model is frozen bfloat16 on ``device``.
    """
    logger.info(f"Loading InternViT from {model_name}...")
    processor = CLIPImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,   # load as bfloat16 on CPU (~12 GB, not 24 GB)
    )
    model = model.to(device=device)
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
        self.features: dict = np.load(feature_path, allow_pickle=True).item()

    def retrieve_batch(
        self,
        concepts: list[str],
        image_files: list[str],
    ) -> torch.Tensor:
        """Return ``(batch, n_layers, 3200)`` float32 tensor."""
        arrs = [
            self.features[(c, f)].astype(np.float32)
            for c, f in zip(concepts, image_files)
        ]
        return torch.from_numpy(np.stack(arrs))

    def __len__(self) -> int:
        return len(self.features)


# ---------------------------------------------------------------------------
# Feature guard (called at training startup)
# ---------------------------------------------------------------------------

def ensure_internvit_features(
    internvit_dir: str,
    layer_ids: list[int],
    train_img_dir: str,
    test_img_dir: str,
    model_name: str = "OpenGVLab/InternViT-6B-448px-V1-5",
    device: str = "cpu",
    batch_size: int = 64,
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
        features.update(
            extract_directory(image_dir, model, processor, layer_ids, _device, batch_size)
        )

    os.makedirs(internvit_dir, exist_ok=True)
    np.save(out_path, features)

    if not os.path.isfile(out_path):
        raise FileNotFoundError(f"Extraction completed but file still missing: {out_path}")
    logger.info(f"InternViT features saved to {out_path} ({len(features):,} images).")
