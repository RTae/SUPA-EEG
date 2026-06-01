"""Extract InternViT multilayer features from THINGS-EEG images.

All model/extraction logic lives in ``src/encoders/vision_encoder.py``.
This file is the Hydra entry point and the ``ensure_internvit_features`` guard.

Usage:
  python scripts/extract_internvit_features.py
  python scripts/extract_internvit_features.py device=cuda:1
  python scripts/extract_internvit_features.py batch_size=32

Output:
  <internvit_dir>/internvit_features.npy
    dict {(concept, img_file): ndarray(n_layers, 3200) float16}
"""

from __future__ import annotations

import os

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from src.encoders.vision_encoder import extract_directory, load_internvit


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device    = torch.device(cfg.device)
    layer_ids = list(cfg.layer_ids)
    logger.info(f"Device: {device} | Layers: {layer_ids}")

    processor, model = load_internvit(cfg.internvit_model, device)

    features: dict = {}
    for image_dir in (cfg.train_img_dir, cfg.test_img_dir):
        features.update(
            extract_directory(image_dir, model, processor, layer_ids, device, cfg.batch_size)
        )

    os.makedirs(cfg.internvit_dir, exist_ok=True)
    out_path = os.path.join(cfg.internvit_dir, "internvit_features.npy")
    logger.info(f"Saving {len(features):,} entries → {out_path}")
    np.save(out_path, features)
    logger.info("Extraction complete.")


# ---------------------------------------------------------------------------
# Guard called at training startup
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


if __name__ == "__main__":
    main()
