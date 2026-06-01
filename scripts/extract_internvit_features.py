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
import sys

# Ensure project root is on sys.path when the script is run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from src.encoders.vision_encoder import load_internvit
from src.encoders.utilities import extract_directory


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


if __name__ == "__main__":
    main()
