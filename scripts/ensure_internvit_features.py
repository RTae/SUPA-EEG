"""Helper: ensure InternViT features are extracted before training begins."""

from __future__ import annotations

import os

from loguru import logger


def ensure_internvit_features(
    internvit_dir: str,
    layer_ids: list[int],
    train_img_dir: str,
    test_img_dir: str,
    metadata_path: str,
    device: str = "cpu",
    batch_size: int = 64,
) -> None:
    """Check if InternViT features exist. Extract if missing.

    Called automatically at the start of training.
    Checks for ALL required .npy files before deciding to re-extract.
    If ANY file is missing, runs full extraction for both splits.

    Args:
        internvit_dir:  Output directory for .npy files
        layer_ids:      e.g. [20, 24, 28, 32, 36]
        train_img_dir:  data/things_eeg/training_images
        test_img_dir:   data/things_eeg/test_images
        metadata_path:  data/things_eeg/image_metadata.npy
        device:         device for InternViT extraction
        batch_size:     images per batch during extraction
    """
    # Build list of all required files
    required = []
    for split in ("train", "test"):
        for lid in layer_ids:
            required.append(
                os.path.join(internvit_dir, f"image_{split}_layer{lid}.npy")
            )

    missing = [f for f in required if not os.path.isfile(f)]

    if not missing:
        logger.info(
            f"InternViT features found ({len(required)} files). Skipping extraction."
        )
        return

    logger.warning(
        f"{len(missing)}/{len(required)} InternViT feature files missing. "
        "Running offline extraction — this may take 10-30 minutes."
    )
    for f in missing:
        logger.warning(f"  Missing: {f}")

    # Import and run extraction
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "scripts/extract_internvit_features.py",
        "--output_dir",    internvit_dir,
        "--train_img_dir", train_img_dir,
        "--test_img_dir",  test_img_dir,
        "--metadata_path", metadata_path,
        "--device",        device,
        "--batch_size",    str(batch_size),
        "--layer_ids",
    ] + [str(lid) for lid in layer_ids]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError("InternViT feature extraction failed.")

    # Verify all files now exist
    still_missing = [f for f in required if not os.path.isfile(f)]
    if still_missing:
        raise FileNotFoundError(
            f"Extraction completed but {len(still_missing)} files still missing: "
            f"{still_missing}"
        )
    logger.info("InternViT feature extraction complete. All files verified.")
