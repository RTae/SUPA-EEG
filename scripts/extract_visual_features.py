"""
Offline Visual Feature Extraction for THINGS-EEG
=================================================

Iterates over every unique image in the THINGS-EEG training and/or test
image directories, passes them through a frozen CLIP or I-JEPA encoder, and
saves a lookup table::

    {
        (concept, image_file): {"S1": Tensor, "S2": Tensor, "S3": Tensor}
    }

to a ``.pt`` file on disk.  The table is loaded at training time by
``src.model.visual_encoder.VisualFeatureLookup``.

Architecture alignment
----------------------
This script implements the **Offline** phase from the project diagram:

    [THINGS-EEG Images]
        → CLIP / I-JEPA encoder (frozen, pretrained)
        → Separate features into S1 (early), S2 (mid), S3 (late) layers
        → Stored as fixed lookup table  ← this file produces that table

Usage
-----
    # CLIP (default)
    python scripts/extract_visual_features.py

    # I-JEPA
    python scripts/extract_visual_features.py --encoder ijepa

    # Custom checkpoint and output path
    python scripts/extract_visual_features.py \\
        --encoder clip \\
        --model_name openai/clip-vit-large-patch14 \\
        --dataset_dir data/things_eeg \\
        --splits train test \\
        --output_path data/visual_features_clip_large.pt \\
        --batch_size 64 \\
        --device cuda

    # Dry-run (first 20 concepts only, useful for sanity-checking)
    python scripts/extract_visual_features.py --max_concepts 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterator  # used in _iter_splits return type

import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.visual_encoder import VisualEncoder  # noqa: E402


# ---------------------------------------------------------------------------
# Lightweight dataset that yields (concept, image_file, PIL.Image) triples
# ---------------------------------------------------------------------------


class _ImageFolderDataset(Dataset):
    """Flat dataset over a concept-organised image directory.

    Expected layout::

        <image_dir>/
            <concept_A>/
                img1.jpg
                img2.jpg
            <concept_B>/
                ...

    Each ``__getitem__`` call returns ``(concept, image_file, pil_image)`` so
    the DataLoader can batch PIL images together for processing.

    Args:
        image_dir:    Root directory containing concept sub-folders.
        max_concepts: If given, only the first *N* concepts are included
                      (useful for quick sanity-checks).
    """

    def __init__(self, image_dir: str | Path, max_concepts: int | None = None) -> None:
        self.image_dir = Path(image_dir)
        self.entries: list[tuple[str, str, Path]] = []  # (concept, filename, path)

        concepts = sorted(
            c for c in os.listdir(self.image_dir) if (self.image_dir / c).is_dir()
        )
        if max_concepts is not None:
            concepts = concepts[:max_concepts]

        for concept in concepts:
            concept_dir = self.image_dir / concept
            for img_file in sorted(os.listdir(concept_dir)):
                img_path = concept_dir / img_file
                if img_path.is_file() and img_path.suffix.lower() in {
                    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
                }:
                    self.entries.append((concept, img_file, img_path))

        logger.info(
            f"ImageFolderDataset | dir={self.image_dir} | "
            f"concepts={len(concepts)} | images={len(self.entries)}"
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[str, str, Image.Image]:
        concept, img_file, img_path = self.entries[index]
        image = Image.open(img_path).convert("RGB")
        return concept, img_file, image


def _collate_pil(
    batch: list[tuple[str, str, Image.Image]],
) -> tuple[list[str], list[str], list[Image.Image]]:
    """Custom collate that keeps PIL images as a plain list."""
    concepts, img_files, images = zip(*batch)
    return list(concepts), list(img_files), list(images)


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------


def _iter_splits(
    dataset_dir: Path,
    splits: list[str],
) -> Iterator[tuple[str, Path]]:
    """Yield (split_name, image_dir) for each valid requested split."""
    split_dir_map = {
        "train": "training_images",
        "test": "test_images",
    }
    for split in splits:
        dir_name = split_dir_map.get(split, split)
        image_dir = dataset_dir / dir_name
        if not image_dir.is_dir():
            logger.warning(f"Image directory not found, skipping split '{split}': {image_dir}")
            continue
        yield split, image_dir


def extract_features(
    encoder: VisualEncoder,
    image_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    max_concepts: int | None = None,
) -> dict[tuple[str, str], dict[str, torch.Tensor]]:
    """Extract S1 / S2 / S3 features for every image under ``image_dir``.

    Args:
        encoder:       Initialised (and frozen) ``VisualEncoder`` instance.
        image_dir:     Root of the concept-organised image directory.
        batch_size:    Number of images processed per forward pass.
        num_workers:   DataLoader worker processes for image loading.
        max_concepts:  Limit to the first N concepts (for debugging).

    Returns:
        ``features``: dict mapping ``(concept, image_file)`` → ``{S1, S2, S3}``.
    """
    dataset = _ImageFolderDataset(image_dir, max_concepts=max_concepts)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_pil,
        pin_memory=False,
    )

    features: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    total = len(dataset)
    processed = 0

    logger.info(f"Starting extraction | total images={total} | batch_size={batch_size}")

    for concepts, img_files, pil_images in loader:
        # Preprocess: PIL list → normalised pixel_values tensor on encoder.device
        pixel_values = encoder.preprocess(pil_images)

        # Forward pass returns {S1: (B,D), S2: (B,D), S3: (B,D)} on CPU
        scale_features = encoder(pixel_values)

        # Un-batch and store in the lookup dict
        for i, (concept, img_file) in enumerate(zip(concepts, img_files)):
            features[(concept, img_file)] = {
                "S1": scale_features["S1"][i],
                "S2": scale_features["S2"][i],
                "S3": scale_features["S3"][i],
            }

        processed += len(concepts)
        if processed % max(batch_size * 10, 1) == 0 or processed == total:
            logger.info(f"  Progress: {processed}/{total} images ({100*processed/total:.1f}%)")

    logger.info(f"Extraction complete | extracted {len(features):,} feature vectors")
    return features


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline visual feature extraction for THINGS-EEG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["clip", "ijepa"],
        default="clip",
        help="Which encoder variant to use.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=(
            "HuggingFace model identifier. "
            "Defaults to 'openai/clip-vit-base-patch32' for CLIP and "
            "'facebook/ijepa_vith14_1k' for I-JEPA."
        ),
    )
    parser.add_argument(
        "--encoder_layers_path",
        type=str,
        default=None,
        help=(
            "Optional dot path to the transformer layer list inside the model. "
            "Useful for custom I-JEPA checkpoints if automatic discovery fails, "
            "for example 'encoder.layer'."
        ),
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/things_eeg",
        help="Root directory of the THINGS-EEG dataset.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "test"],
        default=["train", "test"],
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=(
            "Path for the output .pt file. "
            "Defaults to 'data/visual_features_<encoder>.pt'."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of images per encoder forward pass.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes for image loading.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Compute device, e.g. 'cpu', 'cuda', 'cuda:0', 'mps'. "
            "Auto-detected if not specified."
        ),
    )
    parser.add_argument(
        "--max_concepts",
        type=int,
        default=None,
        help="Limit extraction to the first N concepts (for debugging).",
    )
    return parser.parse_args()


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────────────────
    device = args.device or _auto_device()
    logger.info(f"Using device: {device}")

    # ── Dataset splits ──────────────────────────────────────────────────────
    dataset_dir = Path(args.dataset_dir)
    split_dirs = list(_iter_splits(dataset_dir, args.splits))
    if not split_dirs:
        requested = ", ".join(args.splits)
        raise FileNotFoundError(
            f"No requested image split directories were found under '{dataset_dir}'. "
            f"Requested splits: {requested}."
        )

    # ── Output path ─────────────────────────────────────────────────────────
    output_path = Path(
        args.output_path or f"data/visual_features_{args.encoder}.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Build encoder ────────────────────────────────────────────────────────
    encoder = VisualEncoder(
        encoder_type=args.encoder,
        model_name=args.model_name,
        device=device,
        encoder_layers_path=args.encoder_layers_path,
    )

    # ── Extract for each split ───────────────────────────────────────────────
    all_features: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

    for split, image_dir in split_dirs:
        logger.info(f"--- Processing split '{split}' from {image_dir} ---")
        split_features = extract_features(
            encoder=encoder,
            image_dir=image_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_concepts=args.max_concepts,
        )
        # Merge; duplicate keys (same image appearing in both splits) are overwritten.
        all_features.update(split_features)

    if not all_features:
        raise RuntimeError(
            "No image features were extracted. Check that the requested split "
            "directories contain concept folders with supported image files."
        )

    # ── Save lookup table ────────────────────────────────────────────────────
    payload = {
        "features": all_features,
        "encoder_type": args.encoder,
        "model_name": encoder._model_name,
        "layer_indices": {
            "S1": encoder._s1_idx,
            "S2": encoder._s2_idx,
            "S3": encoder._s3_idx,
        },
        "num_entries": len(all_features),
        "splits": args.splits,
    }

    logger.info(f"Saving lookup table → {output_path} ({len(all_features):,} entries)...")
    torch.save(payload, output_path)
    logger.success(f"Done. Visual features saved to '{output_path}'.")

    # ── Quick sanity check ───────────────────────────────────────────────────
    logger.info("Running sanity check: reloading and verifying first entry...")
    loaded = torch.load(output_path, map_location="cpu", weights_only=True)
    first_key = next(iter(loaded["features"]))
    first_val = loaded["features"][first_key]
    logger.info(
        f"  Key  : {first_key}\n"
        f"  S1   : shape={tuple(first_val['S1'].shape)}, dtype={first_val['S1'].dtype}\n"
        f"  S2   : shape={tuple(first_val['S2'].shape)}, dtype={first_val['S2'].dtype}\n"
        f"  S3   : shape={tuple(first_val['S3'].shape)}, dtype={first_val['S3'].dtype}"
    )
    logger.success("Sanity check passed.")


if __name__ == "__main__":
    main()
