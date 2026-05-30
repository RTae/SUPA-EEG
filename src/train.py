"""SUPAEEG training script.

Run from the project root::

    python src/train.py

Or with a custom subject::

    python src/train.py --subject 1

The script:
  1. Ensures the CLIP visual feature bank is available on disk.
  2. Builds ThingsEEGDataset train / test splits.
  3. Trains SUPAEEG with InfoNCE + Gaussian regulariser + L1 sparsity.
  4. Evaluates every ``eval_every`` epochs using the THINGS-EEG retrieval
     protocol (Top-1 / Top-5 zero-shot concept retrieval).
  5. Saves the best checkpoint to ``checkpoints/supaeeg_best.pt``.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so all src.* imports resolve.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dataset import ThingsEEGDataset  # noqa: E402
from src.encoders.visual_encoder import VisualEncoder, VisualFeatureLookup  # noqa: E402
from src.models.supaeeg import SUPAEEG  # noqa: E402
from src.trainer.metrics import retrieve_all  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Training hyperparameters and path configuration."""

    dataset_dir: str = "data/things_eeg"
    feature_path: str = "data/vision_encoder/clip/visual_features_clip.pt"
    checkpoint_dir: str = "checkpoints"

    subject: int = -1

    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 100
    eval_every: int = 5

    lambda_reg: float = 0.1
    beta_l1: float = 0.01
    tau: float = 0.07

    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1

    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def collate_fn(batch: list[tuple[Any, ...]]) -> dict[str, Any]:
    """Custom collate for ThingsEEGDataset batches.

    ThingsEEGDataset.__getitem__ returns::

        (eeg_tensor, image_tensor, subject_index, repetition_index,
         data_index, image_concept, image_file)

    Only ``eeg_tensor`` (index 0), ``image_concept`` (index 5), and
    ``image_file`` (index 6) are used.  The rest are discarded.

    Args:
        batch: List of tuples from the dataset.

    Returns:
        Dict with keys ``'eeg'``, ``'image_concepts'``, ``'image_files'``.
    """
    eeg_tensors = torch.stack([item[0] for item in batch], dim=0)
    image_concepts = [item[5] for item in batch]
    image_files = [item[6] for item in batch]
    return {
        "eeg": eeg_tensors,
        "image_concepts": image_concepts,
        "image_files": image_files,
    }


# ---------------------------------------------------------------------------
# Visual feature bank
# ---------------------------------------------------------------------------


def ensure_visual_features(config: Config) -> VisualFeatureLookup:
    """Load or extract the CLIP visual feature bank.

    Step 1 — If the feature file already exists, load and return it.
    Step 2 — Otherwise, run offline CLIP extraction over all training and
              test images, save the result, and return it.
    Step 3 — Validate that the loaded features contain at least one entry
              with correct S1/S2/S3 shapes.

    Args:
        config: Training configuration.

    Returns:
        Populated VisualFeatureLookup.

    Raises:
        ValueError: If the loaded features fail shape validation.
    """
    path = config.feature_path

    if os.path.isfile(path):
        logger.info("Visual features found. Loading from disk...")
        lookup = VisualFeatureLookup(path)
        _validate_features(lookup)
        return lookup

    # ------------------------------------------------------------------
    # Re-extract from scratch
    # ------------------------------------------------------------------
    logger.warning("Visual features not found. Running offline extraction...")

    encoder = VisualEncoder(
        encoder_type="clip",
        model_name="openai/clip-vit-base-patch32",
        device=config.device,
    )

    feature_dict: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

    image_base = Path(config.dataset_dir)
    splits = [image_base / "training_images", image_base / "test_images"]

    from PIL import Image  # local import — only needed during extraction

    for split_dir in splits:
        if not split_dir.is_dir():
            logger.warning(f"Image directory not found, skipping: {split_dir}")
            continue

        for concept_dir in sorted(split_dir.iterdir()):
            if not concept_dir.is_dir():
                continue
            concept = concept_dir.name

            for img_path in sorted(concept_dir.iterdir()):
                if not img_path.is_file():
                    continue

                try:
                    pil_image = Image.open(img_path).convert("RGB")
                except Exception as exc:
                    logger.warning(f"Could not open {img_path}: {exc}")
                    continue

                pixel_values = encoder.preprocess([pil_image])
                features = encoder.forward(pixel_values)  # {"S1":…, "S2":…, "S3":…}

                key = (concept, img_path.name)
                feature_dict[key] = {
                    "S1": features["S1"].squeeze(0),
                    "S2": features["S2"].squeeze(0),
                    "S3": features["S3"].squeeze(0),
                }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "features": feature_dict,
            "encoder_type": "clip",
            "model_name": "openai/clip-vit-base-patch32",
            "layer_indices": {"S1": 3, "S2": 7, "S3": 11},
        },
        path,
    )
    logger.info(f"Visual features saved to {path}")

    lookup = VisualFeatureLookup(path)
    _validate_features(lookup)
    return lookup


def _validate_features(lookup: VisualFeatureLookup) -> None:
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


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def _evaluate(
    model: SUPAEEG,
    test_loader: DataLoader,
    feature_lookup: VisualFeatureLookup,
    device: torch.device,
) -> tuple[float, float]:
    """Run zero-shot concept retrieval evaluation on the test set.

    Aggregates per-concept EEG embeddings (averaged over repetitions) and
    paired CLIP image embeddings, then computes Top-1 and Top-5 retrieval
    accuracy.

    Args:
        model:          The SUPAEEG model in eval mode.
        test_loader:    DataLoader over the test split.
        feature_lookup: Visual feature bank.
        device:         Compute device.

    Returns:
        Tuple ``(top1, top5)`` accuracy values in [0, 1].
    """
    model.eval()

    # ------------------------------------------------------------------
    # Step 1 — collect EEG embeddings per concept
    # ------------------------------------------------------------------
    concept_embeddings: dict[str, list[torch.Tensor]] = defaultdict(list)
    concept_to_file: dict[str, str] = {}

    for batch in test_loader:
        eeg = batch["eeg"].to(device)
        z = model.embed(eeg)  # (batch, 2304), already no_grad inside
        for i, (concept, img_file) in enumerate(
            zip(batch["image_concepts"], batch["image_files"])
        ):
            concept_embeddings[concept].append(z[i].cpu())
            concept_to_file[concept] = img_file

    concept_order = sorted(concept_embeddings.keys())

    # Average over repetitions and ℓ2-normalise
    eeg_features = torch.cat(
        [
            F.normalize(
                torch.stack(concept_embeddings[c]).mean(dim=0, keepdim=True),
                dim=1,
            )
            for c in concept_order
        ],
        dim=0,
    ).numpy()  # (N_concepts, 2304)

    # ------------------------------------------------------------------
    # Step 2 — collect image embeddings per concept
    # ------------------------------------------------------------------
    image_features = torch.cat(
        [
            F.normalize(
                torch.cat(
                    [*feature_lookup.retrieve(c, concept_to_file[c])]
                ).unsqueeze(0),
                dim=1,
            )
            for c in concept_order
        ],
        dim=0,
    ).numpy()  # (N_concepts, 2304)

    # ------------------------------------------------------------------
    # Step 3 — retrieval metrics
    # ------------------------------------------------------------------
    top5_count, top1_count, total = retrieve_all(eeg_features, image_features)
    top1 = top1_count / total
    top5 = top5_count / total
    return top1, top5


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(config: Config) -> None:
    """Main training entry point.

    Args:
        config: Fully populated training configuration.
    """
    device = torch.device(config.device)
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Visual feature bank
    # ------------------------------------------------------------------
    feature_lookup = ensure_visual_features(config)

    # ------------------------------------------------------------------
    # 2 & 3. Datasets
    # ------------------------------------------------------------------
    train_dataset = ThingsEEGDataset(
        dataset_dir=config.dataset_dir,
        data_type="train",
        subject=config.subject,
    )
    test_dataset = ThingsEEGDataset(
        dataset_dir=config.dataset_dir,
        data_type="test",
        subject=config.subject,
    )

    # ------------------------------------------------------------------
    # 4 & 5. Data loaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # 6. Model
    # ------------------------------------------------------------------
    model = SUPAEEG(
        feature_lookup=feature_lookup,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        device=device,
    ).to(device)

    # ------------------------------------------------------------------
    # 7. Optimiser
    # ------------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_top1: float = 0.0

    # ------------------------------------------------------------------
    # Training epochs
    # ------------------------------------------------------------------
    for epoch in range(1, config.epochs + 1):
        model.train()

        for batch in train_loader:
            eeg = batch["eeg"].to(device)
            image_concepts: list[str] = batch["image_concepts"]
            image_files: list[str] = batch["image_files"]

            z1, z2, z3, loss, components = model(
                eeg, image_concepts, image_files, return_loss=True
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info(
            f"Epoch {epoch}/{config.epochs} | "
            f"total={components['total']:.4f} | "
            f"infonce={components['infonce']:.4f} | "
            f"sigreg={components['sigreg']:.4f} | "
            f"l1={components['l1']:.4f}"
        )

        # ------------------------------------------------------------------
        # Periodic evaluation
        # ------------------------------------------------------------------
        if epoch % config.eval_every == 0:
            top1, top5 = _evaluate(model, test_loader, feature_lookup, device)
            logger.info(
                f"Eval  epoch {epoch} | Top-1: {top1:.4f} | Top-5: {top5:.4f}"
            )

            if top1 > best_top1:
                best_top1 = top1
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "top1": top1,
                        "top5": top5,
                        "config": config,
                    },
                    f"{config.checkpoint_dir}/supaeeg_best.pt",
                )
                logger.info(
                    f"New best | Top-1: {top1:.4f} | Top-5: {top5:.4f}"
                )

    logger.info(f"Training complete. Best Top-1: {best_top1:.4f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> Config:
    """Parse command-line overrides and return a Config instance."""
    parser = argparse.ArgumentParser(description="Train SUPAEEG")
    parser.add_argument("--subject", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--dataset_dir", type=str, default="data/things_eeg")
    parser.add_argument(
        "--feature_path",
        type=str,
        default="data/vision_encoder/clip/visual_features_clip.pt",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = Config(
        subject=args.subject,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_every=args.eval_every,
        checkpoint_dir=args.checkpoint_dir,
        dataset_dir=args.dataset_dir,
        feature_path=args.feature_path,
    )
    if args.device is not None:
        cfg.device = args.device
    return cfg


if __name__ == "__main__":
    cfg = _parse_args()
    train(cfg)
