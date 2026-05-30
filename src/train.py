from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import typer
from loguru import logger
from torch.utils.data import ConcatDataset, DataLoader

from src.dataset import ThingsEEGDataset
from src.encoders.visual_encoder import VisualEncoder, VisualFeatureLookup, validate_features
from src.utilities import (
    Config,
    evaluate,
    log_results_table,
    make_model,
    make_optimizer,
    save_checkpoint,
    train_one_epoch,
)

app = typer.Typer()


# ---------------------------------------------------------------------------
# Collate function
# --------------------------------------------------------------------------
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
def ensure_visual_features(
    feature_path: str,
    device: str,
    dataset_dir: str,
) -> VisualFeatureLookup:
    """Load or extract the CLIP visual feature bank.

    Step 1 — If the feature file already exists, load and return it.
    Step 2 — Otherwise, run offline CLIP extraction over all training and
              test images, save the result, and return it.
    Step 3 — Validate that the loaded features contain at least one entry
              with correct S1/S2/S3 shapes.

    Args:
        feature_path: Path to the ``.pt`` feature bank file.
        device:       Compute device string (e.g. ``"cuda"`` or ``"cpu"``).
        dataset_dir:  Root directory containing ``training_images/`` and
                      ``test_images/`` subdirectories.

    Returns:
        Populated VisualFeatureLookup.

    Raises:
        ValueError: If the loaded features fail shape validation.
    """
    path = feature_path

    if os.path.isfile(path):
        logger.info("Visual features found. Loading from disk...")
        lookup = VisualFeatureLookup(path)
        validate_features(lookup)
        return lookup

    # ------------------------------------------------------------------
    # Re-extract from scratch
    # ------------------------------------------------------------------
    logger.warning("Visual features not found. Running offline extraction...")

    encoder = VisualEncoder(
        encoder_type="clip",
        model_name="openai/clip-vit-base-patch32",
        device=device,
    )

    feature_dict: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

    image_base = Path(dataset_dir)
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
    validate_features(lookup)
    return lookup

# ---------------------------------------------------------------------------
# Protocol runners
# ---------------------------------------------------------------------------


def run_intra_subject(
    config: Config,
    feature_lookup: VisualFeatureLookup,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    """Train and evaluate one model per subject (intra-subject protocol).

    If ``config.subject`` is -1, iterates over all subjects in
    ``config.all_subjects``; otherwise trains a single subject.

    Args:
        config:         Runtime configuration.
        feature_lookup: Pre-loaded CLIP feature bank (shared across folds).
        device:         Compute device.

    Returns:
        Mapping of subject_id → {``'top1'``: float, ``'top5'``: float}.
    """
    subjects = (
        [config.subject] if config.subject != -1 else config.all_subjects
    )
    all_results: dict[int, dict[str, float]] = {}

    for subject_id in subjects:
        logger.info(f"Intra-subject | subject={subject_id}")

        train_dataset = ThingsEEGDataset(
            dataset_dir=config.dataset_dir,
            data_type="train",
            subject=subject_id,
        )
        test_dataset = ThingsEEGDataset(
            dataset_dir=config.dataset_dir,
            data_type="test",
            subject=subject_id,
        )
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

        model = make_model(feature_lookup, config, device)
        optimizer = make_optimizer(model, config)
        best_top1 = 0.0
        best_top5 = 0.0

        for epoch in range(1, config.epochs + 1):
            mean_loss = train_one_epoch(model, train_loader, optimizer, device)
            logger.info(
                f"Sub{subject_id:02d} | epoch {epoch}/{config.epochs} | loss={mean_loss:.4f}"
            )

            if epoch % config.eval_every == 0:
                top1, top5 = evaluate(model, test_loader, feature_lookup, device)
                logger.info(
                    f"Sub{subject_id:02d} | eval epoch {epoch} | "
                    f"Top-1: {top1:.4f} | Top-5: {top5:.4f}"
                )
                if top1 > best_top1:
                    best_top1 = top1
                    best_top5 = top5
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        top1,
                        top5,
                        path=f"{config.checkpoint_dir}/supaeeg_intra_sub{subject_id:02d}.pt",
                    )

        all_results[subject_id] = {"top1": best_top1, "top5": best_top5}

    avg_top1 = sum(r["top1"] for r in all_results.values()) / len(all_results)
    avg_top5 = sum(r["top5"] for r in all_results.values()) / len(all_results)
    log_results_table(all_results, avg_top1, avg_top5, protocol="intra")
    return all_results


def run_inter_subject(
    config: Config,
    feature_lookup: VisualFeatureLookup,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    """Leave-one-subject-out (LOSO) cross-subject training.

    For each test subject, trains on the remaining 9 subjects' data combined
    via ``ConcatDataset``, then evaluates on the left-out subject's test set.
    A fresh model and optimiser are created for every fold.

    Args:
        config:         Runtime configuration.
        feature_lookup: Pre-loaded CLIP feature bank (shared across folds).
        device:         Compute device.

    Returns:
        Mapping of test_subject_id → {``'top1'``: float, ``'top5'``: float}.
    """
    all_results: dict[int, dict[str, float]] = {}

    for test_subject in config.all_subjects:
        train_subjects = [s for s in config.all_subjects if s != test_subject]
        logger.info(
            f"LOSO | test_subject={test_subject} | train_subjects={train_subjects}"
        )

        train_dataset = ConcatDataset(
            [
                ThingsEEGDataset(
                    dataset_dir=config.dataset_dir,
                    data_type="train",
                    subject=s,
                )
                for s in train_subjects
            ]
        )
        test_dataset = ThingsEEGDataset(
            dataset_dir=config.dataset_dir,
            data_type="test",
            subject=test_subject,
        )
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

        model = make_model(feature_lookup, config, device)
        optimizer = make_optimizer(model, config)
        best_top1 = 0.0
        best_top5 = 0.0

        for epoch in range(1, config.epochs + 1):
            mean_loss = train_one_epoch(model, train_loader, optimizer, device)
            logger.info(
                f"LOSO test=Sub{test_subject:02d} | epoch {epoch}/{config.epochs} | "
                f"loss={mean_loss:.4f}"
            )

            if epoch % config.eval_every == 0:
                top1, top5 = evaluate(model, test_loader, feature_lookup, device)
                logger.info(
                    f"LOSO test=Sub{test_subject:02d} | eval epoch {epoch} | "
                    f"Top-1: {top1:.4f} | Top-5: {top5:.4f}"
                )
                if top1 > best_top1:
                    best_top1 = top1
                    best_top5 = top5
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        top1,
                        top5,
                        path=f"{config.checkpoint_dir}/supaeeg_loso_sub{test_subject:02d}.pt",
                    )

        all_results[test_subject] = {"top1": best_top1, "top5": best_top5}

    avg_top1 = sum(r["top1"] for r in all_results.values()) / len(all_results)
    avg_top5 = sum(r["top5"] for r in all_results.values()) / len(all_results)
    log_results_table(all_results, avg_top1, avg_top5, protocol="inter")
    return all_results


# ---------------------------------------------------------------------------
# Typer entry point
# ---------------------------------------------------------------------------


@app.command()
def train(
    protocol: str = typer.Option(
        "intra", help="Training protocol: 'intra' (per-subject) or 'inter' (LOSO)"
    ),
    dataset_dir: str = typer.Option("data/things_eeg", help="THINGS-EEG2 root directory"),
    subject: int = typer.Option(
        1, help="Subject index for intra protocol (1–10); -1 = all subjects"
    ),
    feature_path: str = typer.Option(
        "data/vision_encoder/clip/visual_features_clip.pt",
        help="Path to the CLIP visual feature bank",
    ),
    device: str = typer.Option(
        os.environ.get("DEVICE", "cuda"),
        help="Compute device (overridden by DEVICE env var)",
    ),
    epochs: int = typer.Option(100, help="Training epochs"),
    batch_size: int = typer.Option(256, help="Batch size"),
    eval_every: int = typer.Option(5, help="Evaluate every N epochs"),
    lambda_reg: float = typer.Option(0.1, help="Gaussian regulariser weight"),
    beta_l1: float = typer.Option(0.01, help="Channel-attention L1 sparsity weight"),
    tau: float = typer.Option(0.07, help="InfoNCE temperature"),
    d_model: int = typer.Option(256, help="Token embedding dimension"),
    nhead: int = typer.Option(8, help="Transformer attention heads"),
    num_layers: int = typer.Option(4, help="Transformer depth"),
    dim_feedforward: int = typer.Option(512, help="FFN hidden size"),
    dropout: float = typer.Option(0.1, help="Dropout"),
    lr: float = typer.Option(3e-4, help="Learning rate"),
    weight_decay: float = typer.Option(1e-4, help="Weight decay"),
    checkpoint_dir: str = typer.Option(
        "outputs/supaeeg", help="Directory for saved checkpoints"
    ),
) -> None:
    """Train SUPAEEG on THINGS-EEG2 using the intra- or inter-subject protocol."""
    if protocol not in ("intra", "inter"):
        raise typer.BadParameter(f"protocol must be 'intra' or 'inter', got {protocol!r}")

    cfg = Config(
        protocol=protocol,
        subject=subject,
        dataset_dir=dataset_dir,
        feature_path=feature_path,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        eval_every=eval_every,
        lambda_reg=lambda_reg,
        beta_l1=beta_l1,
        tau=tau,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        checkpoint_dir=checkpoint_dir,
    )
    logger.info(
        f"Protocol={cfg.protocol!r} subject={cfg.subject} device={cfg.device!r} "
        f"epochs={cfg.epochs} batch_size={cfg.batch_size} lr={cfg.lr}"
    )

    _device = torch.device(
        cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    )
    logger.info(f"Device: {_device}")

    # Load the CLIP feature bank once — shared across all subject folds
    feature_lookup = ensure_visual_features(cfg.feature_path, cfg.device, cfg.dataset_dir)

    if cfg.protocol == "intra":
        run_intra_subject(cfg, feature_lookup, _device)
    else:
        run_inter_subject(cfg, feature_lookup, _device)


if __name__ == "__main__":
    app()
