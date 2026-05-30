"""SUPAEEG training script — intra-subject and inter-subject (LOSO) protocols.

Run from the project root::

    # Intra-subject: train/test on subject 1 only
    python src/train.py --subject 1

    # Intra-subject: all 10 subjects sequentially
    python src/train.py --subject -1

    # Inter-subject: LOSO across all 10 subjects
    python src/train.py --protocol inter

    # CPU training
    python src/train.py --device cpu

The script:
  1. Ensures the CLIP visual feature bank is available on disk.
  2. Dispatches to the intra- or inter-subject protocol runner.
  3. Trains SUPAEEG with InfoNCE + Gaussian regulariser + L1 sparsity.
  4. Evaluates every ``eval_every`` epochs (Top-1 / Top-5 zero-shot retrieval).
  5. Saves per-subject/per-fold checkpoints and logs a results table.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import typer
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so all src.* imports resolve.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dataset import ThingsEEGDataset
from src.encoders.visual_encoder import VisualEncoder, VisualFeatureLookup, validate_features
from src.models.supaeeg import SUPAEEG
from src.trainer.metrics import retrieve_all

app = typer.Typer()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """All runtime hyperparameters in one place."""

    protocol: str = "intra"
    subject: int = 1
    all_subjects: list[int] = field(default_factory=lambda: list(range(1, 11)))
    dataset_dir: str = "data/things_eeg"
    feature_path: str = "data/vision_encoder/clip/visual_features_clip.pt"
    device: str = "cuda"
    epochs: int = 100
    batch_size: int = 256
    eval_every: int = 5
    lambda_reg: float = 0.1
    beta_l1: float = 0.01
    tau: float = 0.07
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    checkpoint_dir: str = "outputs/supaeeg"


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
# Training and evaluation helpers
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: SUPAEEG,
    train_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
) -> float:
    """Run a single training epoch.

    Args:
        model:        SUPAEEG model (will be set to train mode).
        train_loader: DataLoader for the training split.
        optimizer:    AdamW optimiser.
        device:       Compute device.

    Returns:
        Mean total loss over all batches in the epoch.
    """
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        eeg: torch.Tensor = batch["eeg"].to(device)
        image_concepts: list[str] = batch["image_concepts"]
        image_files: list[str] = batch["image_files"]

        _z1, _z2, _z3, loss, components = model(
            eeg, image_concepts, image_files, return_loss=True
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(components["total"])

    return total_loss / max(len(train_loader), 1)


def evaluate(
    model: SUPAEEG,
    test_loader: DataLoader,
    feature_lookup: VisualFeatureLookup,
    device: torch.device,
) -> tuple[float, float]:
    """Zero-shot concept retrieval evaluation on the test set.

    Aggregates per-concept EEG embeddings (averaged over repetitions) and
    paired CLIP image embeddings, then computes Top-1 and Top-5 retrieval
    accuracy via the diagonal-retrieval protocol.

    Args:
        model:          SUPAEEG model.
        test_loader:    DataLoader over the test split.
        feature_lookup: Visual feature bank.
        device:         Compute device.

    Returns:
        Tuple ``(top1, top5)`` accuracy values in [0, 1].
    """
    model.eval()
    concept_embeddings: dict[str, list[torch.Tensor]] = defaultdict(list)
    concept_to_file: dict[str, str] = {}

    with torch.no_grad():
        for batch in test_loader:
            eeg = batch["eeg"].to(device)
            z = model.embed(eeg)  # (batch, 2304), ℓ2-normalised inside embed()
            for i, (concept, img_file) in enumerate(
                zip(batch["image_concepts"], batch["image_files"])
            ):
                concept_embeddings[concept].append(z[i].cpu())
                concept_to_file[concept] = img_file

    concept_order = sorted(concept_embeddings.keys())

    # Average per concept then ℓ2-normalise
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

    # Concat(S1, S2, S3) then ℓ2-normalise
    image_features = torch.cat(
        [
            F.normalize(
                torch.cat([*feature_lookup.retrieve(c, concept_to_file[c])]).unsqueeze(0),
                dim=1,
            )
            for c in concept_order
        ],
        dim=0,
    ).numpy()  # (N_concepts, 2304)

    top5_count, top1_count, total = retrieve_all(eeg_features, image_features)
    return top1_count / total, top5_count / total


def save_checkpoint(
    model: SUPAEEG,
    optimizer: AdamW,
    epoch: int,
    top1: float,
    top5: float,
    path: str,
) -> None:
    """Persist model and optimiser state to disk.

    Args:
        model:     SUPAEEG model.
        optimizer: AdamW optimiser.
        epoch:     Current training epoch.
        top1:      Top-1 accuracy at this checkpoint.
        top5:      Top-5 accuracy at this checkpoint.
        path:      File path for the checkpoint (``.pt``).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "top1": top1,
            "top5": top5,
        },
        path,
    )
    logger.info(f"Checkpoint saved | top1={top1:.4f} | path={path}")


def log_results_table(
    results: dict[int, dict[str, float]],
    avg_top1: float,
    avg_top5: float,
    protocol: str,
) -> None:
    """Log per-subject results in tabular format.

    Matches the Table 1 layout from the Shallow Alignment paper.

    Args:
        results:  Mapping of subject_id → {``'top1'``: float, ``'top5'``: float}.
        avg_top1: Average Top-1 accuracy across all subjects.
        avg_top5: Average Top-5 accuracy across all subjects.
        protocol: ``"intra"`` or ``"inter"``.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Protocol: {protocol.upper()}-SUBJECT")
    logger.info(f"{'Subject':<12} {'Top-1':>8} {'Top-5':>8}")
    logger.info(f"{'-' * 30}")
    for subject_id, r in sorted(results.items()):
        logger.info(
            f"Sub{subject_id:02d}{'':>8} "
            f"{r['top1'] * 100:>7.1f}% "
            f"{r['top5'] * 100:>7.1f}%"
        )
    logger.info(f"{'-' * 30}")
    logger.info(
        f"{'Avg':<12} "
        f"{avg_top1 * 100:>7.1f}% "
        f"{avg_top5 * 100:>7.1f}%"
    )
    logger.info(f"{'=' * 60}\n")


def _make_model(
    feature_lookup: VisualFeatureLookup,
    config: Config,
    device: torch.device,
) -> SUPAEEG:
    """Instantiate a fresh SUPAEEG model from ``config``.

    Args:
        feature_lookup: Pre-loaded visual feature bank.
        config:         Runtime configuration.
        device:         Compute device.

    Returns:
        Initialised SUPAEEG model placed on ``device``.
    """
    return SUPAEEG(
        feature_lookup=feature_lookup,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        device=device,
    ).to(device)


def _make_optimizer(model: SUPAEEG, config: Config) -> AdamW:
    """Build an AdamW optimiser from ``config``.

    Args:
        model:  Model whose parameters will be optimised.
        config: Runtime configuration.

    Returns:
        Configured AdamW optimiser.
    """
    return AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )


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

        model = _make_model(feature_lookup, config, device)
        optimizer = _make_optimizer(model, config)
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

        model = _make_model(feature_lookup, config, device)
        optimizer = _make_optimizer(model, config)
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
