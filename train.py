from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import ThingsEEGDataset
from src.encoders.visual_encoder import VisualEncoder, VisualFeatureLookup, validate_features
from src.utilities import (
    Config,
    EncoderConfig,
    evaluate,
    log_results_table,
    make_model,
    make_optimizer,
    make_scheduler,
    save_checkpoint,
    train_one_epoch,
)


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
    encoder_cfg: EncoderConfig,
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
        encoder_cfg:  Encoder configuration (type, model_name, layer_indices).
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
    logger.info(
        f"Encoder: type={encoder_cfg.type!r} model={encoder_cfg.model_name!r} "
        f"layer_indices={encoder_cfg.layer_indices}"
    )

    encoder = VisualEncoder(
        encoder_type=encoder_cfg.type,
        model_name=encoder_cfg.model_name,
        device="cpu",  # extraction runs on CPU to avoid GPU OOM during setup
        layer_indices=encoder_cfg.layer_indices,
    )

    from scripts.extract_visual_features import extract_features  # local import — only needed during extraction

    feature_dict: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

    image_base = Path(dataset_dir)
    splits = [image_base / "training_images", image_base / "test_images"]

    for split_dir in splits:
        if not split_dir.is_dir():
            logger.warning(f"Image directory not found, skipping: {split_dir}")
            continue
        split_features = extract_features(encoder=encoder, image_dir=split_dir)
        feature_dict.update(split_features)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "features": feature_dict,
            "encoder_type": encoder_cfg.type,
            "model_name": encoder_cfg.model_name,
            "layer_indices": encoder_cfg.layer_indices,
        },
        path,
    )
    logger.info(f"Visual features saved to {path}")

    lookup = VisualFeatureLookup(path)
    validate_features(lookup)
    return lookup


# ---------------------------------------------------------------------------
# Config conversion
# ---------------------------------------------------------------------------

def _cfg_to_config(cfg: DictConfig) -> Config:
    """Convert a Hydra DictConfig into the Config dataclass used by the runners."""
    encoder_cfg = EncoderConfig(
        type=cfg.encoder.type,
        model_name=cfg.encoder.model_name,
        layer_indices=OmegaConf.to_container(cfg.encoder.layer_indices, resolve=True),
    )
    return Config(
        protocol=cfg.protocol,
        subject=cfg.subject,
        all_subjects=list(OmegaConf.to_container(cfg.all_subjects, resolve=True)),
        dataset_dir=cfg.dataset_dir,
        feature_path=cfg.feature_path,
        device=cfg.device,
        encoder=encoder_cfg,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        eval_every=cfg.eval_every,
        lambda_reg=cfg.lambda_reg,
        tau=cfg.tau,
        n_channels=cfg.model.n_channels,
        n_timepoints=cfg.model.n_timepoints,
        feature_dim=cfg.model.feature_dim,
        d_visual=cfg.model.d_visual,
        dropout=cfg.model.dropout,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs,
        grad_clip=cfg.grad_clip,
    )


# ---------------------------------------------------------------------------
# Protocol runners
# ---------------------------------------------------------------------------


def run_intra_subject(
    config: Config,
    feature_lookup: VisualFeatureLookup,
    device: torch.device,
    output_dir: str,
) -> dict[int, dict[str, float]]:
    """Train and evaluate one model per subject (intra-subject protocol).

    If ``config.subject`` is -1, iterates over all subjects in
    ``config.all_subjects``; otherwise trains a single subject.

    Args:
        config:         Runtime configuration.
        feature_lookup: Pre-loaded CLIP feature bank (shared across folds).
        device:         Compute device.
        output_dir:     Hydra run directory for checkpoints and metrics CSV.

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
            drop_last=True,
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

        model = make_model(config, device)
        optimizer = make_optimizer(model, config)
        scheduler = make_scheduler(optimizer, config)
        best_top1 = 0.0
        best_top5 = 0.0

        metrics_path = os.path.join(output_dir, f"metrics_sub{subject_id:02d}.csv")
        metrics_file = open(metrics_path, "w", newline="")
        csv_writer = csv.DictWriter(
            metrics_file, fieldnames=["epoch", "train_loss", "top1", "top5"]
        )
        csv_writer.writeheader()
        metrics_file.flush()

        tb_dir = os.path.join(output_dir, f"tb_sub{subject_id:02d}")
        writer = SummaryWriter(log_dir=tb_dir)

        for epoch in range(1, config.epochs + 1):
            mean_loss = train_one_epoch(
                model, train_loader, optimizer, feature_lookup, device,
                lambda_reg=config.lambda_reg, tau=config.tau,
                grad_clip=config.grad_clip,
            )
            scheduler.step()
            lr_now = scheduler.get_last_lr()[0]
            logger.info(
                f"Sub{subject_id:02d} | epoch {epoch}/{config.epochs} | loss={mean_loss:.4f}"
            )
            writer.add_scalar("train/loss", mean_loss, epoch)
            writer.add_scalar("train/lr", lr_now, epoch)

            row: dict[str, Any] = {"epoch": epoch, "train_loss": round(mean_loss, 6), "top1": "", "top5": ""}

            if epoch % config.eval_every == 0:
                top1, top5 = evaluate(model, test_loader, feature_lookup, device)
                logger.info(
                    f"Sub{subject_id:02d} | eval epoch {epoch} | "
                    f"Top-1: {top1:.4f} | Top-5: {top5:.4f}"
                )
                row["top1"] = round(top1, 6)
                row["top5"] = round(top5, 6)
                writer.add_scalar("eval/top1", top1, epoch)
                writer.add_scalar("eval/top5", top5, epoch)
                if top1 > best_top1:
                    best_top1 = top1
                    best_top5 = top5
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        top1,
                        top5,
                        path=os.path.join(output_dir, f"supaeeg_intra_sub{subject_id:02d}.pt"),
                    )

            csv_writer.writerow(row)
            metrics_file.flush()

        metrics_file.close()
        writer.close()
        logger.info(f"Sub{subject_id:02d} | metrics saved to {metrics_path}")

        all_results[subject_id] = {"top1": best_top1, "top5": best_top5}

    avg_top1 = sum(r["top1"] for r in all_results.values()) / len(all_results)
    avg_top5 = sum(r["top5"] for r in all_results.values()) / len(all_results)
    log_results_table(all_results, avg_top1, avg_top5, protocol="intra")
    return all_results


def run_inter_subject(
    config: Config,
    feature_lookup: VisualFeatureLookup,
    device: torch.device,
    output_dir: str,
) -> dict[int, dict[str, float]]:
    """Leave-one-subject-out (LOSO) cross-subject training.

    For each test subject, trains on the remaining 9 subjects' data combined
    via ``ConcatDataset``, then evaluates on the left-out subject's test set.
    A fresh model and optimiser are created for every fold.

    Args:
        config:         Runtime configuration.
        feature_lookup: Pre-loaded CLIP feature bank (shared across folds).
        device:         Compute device.
        output_dir:     Hydra run directory for checkpoints and metrics CSV.

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
            drop_last=True,
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

        model = make_model(config, device)
        optimizer = make_optimizer(model, config)
        scheduler = make_scheduler(optimizer, config)
        best_top1 = 0.0
        best_top5 = 0.0

        metrics_path = os.path.join(output_dir, f"metrics_loso_sub{test_subject:02d}.csv")
        metrics_file = open(metrics_path, "w", newline="")
        csv_writer = csv.DictWriter(
            metrics_file, fieldnames=["epoch", "train_loss", "top1", "top5"]
        )
        csv_writer.writeheader()
        metrics_file.flush()

        tb_dir = os.path.join(output_dir, f"tb_loso_sub{test_subject:02d}")
        writer = SummaryWriter(log_dir=tb_dir)

        for epoch in range(1, config.epochs + 1):
            mean_loss = train_one_epoch(
                model, train_loader, optimizer, feature_lookup, device,
                lambda_reg=config.lambda_reg, tau=config.tau,
                grad_clip=config.grad_clip,
            )
            scheduler.step()
            lr_now = scheduler.get_last_lr()[0]
            logger.info(
                f"LOSO test=Sub{test_subject:02d} | epoch {epoch}/{config.epochs} | "
                f"loss={mean_loss:.4f}"
            )
            writer.add_scalar("train/loss", mean_loss, epoch)
            writer.add_scalar("train/lr", lr_now, epoch)

            row: dict[str, Any] = {"epoch": epoch, "train_loss": round(mean_loss, 6), "top1": "", "top5": ""}

            if epoch % config.eval_every == 0:
                top1, top5 = evaluate(model, test_loader, feature_lookup, device)
                logger.info(
                    f"LOSO test=Sub{test_subject:02d} | eval epoch {epoch} | "
                    f"Top-1: {top1:.4f} | Top-5: {top5:.4f}"
                )
                row["top1"] = round(top1, 6)
                row["top5"] = round(top5, 6)
                writer.add_scalar("eval/top1", top1, epoch)
                writer.add_scalar("eval/top5", top5, epoch)
                if top1 > best_top1:
                    best_top1 = top1
                    best_top5 = top5
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        top1,
                        top5,
                        path=os.path.join(output_dir, f"supaeeg_loso_sub{test_subject:02d}.pt"),
                    )

            csv_writer.writerow(row)
            metrics_file.flush()

        metrics_file.close()
        writer.close()
        logger.info(f"LOSO sub{test_subject:02d} | metrics saved to {metrics_path}")

        all_results[test_subject] = {"top1": best_top1, "top5": best_top5}

    avg_top1 = sum(r["top1"] for r in all_results.values()) / len(all_results)
    avg_top5 = sum(r["top5"] for r in all_results.values()) / len(all_results)
    log_results_table(all_results, avg_top1, avg_top5, protocol="inter")
    return all_results


# ---------------------------------------------------------------------------
# Typer entry point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """Train SUPAEEG on THINGS-EEG2 using the intra- or inter-subject protocol.

    All options are controlled via conf/config.yaml or CLI overrides, e.g.::

        python train.py subject=2 epochs=500
        python train.py protocol=inter lr=1e-4
    """
    config = _cfg_to_config(cfg)

    if config.protocol not in ("intra", "inter"):
        raise ValueError(f"protocol must be 'intra' or 'inter', got {config.protocol!r}")

    logger.info("\n" + OmegaConf.to_yaml(cfg))

    if config.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        _device = torch.device("cpu")
    elif config.device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        logger.warning("MPS requested but not available; falling back to CPU.")
        _device = torch.device("cpu")
    else:
        _device = torch.device(config.device)
    logger.info(f"Device: {_device}")

    feature_lookup = ensure_visual_features(config.feature_path, config.encoder, config.dataset_dir)

    output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output dir: {output_dir}")

    if config.protocol == "intra":
        run_intra_subject(config, feature_lookup, _device, output_dir)
    else:
        run_inter_subject(config, feature_lookup, _device, output_dir)


if __name__ == "__main__":
    train()
