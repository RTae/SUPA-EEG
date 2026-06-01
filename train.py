import csv
import gc
import os
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import ThingsEEGDataset
from src.encoders.vision_encoder import InternViTFeatureLookup
from src.utilities import (
    Config,
    evaluate,
    log_results_table,
    make_model,
    make_optimizer,
    save_checkpoint,
    train_one_epoch,
)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------
def collate_fn(batch: list[tuple[Any, ...]]) -> dict[str, Any]:
    """Custom collate for ThingsEEGDataset batches.

    ThingsEEGDataset.__getitem__ returns::

        (eeg_tensor, image_tensor, subject_index, repetition_index,
         data_index, image_concept, image_file)

    The collate also derives:
      concept_indices: data_index (0-based concept index)
      image_indices:   repetition_index % n_imgs_per_concept
                       (maps to InternViTFeatureLookup image axis)

    Args:
        batch: List of tuples from the dataset.

    Returns:
        Dict with keys 'eeg', 'image_concepts', 'image_files',
        'concept_indices', 'image_indices'.
    """
    eeg_tensors     = torch.stack([item[0] for item in batch], dim=0)
    image_concepts  = [item[5] for item in batch]
    image_files     = [item[6] for item in batch]
    concept_indices = [item[4] for item in batch]          # data_index
    image_indices   = [item[3] for item in batch]          # repetition_index
    return {
        "eeg":             eeg_tensors,
        "image_concepts":  image_concepts,
        "image_files":     image_files,
        "concept_indices": concept_indices,
        "image_indices":   image_indices,
    }


# ---------------------------------------------------------------------------
# Config conversion
# ---------------------------------------------------------------------------

def _cfg_to_config(cfg: DictConfig) -> Config:
    """Convert a Hydra DictConfig into the Config dataclass used by the runners."""
    return Config(
        protocol=cfg.protocol,
        subject=cfg.subject,
        all_subjects=list(OmegaConf.to_container(cfg.all_subjects, resolve=True)),
        dataset_dir=cfg.dataset_dir,
        device=cfg.device,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        eval_every=cfg.eval_every,
        n_channels=17,
        n_timepoints=100,
        feature_dim=cfg.feature_dim,
        eeg_feature_dim=cfg.eeg_feature_dim,
        image_input_dim=cfg.image_input_dim,
        image_mid_dim=cfg.image_mid_dim,
        dropout=0.3,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
        stage1_epochs=cfg.stage1_epochs,
        stage2_lr=cfg.stage2_lr,
        mmd_start=cfg.mmd_start,
        mmd_end=cfg.mmd_end,
        internvit_model=cfg.internvit_model,
        internvit_dir=cfg.internvit_dir,
        layer_ids=list(OmegaConf.to_container(cfg.layer_ids, resolve=True)),
        train_img_dir=cfg.train_img_dir,
        test_img_dir=cfg.test_img_dir,
        metadata_path=cfg.metadata_path,
    )


# ---------------------------------------------------------------------------
# Protocol runners
# ---------------------------------------------------------------------------


def run_intra_subject(
    config: Config,
    internvit_lookup: InternViTFeatureLookup,
    device: torch.device,
    output_dir: str,
) -> dict[int, dict[str, float]]:
    """Train and evaluate one model per subject (intra-subject protocol).

    If ``config.subject`` is -1, iterates over all subjects in
    ``config.all_subjects``; otherwise trains a single subject.

    Args:
        config:          Runtime configuration.
        internvit_lookup: InternViTFeatureLookup (concept+file keyed).
        device:          Compute device.
        output_dir:      Hydra run directory for checkpoints and metrics CSV.

    Returns:
        Mapping of subject_id → {'top1': float, 'top5': float}.
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
            load_images=False,
        )
        test_dataset = ThingsEEGDataset(
            dataset_dir=config.dataset_dir,
            data_type="test",
            subject=subject_id,
            load_images=False,
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

        model     = make_model(config, device)
        optimizer = make_optimizer(model, config)
        best_top1 = 0.0
        best_top5 = 0.0

        metrics_path = os.path.join(output_dir, f"metrics_sub{subject_id:02d}.csv")
        metrics_file = open(metrics_path, "w", newline="")
        csv_writer = csv.DictWriter(
            metrics_file,
            fieldnames=["epoch", "total", "infonce", "mmd", "mmd_weight", "top1", "top5"],
        )
        csv_writer.writeheader()
        metrics_file.flush()

        tb_dir = os.path.join(output_dir, f"tb_sub{subject_id:02d}")
        writer = SummaryWriter(log_dir=tb_dir)

        for epoch in range(1, config.epochs + 1):
            components = train_one_epoch(
                model, train_loader, optimizer, internvit_lookup, device, epoch, config,
            )
            logger.info(
                f"Sub{subject_id:02d} | epoch {epoch}/{config.epochs} | "
                f"total={components['total']:.4f} | infonce={components['infonce']:.4f} | "
                f"mmd={components['mmd']:.4f} | mmd_w={components['mmd_weight']:.3f}"
            )
            writer.add_scalar("train/total",      components["total"],      epoch)
            writer.add_scalar("train/infonce",    components["infonce"],    epoch)
            writer.add_scalar("train/mmd",        components["mmd"],        epoch)
            writer.add_scalar("train/mmd_weight", components["mmd_weight"], epoch)

            row: dict[str, Any] = {
                "epoch":      epoch,
                "total":      round(components["total"],      6),
                "infonce":    round(components["infonce"],    6),
                "mmd":        round(components["mmd"],        6),
                "mmd_weight": round(components["mmd_weight"], 6),
                "top1": "",
                "top5": "",
            }

            if epoch % config.eval_every == 0:
                top1, top5 = evaluate(model, test_loader, internvit_lookup, device)
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
                        model, optimizer, epoch, top1, top5,
                        path=os.path.join(output_dir, f"supaeeg_intra_sub{subject_id:02d}.pt"),
                    )

            csv_writer.writerow(row)
            metrics_file.flush()

        metrics_file.close()
        writer.close()
        logger.info(f"Sub{subject_id:02d} | metrics saved to {metrics_path}")

        all_results[subject_id] = {"top1": best_top1, "top5": best_top5}

        # Free GPU memory before the next subject
        del model, optimizer, train_loader, test_loader, train_dataset, test_dataset
        gc.collect()
        torch.cuda.empty_cache()

    avg_top1 = sum(r["top1"] for r in all_results.values()) / len(all_results)
    avg_top5 = sum(r["top5"] for r in all_results.values()) / len(all_results)
    log_results_table(all_results, avg_top1, avg_top5, protocol="intra")
    return all_results


def run_inter_subject(
    config: Config,
    internvit_lookup: InternViTFeatureLookup,
    device: torch.device,
    output_dir: str,
) -> dict[int, dict[str, float]]:
    """Leave-one-subject-out (LOSO) cross-subject training.

    For each test subject, trains on the remaining 9 subjects' data combined
    via ``ConcatDataset``, then evaluates on the left-out subject's test set.
    A fresh model and optimiser are created for every fold.

    Args:
        config:          Runtime configuration.
        internvit_lookup: InternViTFeatureLookup (concept+file keyed).
        device:          Compute device.
        output_dir:      Hydra run directory for checkpoints and metrics CSV.

    Returns:
        Mapping of test_subject_id → {'top1': float, 'top5': float}.
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
                    load_images=False,
                )
                for s in train_subjects
            ]
        )
        test_dataset = ThingsEEGDataset(
            dataset_dir=config.dataset_dir,
            data_type="test",
            subject=test_subject,
            load_images=False,
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

        model     = make_model(config, device)
        optimizer = make_optimizer(model, config)
        best_top1 = 0.0
        best_top5 = 0.0

        metrics_path = os.path.join(output_dir, f"metrics_loso_sub{test_subject:02d}.csv")
        metrics_file = open(metrics_path, "w", newline="")
        csv_writer = csv.DictWriter(
            metrics_file,
            fieldnames=["epoch", "total", "infonce", "mmd", "mmd_weight", "top1", "top5"],
        )
        csv_writer.writeheader()
        metrics_file.flush()

        tb_dir = os.path.join(output_dir, f"tb_loso_sub{test_subject:02d}")
        writer = SummaryWriter(log_dir=tb_dir)

        for epoch in range(1, config.epochs + 1):
            components = train_one_epoch(
                model, train_loader, optimizer, internvit_lookup, device, epoch, config,
            )
            logger.info(
                f"LOSO test=Sub{test_subject:02d} | epoch {epoch}/{config.epochs} | "
                f"total={components['total']:.4f} | infonce={components['infonce']:.4f} | "
                f"mmd={components['mmd']:.4f} | mmd_w={components['mmd_weight']:.3f}"
            )
            writer.add_scalar("train/total",      components["total"],      epoch)
            writer.add_scalar("train/infonce",    components["infonce"],    epoch)
            writer.add_scalar("train/mmd",        components["mmd"],        epoch)
            writer.add_scalar("train/mmd_weight", components["mmd_weight"], epoch)

            row: dict[str, Any] = {
                "epoch":      epoch,
                "total":      round(components["total"],      6),
                "infonce":    round(components["infonce"],    6),
                "mmd":        round(components["mmd"],        6),
                "mmd_weight": round(components["mmd_weight"], 6),
                "top1": "",
                "top5": "",
            }

            if epoch % config.eval_every == 0:
                top1, top5 = evaluate(model, test_loader, internvit_lookup, device)
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
                        model, optimizer, epoch, top1, top5,
                        path=os.path.join(output_dir, f"supaeeg_loso_sub{test_subject:02d}.pt"),
                    )

            csv_writer.writerow(row)
            metrics_file.flush()

        metrics_file.close()
        writer.close()
        logger.info(f"LOSO sub{test_subject:02d} | metrics saved to {metrics_path}")

        all_results[test_subject] = {"top1": best_top1, "top5": best_top5}

        # Free GPU memory before the next fold
        del model, optimizer, train_loader, test_loader, train_dataset, test_dataset
        gc.collect()
        torch.cuda.empty_cache()

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

        python train.py subject=2 epochs=60
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

    # Ensure InternViT features are present (no-op if already extracted)
    from src.encoders.vision_encoder import ensure_internvit_features
    ensure_internvit_features(
        internvit_dir  = config.internvit_dir,
        layer_ids      = config.layer_ids,
        train_img_dir  = config.train_img_dir,
        test_img_dir   = config.test_img_dir,
        model_name     = config.internvit_model,
        device         = config.device,
        batch_size     = 64,
    )

    internvit_lookup = InternViTFeatureLookup(
        feature_path=os.path.join(config.internvit_dir, "internvit_features.npy"),
    )

    output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output dir: {output_dir}")

    # Save a human-readable copy of the config alongside results
    config_dump_path = os.path.join(output_dir, "config_used.yaml")
    with open(config_dump_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Config saved to {config_dump_path}")

    if config.protocol == "intra":
        results = run_intra_subject(config, internvit_lookup, _device, output_dir)
    else:
        results = run_inter_subject(config, internvit_lookup, _device, output_dir)

    # Write summary.txt
    summary_path = os.path.join(output_dir, "summary.txt")
    avg_top1 = sum(r["top1"] for r in results.values()) / len(results)
    avg_top5 = sum(r["top5"] for r in results.values()) / len(results)
    with open(summary_path, "w") as f:
        f.write(f"protocol : {config.protocol}\n")
        f.write(f"epochs   : {config.epochs}\n")
        f.write(f"device   : {config.device}\n\n")
        f.write(f"{'subject':<12} {'top1':>8} {'top5':>8}\n")
        f.write("-" * 30 + "\n")
        for subj, r in sorted(results.items()):
            f.write(f"sub-{subj:02d}      {r['top1']:>8.4f} {r['top5']:>8.4f}\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'average':<12} {avg_top1:>8.4f} {avg_top5:>8.4f}\n")
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    train()
