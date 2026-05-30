"""SUPAEEG training script (Hydra-configured).

Run from the project root::

    # Default config (conf/model/supaeeg.yaml)
    python src/train.py

    # Override any key on the CLI
    python src/train.py subject=1 model.epochs=50

    # Multi-run over several subjects
    python src/train.py --multirun subject=1,2,3,4,5

Hydra writes all outputs to::

    outputs/supaeeg/<timestamp>/

The script:
  1. Ensures the CLIP visual feature bank is available on disk.
  2. Builds ThingsEEGDataset train / test splits.
  3. Trains SUPAEEG with InfoNCE + Gaussian regulariser + L1 sparsity.
  4. Evaluates every ``model.eval_every`` epochs using the THINGS-EEG
     retrieval protocol (Top-1 / Top-5 zero-shot concept retrieval).
  5. Saves the best checkpoint to the Hydra run directory.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader

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


def ensure_visual_features(config: DictConfig) -> VisualFeatureLookup:
    """Load or extract the CLIP visual feature bank.

    Step 1 — If the feature file already exists, load and return it.
    Step 2 — Otherwise, run offline CLIP extraction over all training and
              test images, save the result, and return it.
    Step 3 — Validate that the loaded features contain at least one entry
              with correct S1/S2/S3 shapes.

    Args:
        config: Hydra DictConfig with at least ``model.feature_path`` and
                ``model.device`` keys.

    Returns:
        Populated VisualFeatureLookup.

    Raises:
        ValueError: If the loaded features fail shape validation.
    """
    path = config.model.feature_path

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
        device=config.model.device,
    )

    feature_dict: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

    image_base = Path(config.dataset_dir)  # top-level dataset_dir from conf/config.yaml
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


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """Main training entry point, driven by Hydra configuration.

    The active model config group (``conf/model/supaeeg.yaml`` by default) is
    merged under the ``model`` key by Hydra before this function is called.

    Args:
        cfg: Merged Hydra config (``conf/config.yaml`` + ``conf/model/*.yaml``).
    """
    logger.info("Config:\n" + OmegaConf.to_yaml(cfg))

    device = torch.device(
        cfg.model.device
        if hasattr(cfg.model, "device")
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Visual feature bank
    # ------------------------------------------------------------------
    feature_lookup = ensure_visual_features(cfg)

    # ------------------------------------------------------------------
    # 2 & 3. Datasets
    # ------------------------------------------------------------------
    train_dataset = ThingsEEGDataset(
        dataset_dir=cfg.dataset_dir,
        data_type="train",
        subject=cfg.subject,
    )
    test_dataset = ThingsEEGDataset(
        dataset_dir=cfg.dataset_dir,
        data_type="test",
        subject=cfg.subject,
    )

    # ------------------------------------------------------------------
    # 4 & 5. Data loaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # 6. Model
    # ------------------------------------------------------------------
    model = SUPAEEG(
        feature_lookup=feature_lookup,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        device=device,
    ).to(device)

    # ------------------------------------------------------------------
    # 7. Optimiser
    # ------------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.model.optimizer.lr,
        weight_decay=cfg.model.optimizer.weight_decay,
    )

    best_top1: float = 0.0

    # ------------------------------------------------------------------
    # Training epochs
    # ------------------------------------------------------------------
    for epoch in range(1, cfg.model.epochs + 1):
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
            f"Epoch {epoch}/{cfg.model.epochs} | "
            f"total={components['total']:.4f} | "
            f"infonce={components['infonce']:.4f} | "
            f"sigreg={components['sigreg']:.4f} | "
            f"l1={components['l1']:.4f}"
        )

        # ------------------------------------------------------------------
        # Periodic evaluation
        # ------------------------------------------------------------------
        if epoch % cfg.model.eval_every == 0:
            top1, top5 = _evaluate(model, test_loader, feature_lookup, device)
            logger.info(
                f"Eval  epoch {epoch} | Top-1: {top1:.4f} | Top-5: {top5:.4f}"
            )

            if top1 > best_top1:
                best_top1 = top1
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "top1": top1,
                        "top5": top5,
                        "config": OmegaConf.to_container(cfg, resolve=True),
                    },
                    "supaeeg_best.pt",  # written into Hydra's run dir
                )
                logger.info(
                    f"New best | Top-1: {top1:.4f} | Top-5: {top5:.4f}"
                )

    logger.info(f"Training complete. Best Top-1: {best_top1:.4f}")


if __name__ == "__main__":
    train()  # Hydra injects cfg automatically
