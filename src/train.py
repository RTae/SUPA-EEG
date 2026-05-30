"""SUPAEEG training script.

Run from the project root::

    # Default settings
    python src/train.py

    # Override parameters
    python src/train.py --subject 1 --epochs 50

    # CPU training
    python src/train.py --device cpu

The script:
  1. Ensures the CLIP visual feature bank is available on disk.
  2. Builds ThingsEEGDataset train / test splits.
  3. Trains SUPAEEG with InfoNCE + Gaussian regulariser + L1 sparsity.
  4. Evaluates every ``eval_every`` epochs using the THINGS-EEG
     retrieval protocol (Top-1 / Top-5 zero-shot concept retrieval).
  5. Saves the best checkpoint to ``output_dir``.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import typer
from loguru import logger
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

app = typer.Typer()


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


@app.command()
def train(
    dataset_dir: str = typer.Option("data/things_eeg", help="THINGS-EEG2 root directory"),
    subject: int = typer.Option(0, help="Subject index (0 = all subjects)"),
    feature_path: str = typer.Option(
        "data/vision_encoder/clip/visual_features_clip.pt",
        help="Path to the CLIP visual feature bank",
    ),
    device: str = typer.Option(
        os.environ.get("DEVICE", "cuda"), help="Compute device (overridden by DEVICE env var)"
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
    output_dir: str = typer.Option("outputs/supaeeg", help="Directory for saved checkpoints"),
) -> None:
    """Train SUPAEEG on the THINGS-EEG2 dataset."""
    logger.info(
        f"Config: dataset_dir={dataset_dir!r} subject={subject} device={device!r} "
        f"epochs={epochs} batch_size={batch_size} lr={lr} output_dir={output_dir!r}"
    )

    _device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    logger.info(f"Device: {_device}")

    # ------------------------------------------------------------------
    # 1. Visual feature bank
    # ------------------------------------------------------------------
    feature_lookup = ensure_visual_features(feature_path, device, dataset_dir)

    # ------------------------------------------------------------------
    # 2 & 3. Datasets
    # ------------------------------------------------------------------
    train_dataset = ThingsEEGDataset(
        dataset_dir=dataset_dir,
        data_type="train",
        subject=subject,
    )
    test_dataset = ThingsEEGDataset(
        dataset_dir=dataset_dir,
        data_type="test",
        subject=subject,
    )

    # ------------------------------------------------------------------
    # 4 & 5. Data loaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # 6. Model
    # ------------------------------------------------------------------
    model = SUPAEEG(
        feature_lookup=feature_lookup,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        device=_device,
    ).to(_device)

    # ------------------------------------------------------------------
    # 7. Optimiser
    # ------------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    best_top1: float = 0.0

    # ------------------------------------------------------------------
    # Training epochs
    # ------------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()

        for batch in train_loader:
            eeg = batch["eeg"].to(_device)
            image_concepts: list[str] = batch["image_concepts"]
            image_files: list[str] = batch["image_files"]

            z1, z2, z3, loss, components = model(
                eeg, image_concepts, image_files, return_loss=True
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"total={components['total']:.4f} | "
            f"infonce={components['infonce']:.4f} | "
            f"sigreg={components['sigreg']:.4f} | "
            f"l1={components['l1']:.4f}"
        )

        # ------------------------------------------------------------------
        # Periodic evaluation
        # ------------------------------------------------------------------
        if epoch % eval_every == 0:
            top1, top5 = _evaluate(model, test_loader, feature_lookup, _device)
            logger.info(
                f"Eval  epoch {epoch} | Top-1: {top1:.4f} | Top-5: {top5:.4f}"
            )

            if top1 > best_top1:
                best_top1 = top1
                checkpoint_path = out_path / "supaeeg_best.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "top1": top1,
                        "top5": top5,
                        "config": {
                            "dataset_dir": dataset_dir,
                            "subject": subject,
                            "feature_path": feature_path,
                            "device": device,
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "eval_every": eval_every,
                            "lambda_reg": lambda_reg,
                            "beta_l1": beta_l1,
                            "tau": tau,
                            "d_model": d_model,
                            "nhead": nhead,
                            "num_layers": num_layers,
                            "dim_feedforward": dim_feedforward,
                            "dropout": dropout,
                            "lr": lr,
                            "weight_decay": weight_decay,
                        },
                    },
                    checkpoint_path,
                )
                logger.info(
                    f"New best | Top-1: {top1:.4f} | Top-5: {top5:.4f} → {checkpoint_path}"
                )

    logger.info(f"Training complete. Best Top-1: {best_top1:.4f}")


if __name__ == "__main__":
    app()
