from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader

# -- EEG channel groups --
PRE_FRONTAL = ["FP1", "FPZ", "FP2", "AF3", "AF4"]
FRONTAL = ["F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8"]
CENTRAL = ["CZ", "FCZ", "C1", "C2", "C3", "C4", "FC1", "FC2", "FC3", "FC4"]
L_TEMPORAL = ["FT7", "FC5", "T7", "C5", "TP7", "CP5", "P7", "P5"]
R_TEMPORAL = ["FT8", "FC6", "T8", "C6", "TP8", "CP6", "P8", "P6"]
PARIETAL = ["CPZ", "CP1", "CP3", "CP2", "CP4", "PZ", "P1", "P3", "P2", "P4"]
OCCIPITAL = ["POZ", "PO3", "PO5", "PO7", "PO4", "PO6", "PO8", "O1", "O2", "OZ", "CB1", "CB2"]

FREQ_BANDS: dict[str, list[float]] = {
    "delta": [0.5, 4],
    "theta": [4, 8],
    "alpha": [8, 13],
    "beta": [13, 30],
    "gamma": [30, 80],
}

# Resolve project root relative to this file so paths work from any cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"
DEFAULT_IMG_DIR = DEFAULT_DATA_DIR / "imageNet_images"


def _synset_map_path(language: str, img_dir: Path | str | None = None) -> Path:
    if language not in ("ch", "en"):
        raise ValueError(f"Invalid language '{language}'. Expected 'ch' or 'en'.")
    base = Path(img_dir) if img_dir else DEFAULT_IMG_DIR
    return base / f"synset_map_{language}.txt"


def wnid2category(wnid: str, language: str, img_dir: str | None = None) -> str:
    path = _synset_map_path(language, img_dir)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if wnid in line:
                return line.split()[1]
    raise ValueError(f"Could not find wnid: {wnid}")


def category2wnid(category: str, language: str, img_dir: str | None = None) -> str:
    path = _synset_map_path(language, img_dir)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if category in line:
                return line.split()[0]
    raise ValueError(f"Could not find category: {category}")


def get_device() -> torch.device:
    """Return the best available device (NPU > CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(params, opt_cfg):
    """Build a PyTorch optimizer from a Hydra model.optimizer config."""
    if opt_cfg.type == "sgd":
        return optim.SGD(
            params,
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.get("weight_decay", 0),
            momentum=opt_cfg.get("momentum", 0),
        )
    if opt_cfg.type == "adam":
        return optim.Adam(params, lr=opt_cfg.lr)
    raise ValueError(f"Unknown optimizer type: {opt_cfg.type}")


def get_benchmark_split(
    data_list: list[dict],
    metric_type: str,
) -> tuple[list[int], list[int]]:
    """Return (train_indices, test_indices) for the given evaluation paradigm.

    Stage is inferred from the raw ``subject`` field:
      - Stage 1: subject < 8   (first recording session)
      - Stage 2: subject >= 8  (second session, ~7 days later)
    RealID = subject % 8  (maps both sessions to the same person).

    Supported *metric_type* values:
      ``"wt"``  – Within-Time
      ``"ct"``  – Cross-Time
      ``"cp"``  – Cross-Participant
    """
    metric = metric_type.lower()

    # CT : Train on one session, test on the other for the same person.
    if metric == "ct":
        raise NotImplementedError("CT split is not implemented yet.")
    
    # CP : Use subjects 0-9 for training and all remaining subjects for testing.
    elif metric == "cp":
        train_idx = [
            i for i, sample in enumerate(data_list)
            if 0 <= int(sample.get("subject", -1)) <= 9
        ]
        test_idx = [
            i for i, sample in enumerate(data_list)
            if int(sample.get("subject", -1)) > 9
        ]

        if not train_idx:
            raise ValueError("CP split produced an empty training set. Expected subjects 0-9 in the data.")
        if not test_idx:
            raise ValueError("CP split produced an empty test set. Expected subjects above 9 in the data.")

    # WT : follow the original benchmark protocol exactly.
    # After subject / granularity filtering, samples remain arranged in 50-sample
    # category blocks, with the first 30 used for training and the last 20 for test.
    elif metric == "wt":
        train_idx = [i for i in range(len(data_list)) if i % 50 < 30]
        test_idx = [i for i in range(len(data_list)) if i % 50 >= 30]

    else:
        raise ValueError(f"Unknown metric_type '{metric_type}'. Expected 'wt', 'ct', or 'cp'.")

    return train_idx, test_idx


# ---------------------------------------------------------------------------
# SUPAEEG training config & helpers
# ---------------------------------------------------------------------------


@dataclass
class EncoderConfig:
    """Visual encoder settings, mirroring conf/model/supaeeg.yaml encoder block."""

    type: str = "clip"
    model_name: str = "openai/clip-vit-base-patch32"
    layer_indices: dict[str, int] = field(
        default_factory=lambda: {"S1": 3, "S2": 7, "S3": 11}
    )


@dataclass
class Config:
    """All runtime hyperparameters for SUPAEEG training."""

    protocol: str = "intra"
    subject: int = 1
    all_subjects: list[int] = field(default_factory=lambda: list(range(1, 11)))
    dataset_dir: str = "data/things_eeg"
    feature_path: str = "data/vision_encoder/clip/visual_features_clip.pt"
    device: str = "cuda"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
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


def train_one_epoch(
    model: Any,
    train_loader: DataLoader,
    optimizer: AdamW,
    feature_lookup: Any,
    device: torch.device,
    lambda_reg: float = 0.1,
    beta_l1: float = 0.01,
    tau: float = 0.07,
) -> float:
    """Run a single training epoch.

    Args:
        model:          SUPAEEG model (will be set to train mode).
        train_loader:   DataLoader for the training split.
        optimizer:      AdamW optimiser.
        feature_lookup: VisualFeatureLookup used to retrieve CLIP targets.
        device:         Compute device.
        lambda_reg:     Gaussian regulariser weight.
        beta_l1:        Channel-attention L1 sparsity weight.
        tau:            InfoNCE temperature.

    Returns:
        Mean total loss over all batches in the epoch.
    """
    from src.trainer.loss import compute_loss  # local import avoids circular deps

    model.train()
    total_loss = 0.0
    for batch in train_loader:
        eeg: torch.Tensor = batch["eeg"].to(device)
        image_concepts: list[str] = batch["image_concepts"]
        image_files: list[str] = batch["image_files"]

        z1, z2, z3 = model(eeg)

        S1, S2, S3 = feature_lookup.retrieve_batch(image_concepts, image_files)
        S1, S2, S3 = S1.to(device), S2.to(device), S3.to(device)

        loss, components = compute_loss(
            z1, z2, z3, S1, S2, S3,
            model.scale_encoder,
            lambda_reg=lambda_reg,
            beta_l1=beta_l1,
            tau=tau,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(components["total"])

    return total_loss / max(len(train_loader), 1)


def evaluate(
    model: Any,
    test_loader: DataLoader,
    feature_lookup: Any,
    device: torch.device,
) -> tuple[float, float]:
    """Zero-shot concept retrieval evaluation on the test set.

    Aggregates per-concept EEG embeddings (averaged over repetitions) and
    paired CLIP image embeddings, then computes Top-1 and Top-5 retrieval
    accuracy via the diagonal-retrieval protocol.

    Args:
        model:          SUPAEEG model.
        test_loader:    DataLoader over the test split.
        feature_lookup: VisualFeatureLookup instance.
        device:         Compute device.

    Returns:
        Tuple ``(top1, top5)`` accuracy values in [0, 1].
    """
    from src.trainer.metrics import retrieve_all  # local import avoids circular deps

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
    model: Any,
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


def make_model(
    config: Config,
    device: torch.device,
) -> Any:
    """Instantiate a fresh SUPAEEG model from ``config``.

    Args:
        config: Runtime configuration.
        device: Compute device.

    Returns:
        Initialised SUPAEEG model placed on ``device``.
    """
    from src.models.supaeeg import SUPAEEG  # local import avoids circular deps

    return SUPAEEG(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    ).to(device)


def make_optimizer(model: Any, config: Config) -> AdamW:
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
