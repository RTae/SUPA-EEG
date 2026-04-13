from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim

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
    """Return the best available device (CUDA > MPS > CPU)."""
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
    target_subject: int,
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
        train_idx = [i for i, s in enumerate(data_list)
                     if s["subject"] % 8 == target_subject and s["subject"] < 8]
        test_idx = [i for i, s in enumerate(data_list)
                    if s["subject"] % 8 == target_subject and s["subject"] >= 8]

    # CP : Train on all other people, test on the target person (using their first session).
    elif metric == "cp":
        train_idx = [i for i, s in enumerate(data_list)
                     if s["subject"] % 8 != target_subject and s["subject"] < 8]
        test_idx = [i for i, s in enumerate(data_list)
                    if s["subject"] % 8 == target_subject and s["subject"] < 8]

    # WT : Train on part of the target person's data, test on the rest (random 30/20 split of their second session).
    elif metric == "wt":
        # Collect labels for this person's Stage-2 data, then split 30 / 20.
        stage2 = [(i, s) for i, s in enumerate(data_list)
                  if s["subject"] % 8 == target_subject and s["subject"] >= 8]
        unique_labels = sorted(set(s["label"] for _, s in stage2))
        seen_labels = set(unique_labels[:30])
        train_idx = [i for i, s in stage2 if s["label"] in seen_labels]
        test_idx  = [i for i, s in stage2 if s["label"] not in seen_labels]

    else:
        raise ValueError(f"Unknown metric_type '{metric_type}'. Expected 'wt', 'ct', or 'cp'.")

    return train_idx, test_idx


@dataclass
class Args:
    """Container mirroring the old argparse namespace so downstream code stays compatible."""
    dataset_dir: str = "data/"
    granularity: str = "coarse"
    model: str = "eegnet"
    batch_size: int = 40
    subject: int = 0
    output_dir: str = "output/"
    pretrained_model: Optional[str] = None
