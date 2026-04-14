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
    
    # CP : Train on all other people, test on the target person (using their first session).
    elif metric == "cp":
        raise NotImplementedError("CP split is not implemented yet.")

    # WT : follow the original benchmark protocol exactly.
    # After subject / granularity filtering, samples remain arranged in 50-sample
    # category blocks, with the first 30 used for training and the last 20 for test.
    elif metric == "wt":
        train_idx = [i for i in range(len(data_list)) if i % 50 < 30]
        test_idx = [i for i in range(len(data_list)) if i % 50 >= 30]

    else:
        raise ValueError(f"Unknown metric_type '{metric_type}'. Expected 'wt', 'ct', or 'cp'.")

    return train_idx, test_idx


@dataclass
class Args:
    """Container mirroring the old argparse namespace so downstream code stays compatible."""
    dataset_dir: str = "data/"
    granularity: str = "fine"
    model: str = "eegnet"
    batch_size: int = 40
    subject: int = -1
    output_dir: str = "output/"
    pretrained_model: Optional[str] = None
