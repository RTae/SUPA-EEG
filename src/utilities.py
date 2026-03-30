from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer

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


# -- Typer CLI option type aliases --
DatasetDir = Annotated[str, typer.Option("-d", "--dataset-dir", help="EEG-ImageNet dataset directory")]
Granularity = Annotated[str, typer.Option("-g", "--granularity", help="coarse | fine0-fine4 | all")]
Model = Annotated[str, typer.Option("-m", "--model", help="model name")]
BatchSize = Annotated[int, typer.Option("-b", "--batch-size", help="batch size")]
Subject = Annotated[int, typer.Option("-s", "--subject", help="subject id (0-15)")]
OutputDir = Annotated[str, typer.Option("-o", "--output-dir", help="directory to save results")]
PretrainedModel = Annotated[Optional[str], typer.Option("-p", "--pretrained-model", help="pretrained model filename")]


@dataclass
class Args:
    """Container mirroring the old argparse namespace so downstream code stays compatible."""
    dataset_dir: str
    granularity: str
    model: str
    batch_size: int
    subject: int
    output_dir: str
    pretrained_model: Optional[str]
