import argparse
from pathlib import Path

import torch

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


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the shared CLI argument parser used by most scripts."""


    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", required=True, help="EEG-ImageNet dataset directory")
    parser.add_argument("-g", "--granularity", required=True, help="coarse | fine0-fine4 | all")
    parser.add_argument("-m", "--model", required=True, help="model name")
    parser.add_argument("-b", "--batch_size", default=40, type=int, help="batch size")
    parser.add_argument("-p", "--pretrained_model", help="pretrained model filename")
    parser.add_argument("-s", "--subject", default=0, type=int, help="subject id (0-15)")
    parser.add_argument("-o", "--output_dir", required=True, help="directory to save results")
    return parser
