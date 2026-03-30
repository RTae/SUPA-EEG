# Time-Series Project: Decoding Visual Perception from EEG

Decoding visual perception from EEG signals recorded while subjects viewed ImageNet images. This project supports two tasks:

- **Object Classification** : classify viewed object categories from EEG
- **Image Generation** : reconstruct viewed images from EEG via Stable Diffusion

## Baseline: CROSSPT-EEG

The current implementation follows the [CROSSPT-EEG](https://doi.org/10.48550/arXiv.2404.02717) pipeline as a baseline:

| Task | Pipeline | Models |
|------|----------|--------|
| Classification | EEG → DE features → classifier | EEGNet, MLP, RGNN, SVM, RF, KNN |
| Generation | EEG → DE features → MLP → CLIP embedding → Stable Diffusion | MLPMapper (cross-modal projection) |

> **Adding your own model:** Register it in `object_classification.py` (classification) or `image_generation.py` / `gen_eval.py` (generation). The shared dataset, feature extraction, and evaluation infrastructure are reusable.

## Project Structure

```
src/
├── utilities.py              # Shared constants, helpers, CLI parser, device detection
├── dataset.py                # EEGImageNetDataset (PyTorch Dataset)
├── object_classification.py  # Train & evaluate EEG classifiers
├── image_generation.py       # Train MLP mapper (EEG → CLIP embeddings)
├── gen_eval.py               # Generate images from EEG via Stable Diffusion
├── gen_img_list.py           # Export image filename / label reference lists
├── preprocessing/
│   ├── blip_clip.py          # BLIP captioning → CLIP text embeddings (one-time)
│   └── de_feat_cal.py        # Differential-entropy (DE) feature extraction
├── trainer/
│   ├── train.py              # Reusable training loops (classification & generation)
│   ├── inference.py          # Prediction-only loops
│   └── metrics.py            # Label mapping & evaluation helpers
└── model/
    ├── eegnet.py             # [Baseline] EEGNet
    ├── mlp.py                # [Baseline] MLP classifier
    ├── mlp_sd.py             # [Baseline] MLP mapper to CLIP embedding space
    ├── rgnn.py               # [Baseline] Regularized Graph Neural Network
    └── simple_model.py       # [Baseline] Sklearn (SVM, RF, KNN, DT, Ridge)
scripts/
└── merge_dataset.py          # Merge split .pth dataset files
data/
├── EEG-ImageNet.pth          # Merged EEG dataset
├── imageNet_images/          # Stimulus images by synset (generation task only)
└── mode/                     # EEG montage files
```

## Prerequisites

1. Install [uv](https://github.com/astral-sh/uv):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Download the EEG-ImageNet dataset from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/d812f7d1fc474b14bbd0/) and place the `.pth` files in `data/`.

3. *(Generation task only)* Download ImageNet images and place them under `data/imageNet_images/`.

## Installation

```bash
uv venv && uv sync
python scripts/merge_dataset.py data/EEG-ImageNet_1.pth data/EEG-ImageNet_2.pth data/EEG-ImageNet.pth
```

Activate the environment in each new terminal:
```bash
source .venv/bin/activate
```

## Usage

All scripts share the same CLI (see `utilities.build_arg_parser()`):

| Flag | Description |
|------|-------------|
| `-d` | Dataset directory (e.g. `data/`) |
| `-g` | Granularity: `coarse`, `fine0`–`fine4`, or `all` |
| `-m` | Model name (e.g. `eegnet`, `mlp`, `rgnn`, `svm`, `mlp_sd`) |
| `-b` | Batch size (default: 40) |
| `-s` | Subject ID, 0–15 (default: 0) |
| `-o` | Output directory |
| `-p` | Pretrained model filename (optional) |

### 1. Object Classification (Baseline)

```bash
# Deep model
python src/object_classification.py -d data/ -g coarse -m eegnet -s 0 -o output/

# Sklearn baseline
python src/object_classification.py -d data/ -g coarse -m svm -s 0 -o output/
```

### 2. Image Generation (Baseline)

```bash
# Step 1: Generate CLIP embeddings (one-time)
python src/preprocessing/blip_clip.py -d data/ -g all -m mlp_sd -o output/

# Step 2: Train EEG → CLIP mapper
python src/image_generation.py -d data/ -g coarse -m mlp_sd -s 0 -o output/

# Step 3: Generate images from EEG
python src/gen_eval.py -d data/ -g coarse -m mlp_sd -s 0 -o output/ -p mlpsd_s0_0.pth
```

### 3. Visualization

Open `viz.ipynb` in Jupyter to explore the EEG data interactively.