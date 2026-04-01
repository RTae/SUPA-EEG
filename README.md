# Time-Series Project: Decoding Visual Perception from EEG

Decoding visual perception from EEG signals recorded while subjects viewed ImageNet images. This project supports two tasks:

- **Object Classification** : classify viewed object categories from EEG
- **Image Generation** : reconstruct viewed images from EEG via Stable Diffusion

## Baseline: CROSSPT-EEG

The current implementation follows the [CROSSPT-EEG](https://doi.org/10.48550/arXiv.2406.07151) pipeline as a baseline:

| Task | Pipeline | Models |
|------|----------|--------|
| Classification | EEG → DE features → classifier | EEGNet, MLP, RGNN, SVM, RF, KNN |
| Generation | EEG → DE features → MLP → CLIP embedding → Stable Diffusion | MLPMapper (cross-modal projection) |

A self-supervised **EEG-JEPA** (Joint Embedding Predictive Architecture) model is also provided as a template for exploring representation learning on EEG.

> **Adding your own model:** Create a file in `src/model/`, add a Hydra config in `configs/model/`, and register it in `object_classification.py` (classification) or `image_generation.py` / `gen_eval.py` (generation). The shared dataset, feature extraction, and evaluation infrastructure are reusable.

## Project Structure

```
configs/
├── config.yaml               # Shared defaults (dataset, output, diffusion, blip)
└── model/                     # Per-model training hyperparameters
    ├── eegnet.yaml
    ├── mlp.yaml
    ├── rgnn.yaml
    ├── jepa.yaml
    ├── mlp_sd.yaml
    ├── svm.yaml / rf.yaml / knn.yaml / dt.yaml / ridge.yaml
src/
├── utilities.py              # Shared constants, helpers, device detection, benchmark splits
├── dataset.py                # EEGImageNetDataset (PyTorch Dataset)
├── object_classification.py  # Train & evaluate EEG classifiers
├── image_generation.py       # Train MLP mapper (EEG → CLIP embeddings)
├── gen_eval.py               # Generate images from EEG via Stable Diffusion
├── gen_img_list.py           # Export image filename / label reference lists
├── jepa_poc_train.py         # Synthetic CrossPT-style JEPA pretrain + linear-probe PoC
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
    ├── simple_model.py       # [Baseline] Sklearn (SVM, RF, KNN, DT, Ridge)
    └── jepa.py               # EEG-JEPA (self-supervised + classifier)
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

All scripts use [Hydra](https://hydra.cc/) for configuration. Defaults live in `configs/config.yaml` and per-model settings in `configs/model/`. Override any value from the CLI:

| Key | Description | Default |
|-----|-------------|---------|
| `dataset_dir` | Dataset directory | `data/` |
| `granularity` | `coarse`, `fine0`–`fine4`, or `all` | `coarse` |
| `model` | Model config group (`eegnet`, `mlp`, `rgnn`, `svm`, `mlp_sd`, …) | `eegnet` |
| `batch_size` | Batch size | `40` |
| `subject` | Target subject (RealID), 0–7 | `0` |
| `metric` | Evaluation paradigm: `wt`, `ct`, or `cp` | `wt` |
| `output_dir` | Output directory | `outputs/` |
| `pretrained_model` | Pretrained model filename | `null` |

Training hyperparameters (lr, epochs, optimizer, …) are set per-model in `configs/model/<name>.yaml` and can also be overridden:

```bash
python src/object_classification.py model.optimizer.lr=0.005 model.epochs=500
```

### Evaluation Paradigms

The dataset contains data from 16 raw subject IDs (0–15), which correspond to 8 real participants each recorded in two stages separated by ~7 days:

| Raw subject | RealID (`subject % 8`) | Stage |
|:-----------:|:----------------------:|:-----:|
| 0–7         | 0–7                    | 1     |
| 8–15        | 0–7                    | 2     |

Three evaluation paradigms are supported via `metric=`:

| Paradigm | `metric` | Train set | Test set |
|----------|:--------:|-----------|----------|
| **Within-Time** | `wt` | Target subject, Stage 2, first 30 labels | Target subject, Stage 2, remaining 20 labels |
| **Cross-Time** | `ct` | Target subject, Stage 1 | Target subject, Stage 2 |
| **Cross-Participant** | `cp` | All *other* subjects, Stage 1 | Target subject, Stage 1 |


### EEG-JEPA (Self-Supervised)

JEPA first pre-trains a Transformer encoder via masked-patch prediction in latent space, then fine-tunes a linear classifier on the learned `[CLS]` representation.

Synthetic CrossPT-style PoC is integrated in the same codebase:

```bash
# End-to-end synthetic run (pretrain + linear probe + top1/top5 eval)
python src/jepa_poc_train.py

# Quick smoke test
python src/jepa_poc_train.py --pretrain-epochs 1 --finetune-epochs 1 --samples-per-subject 80 --batch-size 32

# Task variants
python src/jepa_poc_train.py --task coarse
python src/jepa_poc_train.py --task fine --fine-group 3
```


### Object Classification

#### Training Loop Diagram

```mermaid
flowchart TD
    A[Load dataset and split by metric wt or ct or cp] --> B[Build train and test loaders]
    B --> D[JEPA with no pretrained_model]

    D --> D1[For each pretrain epoch]
    D1 --> D2[Forward pretrain path: pred and target repr]
    D2 --> D3[Compute JEPA loss]
    D3 --> D4[Backprop and optimizer step]
    D4 --> D5[EMA update of target encoder]
    D5 --> D1
    D1 -->|done| D6[Save JEPA pretrained checkpoint]

    D6 --> I

    I --> I1[For each supervised epoch]
    I1 --> I2[Forward classifier logits]
    I2 --> I3[CrossEntropy loss and optimizer step]
    I3 --> I4[Evaluate on test loader]
    I4 --> I5{Best accuracy}
    I5 -->|yes| I6[Save best checkpoint]
    I5 -->|no| I1
    I6 --> I1
    I1 -->|done| K[Write result.txt]
```

```bash
# Pre-train + fine-tune (pre-training runs automatically before classification)
python src/object_classification.py model=jepa subject=0 metric=ct

# Skip pre-training with a saved checkpoint
python src/object_classification.py model=jepa pretrained_model=jepa_pretrained_s0.pth

# Tune JEPA hyperparameters
python src/object_classification.py model=jepa model.pretrain_epochs=200 model.mask_ratio=0.6 model.embed_dim=256
```

#### Baseline for training and evaluating the baseline classification models:

```bash
# Deep model (defaults to eegnet, within-time)
python src/object_classification.py

# Specify model, subject, and evaluation paradigm
python src/object_classification.py model=rgnn subject=3 metric=ct

# Change training hyperparameters
python src/object_classification.py model.optimizer.lr=0.005 model.epochs=100

# Sklearn baseline
python src/object_classification.py model=svm
```


### Image Generation (Baseline)

```bash
# Step 1: Generate CLIP embeddings (one-time)
python src/preprocessing/blip_clip.py granularity=all

# Step 2: Train EEG → CLIP mapper
python src/image_generation.py model=mlp_sd

# Step 3: Generate images from EEG
python src/gen_eval.py model=mlp_sd pretrained_model=mlpsd_s0_0.pth
```

### Visualization

Open `viz.ipynb` in Jupyter to explore the EEG data interactively.