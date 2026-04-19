# Time-Series Project

Decode visual perception from EEG recorded while subjects view ImageNet images. The repository is currently documented around one primary workflow:

- **Object classification**: predict the viewed category from EEG.

## Overview

| Task | Entrypoint | Output |
|------|------------|--------|
| Object classification | `src/object_classification.py` | best checkpoint + `result.txt` |

All experiment runs are managed with [Hydra](https://hydra.cc/). By default, outputs are written to:

```text
outputs/<model>/<metric>/<timestamp>/
```

## Models

| Model | Config | Feature type | Objective | Notes |
|-------|--------|--------------|-----------|-------|
| EEGNet | `eegnet` | time | cross-entropy | Conv baseline on cropped EEG windows |
| MLP | `mlp` | freq | cross-entropy | DE features, fully connected classifier |
| RGNN | `rgnn` | freq | cross-entropy | Graph neural network over EEG channels |
| SVM / RF / KNN / DT / Ridge | `svm` … | freq | sklearn fit/predict | Classical baselines on DE features |
| Semantic | `semantic` | time | triplet loss | Retrieval-style embedding model |

> **Adding a new model:** add the implementation in `src/model/`, create `configs/model/<name>.yaml`, and register it in `src/object_classification.py`.

## Project Structure

```text
configs/
├── config.yaml               # Shared defaults: dataset, subject, metric, outputs
└── model/                    # Per-model hyperparameters
    ├── eegnet.yaml
    ├── mlp.yaml
    ├── mlp_sd.yaml
    ├── rgnn.yaml
    ├── semantic.yaml
    └── svm.yaml / rf.yaml / knn.yaml / dt.yaml / ridge.yaml
src/
├── dataset.py                # EEGImageNetDataset and BalancedBatchSampler
├── object_classification.py  # Classification training entrypoint
├── utilities.py              # Splits, device helpers, optimizer builder
├── preprocessing/
│   └── de_feat_cal.py        # Differential entropy (DE) feature extraction
├── trainer/
│   ├── loss.py               # Training losses
│   ├── train.py              # Training loops
│   ├── inference.py          # Inference-only helpers
│   └── metrics.py            # Evaluation helpers
└── model/
    ├── eegnet.py
    ├── mlp.py
    ├── rgnn.py
    ├── semantic.py
    └── simple_model.py
scripts/
└── merge_dataset.py          # Merge split dataset parts into EEG-ImageNet.pth
data/
├── EEG-ImageNet.pth          # Merged dataset used by the training scripts
├── de_feat/                  # Cached DE features
└── mode/                     # EEG montage files for RGNN
```

## Setup

### Prerequisites

1. Install [uv](https://github.com/astral-sh/uv):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Download the EEG-ImageNet dataset from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/d812f7d1fc474b14bbd0/) by using a command below (One time setup):
```bash
bash scripts/download_dataset.sh
```

### Installation

1. Install dependencies and create a virtual environment (One time setup):
```bash
uv venv && uv sync
```

2. Activate the virtual environment (every time you start a new terminal session):

```bash
source .venv/bin/activate
```

3. Merge the downloaded dataset parts into a single file (One time setup):

```bash
python scripts/merge_dataset.py data/EEG-ImageNet_1.pth data/EEG-ImageNet_2.pth data/EEG-ImageNet.pth
```

## Configuration

Global defaults live in `configs/config.yaml`. Model-specific defaults live in `configs/model/<name>.yaml`.

| Key | Description | Default |
|-----|-------------|---------|
| `dataset_dir` | Dataset directory | `data/` |
| `granularity` | `coarse`, `fine`, `fine0`-`fine4`, or `all` | `fine` |
| `model` | Hydra model group | `eegnet` |
| `batch_size` | Batch size | `40` |
| `subject` | Raw subject id used by the current run | `0` |
| `metric` | Benchmark split mode | `wt` |
| `output_dir` | Root output directory | `outputs/` |
| `pretrained_model` | Optional checkpoint filename | `null` |

Override any value from the CLI:

```bash
python src/object_classification.py model=mlp model.optimizer.lr=0.0005 model.epochs=300
```

## Benchmark Splits

The raw dataset contains 16 subject ids, corresponding to 8 real participants recorded in two sessions.

| Raw subject | RealID (`subject % 8`) | Stage |
|:-----------:|:----------------------:|:-----:|
| 0-7         | 0-7                    | 1     |
| 8-15        | 0-7                    | 2     |

Current implementation status:

| Paradigm | `metric` | Status | Behavior |
|----------|:--------:|--------|----------|
| Within-Time | `wt` | implemented | exact upstream 30/20 split inside each 50-sample block |
| Cross-Time | `ct` | not implemented | placeholder in code |
| Cross-Participant | `cp` | implemented | subjects 0-9 for training, remaining subjects for testing |

For `wt`, the dataset is first filtered by `subject` and `granularity`, then split with:

```text
train: i % 50 < 30
test:  i % 50 >= 30
```

## Training Pipelines

### How to train a model with DVC:

1. Queue an experiment with `dvc exp run --queue` and the desired overrides:
```bash
uv run dvc exp run --queue -S model=mlp -S subject=2,3,4 -S model.feature_type=freq,time train_mlp -S metric=wt
```

2. Review the experiment queue with `dvc exp show` and run all queued experiments with `dvc exp run --run-all`:
```bash
dvc exp show
```
```bash
dvc exp run --run-all
```

3. Review results with `dvc exp show` and `dvc exp diff`:
```bash
dvc exp show
```
```bash
dvc exp diff <exp_id_1> <exp_id_2>
```

4. Apply the best experiment to the workspace with `dvc exp apply`:
```bash
dvc exp apply <best_exp_id>
```

5. Optionally, push the experiment to the remote DVC storage with `dvc exp push`:
```bash
dvc exp push <best_exp_id>
```

6. Optionally, remove the experiment from the queue with `dvc exp remove`:
```bash
dvc exp remove <exp_id>

dvc exp remove --queue -- all
```

### Object Classification Pipeline

The classification entrypoint is `src/object_classification.py`. The high-level flow is:

```mermaid
flowchart LR
    A[Hydra config] --> B[Load EEGImageNetDataset]
    B --> C{model.feature_type}
    C -->|time| D[Use time-domain features]
    C -->|freq| E[Use frequency-domain features]
    D --> F[Apply benchmark split]
    E --> F
    F --> G[Build train/test subsets]
    G --> H{model type}
    H -->|simple| I[sklearn fit / predict]
    H -->|deep| J[CrossEntropy training]
    H -->|semantic| K[BalancedBatchSampler\n+ triplet loss]
    I --> L[result.txt]
    J --> L
    K --> L
```

Detailed steps:

1. `EEGImageNetDataset` loads `EEG-ImageNet.pth`, filters by `subject` and `granularity`, and returns dataset-local contiguous class ids.
2. If `model.feature_type=freq`, DE features are computed or loaded from cache. If `model.feature_type=time`, the cropped EEG window is used directly.
3. `utilities.get_benchmark_split` builds train/test indices for the selected metric.
4. `torch.utils.data.Subset` objects are created over a dataset that already emits contiguous labels.
5. Training dispatch depends on `model.type` and `model.name`.
6. The best checkpoint and summary metrics are written to the Hydra run directory.

Implementation references:

| Stage | Main file | What happens there |
|------|-----------|--------------------|
| Experiment entrypoint | `src/object_classification.py` | Hydra config loading, feature routing, split creation, train dispatch |
| Dataset loading | `src/dataset.py` | subject filtering, granularity filtering, contiguous label generation |
| Benchmark split | `src/utilities.py` | `wt` split logic and general helper utilities |
| Frequency features | `src/preprocessing/de_feat_cal.py` | DE feature extraction and cache reuse |
| Training loops | `src/trainer/train.py` | classifier training and semantic training loops |
| Evaluation | `src/trainer/metrics.py` | top-k metrics, embedding evaluation, prototype-based scoring |
| Losses | `src/trainer/loss.py` | triplet loss |

More concrete execution flow:

1. `src/object_classification.py` reads `configs/config.yaml` plus `configs/model/<name>.yaml` and creates the selected model.
2. `src/dataset.py` constructs `EEGImageNetDataset`, filters records, and exposes labels as contiguous indices local to the filtered dataset.
3. `src/object_classification.py` decides whether to call the time-domain path directly or to populate `dataset.frequency_feat` via `src/preprocessing/de_feat_cal.py`.
4. `src/utilities.py` creates the benchmark split indices, which are then wrapped as `Subset` objects in `src/object_classification.py`.
5. `src/object_classification.py` dispatches to either the sklearn path, the cross-entropy path, or the semantic path.
6. `src/trainer/train.py` runs optimization, while `src/trainer/metrics.py` evaluates top-1/top-5 or prototype-retrieval accuracy.
7. Semantic training additionally uses `BalancedBatchSampler` from `src/dataset.py` and `batch_hard_triplet_loss` from `src/trainer/loss.py`.

### Model-Specific Training Behavior

#### Classical Baselines

Models: `svm`, `rf`, `knn`, `dt`, `ridge`

1. Frequency-domain DE features are flattened.
2. The sklearn wrapper in `src/model/simple_model.py` is fit on the train split.
3. Evaluation is top-1 accuracy on the test split.

Reference files:

- `src/object_classification.py`
- `src/model/simple_model.py`
- `src/preprocessing/de_feat_cal.py`

#### Deep Classifiers

Models: `eegnet`, `mlp`, `rgnn`

1. A PyTorch `DataLoader` is built for train and test subsets.
2. The model is optimized with cross-entropy.
3. Each epoch evaluates on the test split with top-1, top-5, and average loss.
4. The checkpoint with the best top-1 score is saved.

Feature routing:

- `eegnet`: time-domain EEG window
- `mlp`: DE features
- `rgnn`: DE features

Reference files:

- `src/object_classification.py`
- `src/trainer/train.py`
- `src/trainer/metrics.py`
- `src/model/eegnet.py`
- `src/model/mlp.py`
- `src/model/rgnn.py`

##### DVC commands for running experiments with different overrides and tracking them with DVC:

1. Create a experiment queue with `dvc exp run --queue` and the desired overrides:
```bash
# Train MLP on all subjects with DE features and within-time split
uv run dvc exp run --queue -S model=mlp -S subject=-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 -S model.feature_type=freq,time train_mlp -S metric=wt

# Train MLP on all subjects with DE features and cross-participant split
uv run dvc exp run --queue -S model=mlp -S subject=-1 -S model.feature_type=freq,time train_mlp -S metric=cp
```

#### Semantic Training

`SemanticModel` does not train a softmax classifier. It learns an embedding with batch-hard triplet loss.

1. The dataset stays in time-domain mode.
2. `BalancedBatchSampler` builds batches containing multiple examples from multiple classes.
3. The model outputs L2-normalized embeddings.
4. Training uses batch-hard triplet loss.
5. Evaluation builds class prototypes from the eval split and measures retrieval-style top-1 and top-5 accuracy.

The triplet objective is implemented in `src/trainer/loss.py`, while evaluation logic lives in `src/trainer/metrics.py`.

Reference files:

- `src/object_classification.py`
- `src/dataset.py`
- `src/trainer/train.py`
- `src/trainer/loss.py`
- `src/trainer/metrics.py`
- `src/model/semantic.py`

Backbone options are selected with `model.backbone`.

**transformer**: patch-based encoder using a `[CLS]` token

```mermaid
flowchart LR
    A["EEG input\nB x C x T"] --> B["PatchEmbed\nConv1d"]
    B --> C["patch tokens\n+ [CLS] + pos_embed"]
    C --> D["TransformerEncoder\nx depth"]
    D --> E["[CLS] token"]
    E --> F["projection head"]
    F --> G["L2 embedding"]
```

**jepa**: transformer backbone plus EMA-updated target encoder

```mermaid
flowchart LR
    A["EEG input\nB x C x T"] --> B["PatchEmbed\nConv1d"]
    B --> C["online encoder"]
    C --> D["projection head"]
    D --> E["embedding"]
    B --> F["target encoder\nEMA copy"]
    C -->|EMA update| F
```

**nn**: lightweight Conv1d encoder

```mermaid
flowchart LR
    A["EEG input\nB x C x T"] --> B["Conv1d -> BN -> GELU"]
    B --> C["Conv1d -> BN -> GELU"]
    C --> D["AdaptiveAvgPool1d(1)"]
    D --> E["projection head"]
    E --> F["L2 embedding"]
```

## Usage

### Object Classification

#### Baseline Models

```bash
# Default run: EEGNet, subject 0, within-time split
python src/object_classification.py

# MLP baseline on DE features
python src/object_classification.py model=mlp

# RGNN baseline
python src/object_classification.py model=rgnn

# Classical ML baseline
python src/object_classification.py model=svm

# Override optimizer or epoch count
python src/object_classification.py model=mlp model.optimizer.lr=0.0005 model.epochs=300
```

#### Semantic Model

```bash
# Default semantic backbone: transformer
python src/object_classification.py model=semantic

# Switch backbone
python src/object_classification.py model=semantic model.backbone=jepa
python src/object_classification.py model=semantic model.backbone=nn

# Tune triplet settings
python src/object_classification.py model=semantic model.triplet_margin=0.25 model.samples_per_class=6
```

### Visualization

Open `viz.ipynb` in Jupyter to inspect samples, splits, and feature preparation interactively.