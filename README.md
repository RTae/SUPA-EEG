# SUPA-EEG: Scale Unified Parieto-occipital Architecture for Electroencephalography Decoding

Zero-shot visual decoding from EEG using the [THINGS-EEG2](https://osf.io/anp5v/)
dataset. SUPAEEG aligns EEG embeddings to frozen
[InternViT-6B](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5) image features
via a two-stage MMD + InfoNCE objective with a shared encoder for both modalities.

## Model Architecture

### Training Pipeline

```mermaid
flowchart TD
    subgraph OFFLINE["Offline — frozen InternViT-6B features (run once)"]
        IV["InternViT-6B-448px-V1-5\nlayers 20 · 24 · 28 · 32 · 36\n(n_concepts, n_imgs, 5, 3200)"]
    end

    subgraph INPUT["Input"]
        EEG["EEG signal\n(batch, 17, 100)"]
        IMG["Image features\n(batch, 5, 3200)"]
    end

    subgraph EEG_PATH["EEG branch"]
        EE["EEGNetEncoder\nLinear(1700→1024) + ResBlock + LN\n→ (batch, 1024)"]
        EP["eeg_projector\nLinear(1024→512)"]
    end

    subgraph IMG_PATH["Image branch"]
        RT["SubjectAwareRouter\nsoftmax weights over 5 layers\n(train: subject-aware)"]
        AVG["weighted sum over 5 layers\n→ (batch, 3200)"]
        IP1["img_pre_projector\nLinear(3200→1024)"]
        IP2["img_projector\nLinear(1024→512)"]
    end

    SE["share_encoder\nLinear(512→512)\nsame weights · both paths"]

    subgraph LOSS["Two-stage loss"]
        S1["Stage 1 (epochs 1–20)\nmmd_w · MMD_RBF + (1−mmd_w) · InfoNCE\nmmd_w: 0.9 → 0.2 (linear decay)"]
        S2["Default run\nflat LR (warmup_epochs=0)\noptional warmup + cosine if enabled"]
    end

    EEG --> EE --> EP --> SE
    IMG --> RT --> AVG --> IP1 --> IP2 --> SE
    SE -->|"zE, zI  (batch, 512) ℓ2-norm"| LOSS
    IV -.->|stop-grad| IMG
```

### EEGNetEncoder

```mermaid
flowchart LR
    IN["EEG input\n(batch, 17, 100)"]
    FLAT["reshape / flatten\n1700 = 17 x 100"]
    FC1["Linear\n1700 -> 1024"]
    RES["ResidualAdd"]
    GELU["GELU"]
    FC2["Linear\n1024 -> 1024"]
    DROP["Dropout\np = 0.3"]
    LN["LayerNorm\n1024"]
    OUT["encoder output\n(batch, 1024)"]

    IN --> FLAT --> FC1 --> RES --> LN --> OUT
    RES --> GELU --> FC2 --> DROP --> RES
```

### SubjectAwareRouter

```mermaid
flowchart TD
    SID["subject_ids\n(batch,) int64 or None"]
    GL["global_logits\nlearned prior over 5 layers"]
    SB["subject_bias\nEmbedding(10, 5)"]
    SM["subject dropout mask\np = 0.3"]
    ADD["add subject bias\nto global prior"]
    LM["layer dropout mask\np = 0.1"]
    LOGITS["router logits\n(batch, 5)"]
    TEMP["divide by temperature\nT = 1.0"]
    SOFT["softmax over layers"]
    W["layer weights\n(batch, 5)"]
    INF["inference path\nuse global prior only"]

    GL --> ADD
    SID --> SB --> SM --> ADD
    ADD --> LM --> LOGITS --> TEMP --> SOFT --> W
    GL --> INF --> TEMP
```

### Inference Pipeline

```mermaid
flowchart LR
    EEG["EEG signal\n(batch, 17, 100)"]

    subgraph MODEL["Learned encoder"]
        EE["EEGNetEncoder\n→ (batch, 1024)"]
        EP["eeg_projector\n→ (batch, 512)"]
        SE["share_encoder\n→ (batch, 512)  ℓ2-norm"]
    end

    subgraph GALLERY["Image gallery · 200 test concepts"]
        IF["InternViT features\n(200, 5, 3200)"]
        IP["router(global prior) → weighted sum\n→ img_pre_projector → img_projector\n→ share_encoder  ℓ2-norm\n→ (200, 512)"]
    end

    RET["Cosine similarity\n200-way ranking\nTop-1 / Top-5"]

    EEG --> EE --> EP --> SE
    SE -->|query| RET
    IF --> IP -->|gallery| RET
```

## Project Structure

```text
conf/
└── config.yaml                      # all hyperparameters and Hydra settings
scripts/
└── extract_internvit_features.py    # offline feature extraction + ensure guard
src/
├── dataset.py                       # ThingsEEGDataset
├── utilities.py                     # Config dataclass + training helpers
├── encoders/
│   ├── eegnet_encoder.py            # MLP encoder  (B,17,100) → (B,1024)
│   └── vision_encoder.py            # InternViTFeatureLookup
├── models/
│   └── supaeeg.py                   # SUPAEEG — shared-encoder alignment model
└── trainer/
    ├── loss.py                      # mmd_rbf, info_nce_loss, compute_loss
    └── metrics.py                   # retrieve_all, retrieve_topk
train.py                             # Hydra entry point
data/
└── things_eeg/
    ├── sub-01/ … sub-10/            # preprocessed_eeg_training.npy / _test.npy
    ├── training_images/             # <concept>/<image>.jpg  (1654 concepts × 10)
    ├── test_images/                 # <concept>/<image>.jpg  (200 concepts × 1)
    ├── image_metadata.npy
    └── image_feature/
        └── internvit_multilevel_20_24_28_32_36/
            └── internvit_features.npy   # dict {(concept, img_file): ndarray(n_layers, 3200)} float16
```

## Setup

### Install

```bash
# Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtualenv and install dependencies
uv venv && uv sync

# Install flash-attn and einops separately (require --no-build-isolation)
uv pip install einops flash-attn --no-build-isolation

# Activate (every session)
source .venv/bin/activate
```

### Data

Download EEG data and images:

```bash
sudo apt-get install aria2
bash scripts/download_data.sh
```

This fetches:
- Preprocessed EEG for subjects 1–10 with only 17 channels `data/things_eeg/sub-XX/`
- Image metadata, training images (1654 concepts × 10 images), test images (200 concepts × 1 image)
- InternViT features are also available for download in this script, but you can also extract them locally (see below).

for 63 channels, run
```bash
bash scripts/download_dataset_full.sh
```
which fetches the full dataset (including 63-channel EEG) but not the vision features (see below).

Manual sources:

| Item | URL |
|------|-----|
| EEG data (preprocessed) | [OSF — anp5v](https://osf.io/anp5v/files/osfstorage) |
| Image metadata | [OSF — y63gw/qkgtf](https://osf.io/y63gw/files/qkgtf) |
| Training images | [OSF — y63gw/3v527](https://osf.io/y63gw/files/3v527) |
| Test images | [OSF — y63gw/znu7b](https://osf.io/y63gw/files/znu7b) |

### InternViT features

InternViT-6B features are extracted locally before training. `train.py` does
this automatically on the first run (no-op if already extracted):

```bash
# Run extraction manually (optional — train.py calls this automatically)
python scripts/extract_internvit_features.py

# Override device or batch size
python scripts/extract_internvit_features.py device=cuda extract_batch_size=32
```
The extracted `.npy` files are written to `internvit_dir` (see config) and are
not re-extracted on subsequent runs.

## Configuration

All keys live in `conf/config.yaml` and can be overridden as Hydra `key=value` pairs.

### Data & device

| Key | Description | Default |
|-----|-------------|---------|
| `dataset_dir` | THINGS-EEG2 root | `data/things_eeg` |
| `device` | Compute device (`DEVICE` env var overrides) | `cuda` |
| `data_average` | Average repetitions in the training split | `true` |
| `data_average_test` | Average repetitions in the test split | `false` |
| `eeg_suffix` | EEG folder suffix: `""` = 17-ch, `"_63"` = 63-ch | `""` |

### EEG time window

| Key | Description | Default |
|-----|-------------|---------|
| `eeg_t_start` | Crop start time in seconds | `-0.2` |
| `eeg_t_end` | Crop end time in seconds | `0.8` |

### Protocol

| Key | Description | Default |
|-----|-------------|---------|
| `protocol` | `intra` (per-subject) or `inter` (LOSO) | `intra` |
| `subject` | Subject index 1–10; `-1` = all subjects (intra only) | `1` |

### InternViT config keys

| Key | Description | Default |
|-----|-------------|---------|
| `internvit_model` | HuggingFace model ID | `OpenGVLab/InternViT-6B-448px-V1-5` |
| `internvit_dir` | Output directory for extracted `.npy` files | `data/things_eeg/image_feature/internvit_multilevel_20_24_28_32_36` |
| `layer_ids` | Transformer layers to extract | `[20, 24, 28, 32, 36]` |
| `train_img_dir` | Training image directory | `data/things_eeg/training_images` |
| `test_img_dir` | Test image directory | `data/things_eeg/test_images` |
| `metadata_path` | Image metadata `.npy` | `data/things_eeg/image_metadata.npy` |

### Architecture

| Key | Description | Default |
|-----|-------------|---------|
| `n_channels` | Number of EEG channels | `17` |
| `n_timepoints` | EEG timepoints per sample | `100` |
| `feature_dim` | Shared embedding dimension | `512` |
| `eeg_feature_dim` | EEGNetEncoder output dimension | `1024` |
| `image_input_dim` | InternViT feature dimension per layer | `3200` |
| `image_mid_dim` | Image pre-projector hidden dimension | `1024` |
| `router_temperature` | Router softmax temperature | `1.0` |
| `subject_dropout_rate` | Subject-bias dropout in router | `0.3` |
| `layer_dropout_rate` | Layer-logit dropout in router | `0.1` |

### Training

| Key | Description | Default |
|-----|-------------|---------|
| `epochs` | Total training epochs | `200` |
| `batch_size` | Batch size | `1024` |
| `eval_every` | Evaluate every N epochs | `1` |
| `lr` | Initial learning rate | `1e-4` |
| `weight_decay` | AdamW weight decay | `1e-4` |
| `grad_clip` | Max gradient norm | `1.0` |
| `stage1_epochs` | Epochs in MMD+InfoNCE stage | `20` |
| `stage2_lr` | Minimum LR used by warmup/cosine schedule | `1e-5` |
| `mmd_start` | MMD weight at epoch 1 | `0.9` |
| `mmd_end` | MMD weight at end of stage 1 | `0.2` |
| `smooth_prob` | Gaussian smooth aug probability | `0.3` |
| `smooth_kernel_size` | Smooth kernel size (timepoints) | `5` |
| `smooth_sigma` | Smooth kernel sigma | `1.0` |
| `early_stop_patience` | Eval rounds before early stopping | `50` |
| `warmup_epochs` | LR warmup epochs (0 = flat lr throughout) | `0` |
| `eeg_t_start` | EEG epoch crop start (seconds) | `-0.2` |
| `eeg_t_end` | EEG epoch crop end (seconds) | `0.8` |
| `n_timepoints` | Timepoints after crop (`int((eeg_t_end - eeg_t_start) * 100)`) | `100` |
| `eeg_suffix` | Subject folder suffix: `""` = 17-ch (`sub-XX`), `"_63"` = 63-ch (`sub-XX_63`) | `""` |

## Implementation References

| Component | File | Role |
|-----------|------|------|
| Hydra entry point | `train.py` | Protocol dispatch, feature guard, training loop |
| Config dataclass | `src/utilities.py` | `Config`; `make_model`, `train_one_epoch`, `evaluate` |
| Hyperparameter reference | `conf/config.yaml` | YAML source of all defaults |
| EEG encoder | `src/encoders/eegnet_encoder.py` | Flatten → Linear(1700,1024) → ResBlock → LayerNorm |
| EEG augmentation | `src/encoders/eeg_augmentation.py` | `smooth_eeg` — Gaussian smoothing along time axis |
| Image feature lookup | `src/encoders/vision_encoder.py` | `InternViTFeatureLookup` — loads `.npy` per layer |
| Full model | `src/models/supaeeg.py` | `SUPAEEG` — shared-encoder alignment |
| Loss functions | `src/trainer/loss.py` | `mmd_rbf`, `info_nce_loss`, `compute_loss` |
| Retrieval eval | `src/trainer/metrics.py` | `retrieve_all` — Top-1 / Top-5 diagonal retrieval |
| Feature extraction | `scripts/extract_internvit_features.py` | Offline InternViT feature extraction + `ensure_internvit_features` guard |

## Recommended Configuration

Empirically validated best configs based on ablation experiments.

#### Inter-subject (LOSO):

```bash
# Single fold (quick test, ~2 min)
python train.py protocol=inter subject=1

# All 10 folds (~20 min)
nohup python train.py protocol=inter subject=-1 \
  > training_inter.log 2>&1 &
tail -f training_inter.log
```

Key findings for inter-subject:
- `data_average=true` — average 4 training repetitions per trial; significantly improves SNR
- `batch_size=1024` — doubles InfoNCE negatives (511→1023) for stronger gradient signal
- `eval_every=1` — evaluates every epoch so the exact peak checkpoint is always saved
- `epochs=15 stage1_epochs=20` — stage 2 (pure InfoNCE) consistently overfits; `stage1_epochs=epochs` disables it entirely
- `smooth_prob=0.3` — Gaussian smoothing augmentation along the time axis reduces high-frequency noise
- `early_stop_patience=2` — stops quickly if stage 2 degrades

#### Intra-subject:

```bash
# Single subject
python train.py subject=1 \
  eeg_t_start=0.0 eeg_t_end=0.7 n_timepoints=70

# All subjects (~30 min)
nohup python train.py subject=-1 \
  eeg_t_start=0.0 eeg_t_end=0.7 n_timepoints=70 \
  data_average=false \
  > training_intra.log 2>&1 &
tail -f training_intra.log
```

Key findings for intra-subject:
- `eeg_t_start=0.0 eeg_t_end=0.7 n_timepoints=70` — crop the 200ms pre-stimulus baseline; keeps only the visual response window (0–700ms)
- Use default `epochs` and config otherwise — intra converges well with the standard settings


## Testing

Test files live alongside the source they test (no separate `tests/` folder required).
pytest is configured in `pyproject.toml` to discover `test_*.py` files anywhere in the project.

```bash
# Run all tests
pytest

# Run a specific test file
pytest src/test_dataset.py
pytest src/trainer/test_metrics.py
pytest src/models/test_supaeeg_router.py
pytest test_train.py

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

## Dataset Explorer

Open `viz_thingeeg.ipynb` in Jupyter to inspect EEG samples, image concepts, and
feature distributions interactively.

## Experimental Analysis

Please refer to `reports/experiment.md` for detailed analysis of results.