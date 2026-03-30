# Time Series Project: EEG-ImageNet Dataset

This project trains models to decode visual perception from EEG signals recorded while subjects viewed ImageNet images which supports two main tasks:

- **Object classification** — classify viewed object categories from EEG (EEGNet, MLP, RGNN, SVM, RF, KNN, etc.)
- **Image generation** — reconstruct viewed images from EEG via Stable Diffusion (BLIP captioning → CLIP embedding → MLP mapper → diffusion)

## Project Structure

```
src/
├── utilities.py              # Shared constants, helpers, CLI parser, device detection
├── dataset.py                # EEGImageNetDataset (PyTorch Dataset)
├── de_feat_cal.py            # Differential-entropy feature extraction
├── blip_clip.py              # Generate CLIP text embeddings from images via BLIP captions
├── image_generation.py       # Train MLP mapper (EEG → CLIP embeddings)
├── gen_eval.py               # Generate images from EEG via Stable Diffusion
├── gen_img_list.py           # Export image filename / label reference lists
├── object_classification.py  # Train & evaluate EEG classifiers
└── model/
    ├── eegnet.py             # EEGNet architecture
    ├── mlp.py                # MLP classifier
    ├── mlp_sd.py             # MLP mapper to CLIP embedding space
    ├── rgnn.py               # Regularized Graph Neural Network
    └── simple_model.py       # Sklearn baselines (SVM, RF, KNN, DT, Ridge)
scripts/
└── merge_dataset.py          # Merge split .pth dataset files
data/
├── EEG-ImageNet.pth          # Merged dataset
├── imageNet_images/          # Stimulus images (by synset)
└── mode/                     # EEG montage files
```


## Prerequisites

1. Install [uv](https://github.com/astral-sh/uv):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Download the EEG-ImageNet dataset from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/d812f7d1fc474b14bbd0/) and place the `.pth` files in the `data/` directory.

3. Download the ImageNet stimulus images and place them under `data/imageNet_images/`.

## Installation
1. Create the virtual environment and and preprocess the dataset (one-time setup):
```bash
uv venv && uv sync

uv python scripts/merge_dataset.py data/EEG-ImageNet_1.pth data/EEG-ImageNet_2.pth data/EEG-ImageNet.pth
```

## Development
Activate the virtual environment:
```bash
source .venv/bin/activate
```

## Usage

All scripts share the same CLI interface (see `utilities.build_arg_parser()`):

| Flag | Description |
|------|-------------|
| `-d` | Dataset directory (e.g. `data/`) |
| `-g` | Granularity: `coarse`, `fine0`–`fine4`, or `all` |
| `-m` | Model name (e.g. `eegnet`, `mlp`, `rgnn`, `svm`, `mlp_sd`) |
| `-b` | Batch size (default: 40) |
| `-s` | Subject ID, 0–15 (default: 0) |
| `-o` | Output directory |
| `-p` | Pretrained model filename (optional) |

### 1. Generate CLIP embeddings (one-time per dataset)

```bash
python src/blip_clip.py -d data/ -g all -m mlp_sd -o output/
```

### 2. Object classification

```bash
# Deep model (EEGNet)
python src/object_classification.py -d data/ -g coarse -m eegnet -s 0 -o output/

# Sklearn baseline
python src/object_classification.py -d data/ -g coarse -m svm -s 0 -o output/
```

### 3. Train EEG to create image mapper (MLP to CLIP embedding space)

```bash
python src/image_generation.py -d data/ -g coarse -m mlp_sd -s 0 -o output/
```

### 4. Generate images from EEG

```bash
python src/gen_eval.py -d data/ -g coarse -m mlp_sd -s 0 -o output/ -p mlpsd_s0_0.pth
```

### 5. Visualization
#### 5.1 Dataset visualization
Open `viz.ipynb` in Jupyter to explore the EEG data interactively.