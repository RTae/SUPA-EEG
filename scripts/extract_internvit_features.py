"""Extract InternViT multilayer features from THINGS-EEG images.

Usage:
  python scripts/extract_internvit_features.py
  python scripts/extract_internvit_features.py --device cuda:1
  python scripts/extract_internvit_features.py --batch_size 32

Output:
  data/things_eeg/image_feature/internvit_multilevel_20_24_28_32_36/
    image_train_layer20.npy   (1654, 10, 3200) float16
    image_train_layer24.npy   ...
    image_train_layer28.npy
    image_train_layer32.npy
    image_train_layer36.npy
    image_test_layer20.npy    (200, 1, 3200)   float16
    image_test_layer24.npy    ...
    image_test_layer28.npy
    image_test_layer32.npy
    image_test_layer36.npy
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor
from loguru import logger


LAYER_IDS     = [20, 24, 28, 32, 36]
OUTPUT_DIM    = 3200   # InternViT-6B hidden dim
MODEL_NAME    = "OpenGVLab/InternViT-6B-448px-V1-5"
OUTPUT_DIR    = "data/things_eeg/image_feature/internvit_multilevel_20_24_28_32_36"
TRAIN_IMG_DIR = "data/things_eeg/training_images"
TEST_IMG_DIR  = "data/things_eeg/test_images"

# image_metadata.npy contains concept and file ordering
# must match the ordering used by ThingsEEGDataset
METADATA_PATH = "data/things_eeg/image_metadata.npy"


def load_internvit(model_name: str, device: torch.device):
    """Load frozen InternViT encoder with forward hooks on target layers."""
    logger.info(f"Loading InternViT from {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    logger.info(
        f"InternViT loaded. Parameters: {sum(p.numel() for p in model.parameters()):,} (all frozen)"
    )
    return processor, model


def extract_layer_features(
    model,
    processor,
    images: list,
    layer_ids: list[int],
    device: torch.device,
) -> dict[int, np.ndarray]:
    """Extract features from specific transformer layers via forward hooks.

    Args:
        model:     Frozen InternViT model
        processor: HuggingFace image processor
        images:    List of PIL images
        layer_ids: List of layer indices to extract e.g. [20, 24, 28, 32, 36]
        device:    torch device

    Returns:
        dict {layer_id: np.ndarray shape (n_images, 3200) float32}
    """
    intermediate: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(lid: int):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # hidden: (batch, tokens, dim) — mean pool over tokens -> (batch, dim)
            intermediate[lid] = hidden.mean(dim=1).detach().float().cpu()
        return hook

    def find_layers(m):
        """Return the transformer layer list by probing known attribute paths."""
        for path in ("encoder.layers", "vision_model.encoder.layers"):
            obj = m
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        raise AttributeError(
            "Cannot find transformer layers. "
            "Check model structure with: print(model)"
        )

    layers = find_layers(model)

    for lid in layer_ids:
        hooks.append(layers[lid].register_forward_hook(make_hook(lid)))

    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        model(pixel_values=pixel_values)

    for h in hooks:
        h.remove()

    return {lid: intermediate[lid].numpy() for lid in layer_ids}


def get_ordered_image_paths(
    image_dir: str,
    metadata: dict,
    split: str,
) -> list[tuple[str, str, str]]:
    """Return ordered list of image paths matching metadata ordering.

    Args:
        image_dir: path to training_images or test_images
        metadata:  loaded image_metadata.npy dict
        split:     'train' or 'test'

    Returns:
        List of (concept, image_file, full_path) tuples
        in the same order as metadata
    """
    concepts  = metadata[f"{split}_img_concepts"]
    img_files = metadata[f"{split}_img_files"]

    result = []
    for concept, img_file in zip(concepts, img_files):
        path = os.path.join(image_dir, concept, img_file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image not found: {path}")
        result.append((concept, img_file, path))
    return result


def extract_split(
    split: str,
    image_dir: str,
    metadata: dict,
    model,
    processor,
    layer_ids: list[int],
    device: torch.device,
    batch_size: int,
    output_dir: str,
) -> None:
    """Extract features for one split (train or test) and save per layer.

    Output arrays:
      train: (n_concepts, n_images_per_concept, 3200) float16
      test:  (n_concepts, 1, 3200) float16
    """
    image_paths = get_ordered_image_paths(image_dir, metadata, split)
    n_samples   = len(image_paths)

    concepts = metadata[f"{split}_img_concepts"]

    # Determine concept grouping
    # train: 1654 concepts x 10 images = 16540 total
    # test:  200  concepts x 1  image  = 200   total
    unique_concepts = list(dict.fromkeys(concepts))   # ordered unique
    n_concepts      = len(unique_concepts)
    n_imgs_per      = n_samples // n_concepts
    logger.info(
        f"Split={split} | n_concepts={n_concepts} | "
        f"n_imgs_per_concept={n_imgs_per} | total={n_samples}"
    )

    # Accumulate all features
    all_features: dict[int, list[np.ndarray]] = {lid: [] for lid in layer_ids}

    for start in tqdm(range(0, n_samples, batch_size), desc=f"Extracting {split}"):
        end         = min(start + batch_size, n_samples)
        batch_paths = [image_paths[i][2] for i in range(start, end)]
        images      = [Image.open(p).convert("RGB") for p in batch_paths]

        feats = extract_layer_features(model, processor, images, layer_ids, device)
        for lid in layer_ids:
            all_features[lid].append(feats[lid])

    os.makedirs(output_dir, exist_ok=True)
    for lid in layer_ids:
        arr = np.concatenate(all_features[lid], axis=0)  # (n_samples, 3200)
        # reshape to (n_concepts, n_imgs_per, 3200)
        arr = arr.reshape(n_concepts, n_imgs_per, OUTPUT_DIM)
        arr = arr.astype(np.float16)

        out_path = os.path.join(output_dir, f"image_{split}_layer{lid}.npy")
        np.save(out_path, arr)
        logger.info(f"Saved {out_path} shape={arr.shape} dtype={arr.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract InternViT multilayer features from THINGS-EEG images"
    )
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--model_name",    default=MODEL_NAME)
    parser.add_argument("--output_dir",    default=OUTPUT_DIR)
    parser.add_argument("--train_img_dir", default=TRAIN_IMG_DIR)
    parser.add_argument("--test_img_dir",  default=TEST_IMG_DIR)
    parser.add_argument("--metadata_path", default=METADATA_PATH)
    parser.add_argument("--layer_ids",     nargs="+", type=int, default=LAYER_IDS)
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Device: {device} | Layers: {args.layer_ids}")

    metadata          = np.load(args.metadata_path, allow_pickle=True).item()
    processor, model  = load_internvit(args.model_name, device)

    extract_split(
        "train", args.train_img_dir, metadata,
        model, processor, args.layer_ids,
        device, args.batch_size, args.output_dir,
    )
    extract_split(
        "test", args.test_img_dir, metadata,
        model, processor, args.layer_ids,
        device, args.batch_size, args.output_dir,
    )
    logger.info("Extraction complete.")


if __name__ == "__main__":
    main()
