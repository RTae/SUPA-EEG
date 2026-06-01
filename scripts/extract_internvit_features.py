"""Extract InternViT multilayer features from THINGS-EEG images.

Usage:
  python scripts/extract_internvit_features.py
  python scripts/extract_internvit_features.py --device cuda:1
  python scripts/extract_internvit_features.py --batch_size 32

Output:
  data/things_eeg/image_feature/internvit_multilevel_20_24_28_32_36/
    internvit_features.npy   dict {(concept, img_file): ndarray(n_layers, 3200) float16}
"""

import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, CLIPImageProcessor
from loguru import logger


OUTPUT_DIM = 3200   # InternViT-6B hidden dim (architectural constant)


def load_internvit(model_name: str, device: torch.device):
    """Load frozen InternViT encoder (bfloat16, following the official HF example)."""
    logger.info(f"Loading InternViT from {model_name}...")
    processor = CLIPImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    for p in model.parameters():
        p.requires_grad = False
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
    pixel_values = inputs["pixel_values"].to(dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        model(pixel_values=pixel_values)

    for h in hooks:
        h.remove()

    return {lid: intermediate[lid].numpy() for lid in layer_ids}


def extract_directory(
    image_dir: str,
    model,
    processor,
    layer_ids: list[int],
    device: torch.device,
    batch_size: int,
) -> dict:
    """Extract InternViT features for all images in a concept-organised directory.

    Expected layout::

        <image_dir>/<concept>/<image_file>

    Args:
        image_dir:  Root directory containing concept sub-folders.
        model:      Frozen InternViT model.
        processor:  HuggingFace image processor.
        layer_ids:  Transformer layer indices to extract.
        device:     Torch device.
        batch_size: Images per forward pass.

    Returns:
        dict mapping ``(concept, img_file)`` to ``ndarray(n_layers, 3200)`` float16.
    """
    entries: list[tuple[str, str, str]] = []
    for concept in sorted(os.listdir(image_dir)):
        concept_dir = os.path.join(image_dir, concept)
        if not os.path.isdir(concept_dir):
            continue
        for img_file in sorted(os.listdir(concept_dir)):
            img_path = os.path.join(concept_dir, img_file)
            if os.path.isfile(img_path) and img_file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                entries.append((concept, img_file, img_path))

    logger.info(f"Scanning {image_dir}: {len(entries)} images found.")
    features: dict = {}

    for start in tqdm(range(0, len(entries), batch_size), desc=f"Extracting {image_dir}"):
        batch_entries = entries[start : start + batch_size]
        images = [Image.open(e[2]).convert("RGB") for e in batch_entries]
        feats = extract_layer_features(model, processor, images, layer_ids, device)
        # feats: {lid: (n_batch, 3200)}
        for i, (concept, img_file, _) in enumerate(batch_entries):
            features[(concept, img_file)] = np.stack(
                [feats[lid][i] for lid in layer_ids]
            ).astype(np.float16)  # (n_layers, 3200)

    return features


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device    = torch.device(cfg.device)
    layer_ids = list(cfg.layer_ids)
    logger.info(f"Device: {device} | Layers: {layer_ids}")

    processor, model = load_internvit(cfg.internvit_model, device)

    features: dict = {}
    for image_dir in (cfg.train_img_dir, cfg.test_img_dir):
        features.update(
            extract_directory(image_dir, model, processor, layer_ids, device, cfg.batch_size)
        )

    os.makedirs(cfg.internvit_dir, exist_ok=True)
    out_path = os.path.join(cfg.internvit_dir, "internvit_features.npy")
    logger.info(f"Saving {len(features):,} entries → {out_path}")
    np.save(out_path, features)
    logger.info("Extraction complete.")


def ensure_internvit_features(
    internvit_dir: str,
    layer_ids: list[int],
    train_img_dir: str,
    test_img_dir: str,
    model_name: str = "OpenGVLab/InternViT-6B-448px-V1-5",
    device: str = "cpu",
    batch_size: int = 64,
) -> None:
    """Check if InternViT features exist. Extract if missing.

    Called automatically at the start of training.

    Args:
        internvit_dir:  Directory for the output .npy file
        layer_ids:      e.g. [20, 24, 28, 32, 36]
        train_img_dir:  data/things_eeg/training_images
        test_img_dir:   data/things_eeg/test_images
        model_name:     HuggingFace model ID
        device:         device for InternViT extraction
        batch_size:     images per batch during extraction
    """
    out_path = os.path.join(internvit_dir, "internvit_features.npy")

    if os.path.isfile(out_path):
        logger.info(f"InternViT features found at {out_path}. Skipping extraction.")
        return

    logger.warning(
        f"InternViT feature file not found: {out_path}\n"
        "Running offline extraction — this may take 10-30 minutes."
    )

    processor, model = load_internvit(model_name, torch.device(device))

    features: dict = {}
    for image_dir in (train_img_dir, test_img_dir):
        features.update(
            extract_directory(image_dir, model, processor, layer_ids, torch.device(device), batch_size)
        )

    os.makedirs(internvit_dir, exist_ok=True)
    np.save(out_path, features)

    if not os.path.isfile(out_path):
        raise FileNotFoundError(f"Extraction completed but file still missing: {out_path}")
    logger.info(f"InternViT features saved to {out_path} ({len(features):,} images).")


if __name__ == "__main__":
    main()
