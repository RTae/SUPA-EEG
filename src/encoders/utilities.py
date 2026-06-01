from PIL import Image
from tqdm import tqdm

import numpy as np
import torch
import os
from loguru import logger

# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _find_layers(m):
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
        "Cannot find transformer layers. Check model structure with: print(model)"
    )


def extract_layer_features(
    model,
    processor,
    images: list,
    layer_ids: list[int],
    device: torch.device,
) -> dict[int, np.ndarray]:
    """Extract hidden states from specific transformer layers via forward hooks.

    Args:
        model:     Frozen InternViT model.
        processor: ``CLIPImageProcessor`` instance.
        images:    List of PIL images.
        layer_ids: Transformer layer indices to capture, e.g. ``[20, 24, 28, 32, 36]``.
        device:    Torch device.

    Returns:
        ``{layer_id: ndarray(n_images, 3200)}`` in float32.
    """
    intermediate: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(lid: int):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            intermediate[lid] = hidden.mean(dim=1).detach().float().cpu()
        return hook

    layers = _find_layers(model)
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
    """Extract InternViT features for every image in a concept-organised directory.

    Expected layout::

        <image_dir>/<concept>/<image_file>

    Args:
        image_dir:  Root directory containing concept sub-folders.
        model:      Frozen InternViT model.
        processor:  ``CLIPImageProcessor`` instance.
        layer_ids:  Transformer layer indices to extract.
        device:     Torch device.
        batch_size: Images per forward pass.

    Returns:
        ``{(concept, img_file): ndarray(n_layers, 3200)}`` in float16.
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

    for start in tqdm(range(0, len(entries), batch_size), desc=f"Extracting {os.path.basename(image_dir)}"):
        batch_entries = entries[start : start + batch_size]
        images = [Image.open(e[2]).convert("RGB") for e in batch_entries]
        feats = extract_layer_features(model, processor, images, layer_ids, device)
        for i, (concept, img_file, _) in enumerate(batch_entries):
            features[(concept, img_file)] = np.stack(
                [feats[lid][i] for lid in layer_ids]
            ).astype(np.float16)  # (n_layers, 3200)

    return features

