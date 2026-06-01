from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, CLIPImageProcessor
from loguru import logger


OUTPUT_DIM = 3200   # InternViT-6B hidden dim (architectural constant)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_internvit(model_name: str, device: torch.device):
    """Load frozen InternViT encoder directly to device using device_map.

    Uses ``device_map`` to stream shards directly to the target device,
    avoiding a full float32 copy in CPU RAM (~24 GB). Cast to bfloat16
    after init to sidestep meta-tensor issues that arise when ``torch_dtype``
    is passed to ``from_pretrained``.

    Args:
        model_name: HuggingFace model ID, e.g. ``OpenGVLab/InternViT-6B-448px-V1-5``
        device:     Target torch device.

    Returns:
        ``(processor, model)`` tuple — processor is a ``CLIPImageProcessor``,
        model is frozen bfloat16 InternViT.
    """
    logger.info(f"Loading InternViT from {model_name}...")
    processor = CLIPImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map={"": str(device)},
    )
    model = model.to(dtype=torch.bfloat16)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    logger.info(
        f"InternViT loaded on {device} as bfloat16. "
        f"Parameters: {sum(p.numel() for p in model.parameters()):,} (all frozen)"
    )
    return processor, model


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


# ---------------------------------------------------------------------------
# Feature lookup (used at training / evaluation time)
# ---------------------------------------------------------------------------

class InternViTFeatureLookup:
    """InternViT feature store keyed by (concept, image_file).

    Loads a single ``internvit_features.npy`` produced by
    ``scripts/extract_internvit_features.py``::

        {(concept_str, img_file_str): np.ndarray(n_layers, 3200) float16}

    Args:
        feature_path: path to ``internvit_features.npy``
    """

    def __init__(self, feature_path: str) -> None:
        self.features: dict = np.load(feature_path, allow_pickle=True).item()

    def retrieve_batch(
        self,
        concepts: list[str],
        image_files: list[str],
    ) -> torch.Tensor:
        """Return ``(batch, n_layers, 3200)`` float32 tensor."""
        arrs = [
            self.features[(c, f)].astype(np.float32)
            for c, f in zip(concepts, image_files)
        ]
        return torch.from_numpy(np.stack(arrs))

    def __len__(self) -> int:
        return len(self.features)


# ---------------------------------------------------------------------------
# Feature guard (called at training startup)
# ---------------------------------------------------------------------------

def ensure_internvit_features(
    internvit_dir: str,
    layer_ids: list[int],
    train_img_dir: str,
    test_img_dir: str,
    model_name: str = "OpenGVLab/InternViT-6B-448px-V1-5",
    device: str = "cpu",
    batch_size: int = 64,
) -> None:
    """Check if InternViT features exist; extract if missing.

    Called automatically at the start of training (no-op if the file is present).

    Args:
        internvit_dir:  Directory for the output ``.npy`` file.
        layer_ids:      e.g. ``[20, 24, 28, 32, 36]``
        train_img_dir:  ``data/things_eeg/training_images``
        test_img_dir:   ``data/things_eeg/test_images``
        model_name:     HuggingFace model ID.
        device:         Device for InternViT extraction.
        batch_size:     Images per batch during extraction.
    """
    out_path = os.path.join(internvit_dir, "internvit_features.npy")

    if os.path.isfile(out_path):
        logger.info(f"InternViT features found at {out_path}. Skipping extraction.")
        return

    logger.warning(
        f"InternViT feature file not found: {out_path}\n"
        "Running offline extraction — this may take 10-30 minutes."
    )

    _device = torch.device(device)
    processor, model = load_internvit(model_name, _device)

    features: dict = {}
    for image_dir in (train_img_dir, test_img_dir):
        features.update(
            extract_directory(image_dir, model, processor, layer_ids, _device, batch_size)
        )

    os.makedirs(internvit_dir, exist_ok=True)
    np.save(out_path, features)

    if not os.path.isfile(out_path):
        raise FileNotFoundError(f"Extraction completed but file still missing: {out_path}")
    logger.info(f"InternViT features saved to {out_path} ({len(features):,} images).")
