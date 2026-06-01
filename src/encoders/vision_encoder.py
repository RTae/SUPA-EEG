"""Visual feature lookups for SUPAEEG training."""

from __future__ import annotations

import os

import numpy as np
import torch


class InternViTFeatureLookup:
    """Load pre-extracted InternViT multilayer .npy features.

    Files are expected at:
        feature_dir/image_{split}_layer{lid}.npy

    Each file has shape:
        train: (n_concepts, n_images_per_concept, 3200) float16
        test:  (n_concepts, 1, 3200)                   float16

    After loading they are stacked into:
        self.features: (n_concepts, n_images, n_layers, 3200) float32

    Args:
        feature_dir: path to internvit_multilevel_20_24_28_32_36/
        layer_ids:   e.g. [20, 24, 28, 32, 36]
        split:       'train' or 'test'
    """

    def __init__(self, feature_dir: str, layer_ids: list[int], split: str) -> None:
        arrays = []
        for lid in layer_ids:
            path = os.path.join(feature_dir, f"image_{split}_layer{lid}.npy")
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"InternViT feature file not found: {path}\n"
                    f"Run: python scripts/extract_internvit_features.py"
                )
            arrays.append(np.load(path).astype(np.float32))
        # stack -> (n_concepts, n_images, n_layers, 3200)
        self.features  = np.stack(arrays, axis=2)
        self.layer_ids = layer_ids
        self.split     = split

    def retrieve_batch(
        self,
        concept_indices: list[int],
        image_indices: list[int],
    ) -> torch.Tensor:
        """Return (batch, n_layers, 3200) float32 tensor.

        Args:
            concept_indices: integer concept index for each sample (0-based)
            image_indices:   integer image-within-concept index for each sample (0-based)
        """
        feats = self.features[concept_indices, image_indices]
        return torch.from_numpy(feats)

    def retrieve_all_test_concepts(self) -> torch.Tensor:
        """Return (n_concepts, n_layers, 3200) averaged over image repetitions."""
        return torch.from_numpy(self.features.mean(axis=1))

    def __len__(self) -> int:
        return self.features.shape[0]
