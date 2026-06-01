import numpy as np
import torch


class InternViTFeatureLookup:
    """InternViT feature store keyed by (concept, image_file).

    Loads a single `internvit_features.npy` produced by
    ``scripts/extract_internvit_features.py``, which is a pickled dict::

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
        """Return ``(batch, n_layers, 3200)`` float32 tensor.

        Args:
            concepts:    concept name for each sample
            image_files: image filename for each sample
        """
        arrs = [
            self.features[(c, f)].astype(np.float32)
            for c, f in zip(concepts, image_files)
        ]
        return torch.from_numpy(np.stack(arrs))

    def __len__(self) -> int:
        return len(self.features)
