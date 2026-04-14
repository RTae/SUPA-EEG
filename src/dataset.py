import os
import pickle
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
import random
from collections import defaultdict
from loguru import logger

class BalancedBatchSampler(Sampler):
    """
    In triplet-based semantic training, we want each batch to contain multiple examples of a few classes.
    This sampler ensures that by grouping dataset indices by class and sampling accordingly.
    It requires a label_map to remap raw dataset labels to contiguous class indices.
    
    For example, with num_classes_per_batch=8 and samples_per_class=4, each batch will have 32 samples from 8 classes.
    """

    def __init__(self, dataset, label_map: dict[int, int], num_classes_per_batch: int, samples_per_class: int) -> None:
        super().__init__()
        self.samples_per_class = samples_per_class
        self.num_classes_per_batch = num_classes_per_batch

        # Group dataset indices by remapped class label.
        groups: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(dataset)):
            _, raw_label = dataset[idx]
            remapped = label_map.get(int(raw_label))
            if remapped is not None:
                groups[remapped].append(idx)
        self.groups = {k: v for k, v in groups.items() if len(v) >= samples_per_class}
        self.classes = list(self.groups.keys())
        self.num_batches = max(1, len(self.classes) // num_classes_per_batch)

    def __iter__(self):
        classes = self.classes.copy()
        random.shuffle(classes)
        for i in range(0, len(classes) - self.num_classes_per_batch + 1, self.num_classes_per_batch):
            batch_classes = classes[i : i + self.num_classes_per_batch]
            batch = []
            for cls in batch_classes:
                batch.extend(random.choices(self.groups[cls], k=self.samples_per_class))
            random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches * self.num_classes_per_batch * self.samples_per_class



class EEGImageNetDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        transform: Callable[[Image.Image], Any] | None = None,
        pth_name: str = "EEG-ImageNet.pth",
        subject: int = -1,
        granularity: str = "all",
        eeg_window: tuple[int, int] = (40, 440),
        map_location: str | torch.device = "cpu",
    ) -> None:
        self.dataset_dir = dataset_dir
        self.subject = subject
        self.granularity = granularity
        self.transform = transform
        self.eeg_window = eeg_window

        loaded = self._load_checkpoint(os.path.join(self.dataset_dir, pth_name), map_location)
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}

        chosen_data = self._filter_subject(loaded["dataset"], self.subject)
        self.data = self._filter_granularity(chosen_data, self.granularity)

        self.use_frequency_feat = False
        self.frequency_feat = None
        self.use_image_label = False

    @classmethod
    def from_args(
        cls,
        args: Any,
        transform: Callable[[Image.Image], Any] | None = None,
        pth_name: str = "EEG-ImageNet.pth",
        eeg_window: tuple[int, int] = (40, 440),
        map_location: str | torch.device = "cpu",
    ) -> "EEGImageNetDataset":
        dataset_dir = cls._read_opt(args, "dataset_dir", required=True)
        subject = cls._read_opt(args, "subject", default=-1)
        granularity = cls._read_opt(args, "granularity", default="all")
        return cls(
            dataset_dir=dataset_dir,
            transform=transform,
            pth_name=pth_name,
            subject=subject,
            granularity=granularity,
            eeg_window=eeg_window,
            map_location=map_location,
        )

    @staticmethod
    def _read_opt(args: Any, name: str, default: Any = None, required: bool = False) -> Any:
        if args is not None:
            if isinstance(args, dict) and name in args:
                return args[name]
            if hasattr(args, name):
                return getattr(args, name)
        if required:
            raise ValueError(f"Missing required option '{name}'.")
        return default

    @staticmethod
    def _load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        # Our data is float64, but some environments (like MPS) don't support it. Load to CPU first and convert to float32 if needed.
        if map_location == torch.device("mps"):
            map_location = torch.device("cpu")
            
        logger.info(f"Loading dataset checkpoint from: {path}" f" (map_location={map_location})")
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except pickle.UnpicklingError:
            return torch.load(path, map_location=map_location, weights_only=False)

    def _filter_subject(self, dataset: list[dict[str, Any]], subject: int) -> list[dict[str, Any]]:
        if subject == -1:
            return dataset
        return [item for item in dataset if item.get("subject") == subject]

    def _filter_granularity(self, dataset: list[dict[str, Any]], granularity: str) -> list[dict[str, Any]]:
        if granularity == "all":
            return dataset
        if granularity == "coarse":
            return [item for item in dataset if item.get("granularity") == "coarse"]
        if granularity == "fine":
            return [item for item in dataset if item.get("granularity") == "fine"]

        # Support grouped fine categories like "fine0", "fine1", etc.
        if isinstance(granularity, str) and granularity.startswith("fine") and granularity[4:].isdigit():
            fine_num = int(granularity[4:])
        elif isinstance(granularity, str) and granularity[-1].isdigit():
            # Keep compatibility with previous behavior for values like "granularity3".
            fine_num = int(granularity[-1])
        else:
            raise ValueError(
                "Invalid granularity. Expected one of: 'all', 'coarse', 'fine', 'fineN' (e.g. fine3)."
            )

        start = 8 * fine_num
        end = start + 8
        return [
            item
            for item in dataset
            if item.get("granularity") == "fine"
            and start <= self.label_to_index.get(item.get("label"), -1) < end
        ]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any]:
        
        if self.use_image_label:
            path = self.data[index]["image"]
            image_path = os.path.join(self.dataset_dir, "imageNet_images", path.split('_')[0], path)
            with Image.open(image_path) as image:
                if image.mode == "L":
                    image = image.convert("RGB")
                label = self.transform(image) if self.transform else path
        else:
            label = self.label_to_index[self.data[index]["label"]]

        if self.use_frequency_feat:
            feat = self.frequency_feat[index]
        else:
            eeg_data = self.data[index]["eeg_data"].float()
            start, end = self.eeg_window
            feat = eeg_data[:, start:end]
        return feat, label

    def __len__(self) -> int:
        return len(self.data)

    def add_frequency_feat(self, feat: Any) -> None:
        self.set_frequency_feat(feat)

    def set_frequency_feat(self, feat: Any) -> None:
        if len(feat) != len(self.data):
            raise ValueError("Frequency features must have same length")
        if isinstance(feat, torch.Tensor):
            self.frequency_feat = feat.float()
        else:
            self.frequency_feat = torch.as_tensor(feat).float()
        self.use_frequency_feat = True

    def clear_frequency_feat(self) -> None:
        self.frequency_feat = None
        self.use_frequency_feat = False

    def set_label_mode(self, mode: str) -> None:
        if mode not in ("index", "image"):
            raise ValueError("label mode must be 'index' or 'image'")
        self.use_image_label = mode == "image"