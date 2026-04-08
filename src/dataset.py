import os
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


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
        # Always load to CPU to avoid dtype issues (e.g. MPS doesn't support float64).
        # Device transfer is handled later by DataLoader / training loop.
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            return torch.load(path, map_location="cpu", weights_only=False)

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


@dataclass(slots=True)
class SyntheticSample:
    """One synthetic CrossPT-EEG style trial."""

    eeg: torch.Tensor
    label: int
    subject_id: int


class CrossPTEEGSyntheticDataset(Dataset[tuple[torch.Tensor, int, int]]):
    """Synthetic dataset with CrossPT-EEG shape and subject shifts.

    Each trial has shape (62, 1000) and labels are in [0, 79].
    """

    def __init__(
        self,
        seq_len: int = 1000,
        n_channels: int = 62,
        num_subjects: int = 16,
        samples_per_subject: int = 480,
        seed: int = 7,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.num_subjects = num_subjects
        self.samples_per_subject = samples_per_subject
        self._generator = torch.Generator().manual_seed(seed)
        self.samples = self._build_samples()

    def _build_samples(self) -> list[SyntheticSample]:
        total_samples = self.num_subjects * self.samples_per_subject
        time = torch.linspace(0.0, 1.0, self.seq_len)
        time = time.unsqueeze(0).expand(self.n_channels, -1)
        channel_axis = torch.linspace(-1.0, 1.0, self.n_channels).unsqueeze(1)
        stimulus_envelope = torch.sigmoid((time - 0.08) * 24.0) * torch.exp(-1.8 * time)

        class_codes = torch.randn(80, self.n_channels, 4, generator=self._generator) * 0.18
        subject_offsets = torch.randn(self.num_subjects, self.n_channels, 1, generator=self._generator) * 0.08
        subject_gains = 1.0 + 0.12 * torch.randn(self.num_subjects, self.n_channels, 1, generator=self._generator)
        phase_offsets = 2.0 * torch.pi * torch.rand(self.num_subjects, 4, generator=self._generator)
        frequencies = torch.tensor([6.0, 10.0, 18.0, 28.0]).view(1, 1, 4)

        samples: list[SyntheticSample] = []
        for sample_index in range(total_samples):
            subject_id = sample_index // self.samples_per_subject
            label = sample_index % 80
            code = class_codes[label]
            phase = phase_offsets[subject_id].view(1, 4)

            harmonics = torch.sin(2.0 * torch.pi * frequencies.squeeze(0) * time.unsqueeze(-1) + phase)
            harmonics = harmonics.permute(0, 2, 1)
            template = (code.unsqueeze(-1) * harmonics).sum(dim=1)
            template = template * (1.0 + 0.15 * channel_axis)
            template = template * stimulus_envelope

            drift = 0.04 * torch.sin(2.0 * torch.pi * (subject_id + 1) * time / 6.0)
            colored_noise = torch.randn(self.n_channels, self.seq_len, generator=self._generator)
            colored_noise = F.avg_pool1d(colored_noise.unsqueeze(0), kernel_size=9, stride=1, padding=4).squeeze(0)

            eeg = subject_gains[subject_id] * template
            eeg = eeg + subject_offsets[subject_id] + drift + 0.05 * colored_noise
            eeg = eeg.float().contiguous()
            samples.append(SyntheticSample(eeg=eeg, label=label, subject_id=subject_id))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        sample = self.samples[index]
        return sample.eeg, sample.label, sample.subject_id


def map_crosspt_label(raw_label: int, task: str, fine_group: int = 0) -> int | None:
    """Map an 80-way synthetic label into all/coarse/fine task spaces."""
    if task == "all":
        return raw_label
    if task == "coarse":
        return raw_label // 2
    if task == "fine":
        start = fine_group * 8
        end = start + 8
        if start <= raw_label < end:
            return raw_label - start
        return None
    raise ValueError(f"Unsupported task '{task}'.")


def _subject_indices(
    dataset: CrossPTEEGSyntheticDataset,
    subjects: Iterable[int],
    task: str,
    fine_group: int,
) -> list[int]:
    wanted = set(subjects)
    indices: list[int] = []
    for idx, sample in enumerate(dataset.samples):
        if sample.subject_id not in wanted:
            continue
        if map_crosspt_label(sample.label, task, fine_group) is None:
            continue
        indices.append(idx)
    return indices


def synthetic_crosspt_collate(task: str, fine_group: int):
    """Return a task-aware collate function for synthetic CrossPT batches."""

    def collate_fn(batch: list[tuple[torch.Tensor, int, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eeg = torch.stack([x[0] for x in batch], dim=0).contiguous()
        labels = torch.tensor([map_crosspt_label(x[1], task, fine_group) for x in batch], dtype=torch.long)
        subjects = torch.tensor([x[2] for x in batch], dtype=torch.long)
        return eeg, labels, subjects

    return collate_fn


def build_synthetic_crosspt_loaders(
    *,
    seq_len: int = 1000,
    n_channels: int = 62,
    num_subjects: int = 16,
    samples_per_subject: int = 480,
    batch_size: int = 64,
    task: str = "all",
    fine_group: int = 0,
    train_subjects: tuple[int, ...] = tuple(range(12)),
    test_subjects: tuple[int, ...] = (12, 13, 14, 15),
    num_workers: int = 0,
    seed: int = 7,
) -> tuple[
    CrossPTEEGSyntheticDataset,
    DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    """Create pretrain/train/test loaders for a synthetic cross-participant split."""
    dataset = CrossPTEEGSyntheticDataset(
        seq_len=seq_len,
        n_channels=n_channels,
        num_subjects=num_subjects,
        samples_per_subject=samples_per_subject,
        seed=seed,
    )
    pretrain_idx = _subject_indices(dataset, train_subjects, task="all", fine_group=fine_group)
    train_idx = _subject_indices(dataset, train_subjects, task=task, fine_group=fine_group)
    test_idx = _subject_indices(dataset, test_subjects, task=task, fine_group=fine_group)

    pretrain_loader = DataLoader(
        Subset(dataset, pretrain_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=synthetic_crosspt_collate("all", fine_group),
    )
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=synthetic_crosspt_collate(task, fine_group),
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=synthetic_crosspt_collate(task, fine_group),
    )
    return dataset, pretrain_loader, train_loader, test_loader
