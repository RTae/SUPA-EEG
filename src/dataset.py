import os
import pickle
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
import torchvision.transforms.functional as tf
import random
from collections import defaultdict
from loguru import logger
import numpy as np

class BalancedBatchSampler(Sampler):
    """
    In triplet-based semantic training, we want each batch to contain multiple examples of a few classes.
    This sampler ensures that by grouping dataset indices by class and sampling accordingly.
    It assumes the dataset already returns contiguous class indices.
    
    For example, with num_classes_per_batch=8 and samples_per_class=4, each batch will have 32 samples from 8 classes.
    """

    def __init__(self, dataset, num_classes_per_batch: int, samples_per_class: int) -> None:
        super().__init__()
        self.samples_per_class = samples_per_class
        self.num_classes_per_batch = num_classes_per_batch

        # Group dataset indices by the labels returned by the dataset.
        groups: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            groups[int(label)].append(idx)
        self.groups = {k: v for k, v in groups.items() if len(v) >= samples_per_class}
        self.classes = list(self.groups.keys())

        if len(self.classes) < num_classes_per_batch:
            raise ValueError(
                "BalancedBatchSampler requires at least num_classes_per_batch eligible classes"
            )

        self.class_chunks = {
            cls: len(indices) // self.samples_per_class for cls, indices in self.groups.items()
        }
        self.num_batches = sum(self.class_chunks.values()) // self.num_classes_per_batch
        if self.num_batches < 1:
            raise ValueError("BalancedBatchSampler could not form a full batch with the current settings")

    def __iter__(self):
        remaining_chunks: dict[int, list[list[int]]] = {}
        for cls, indices in self.groups.items():
            shuffled = indices.copy()
            random.shuffle(shuffled)
            chunks = [
                shuffled[i : i + self.samples_per_class]
                for i in range(0, len(shuffled) - self.samples_per_class + 1, self.samples_per_class)
            ]
            random.shuffle(chunks)
            remaining_chunks[cls] = chunks

        eligible_classes = [cls for cls, chunks in remaining_chunks.items() if chunks]
        while len(eligible_classes) >= self.num_classes_per_batch:
            batch_classes = random.sample(eligible_classes, k=self.num_classes_per_batch)
            batch = []
            for cls in batch_classes:
                batch.extend(remaining_chunks[cls].pop())
            random.shuffle(batch)
            yield batch

            eligible_classes = [cls for cls, chunks in remaining_chunks.items() if chunks]

    def __len__(self) -> int:
        return self.num_batches

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
        self.device = map_location if isinstance(map_location, torch.device) else torch.device(map_location)
        
        path = os.path.join(self.dataset_dir, pth_name)
        
        logger.info(f"data_path={path} " 
            f" device={self.device} "
            f" granularity={self.granularity} "
            f" subject={self.subject}")

        loaded = self._load_checkpoint(path, map_location)
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.global_label_to_index = {label: idx for idx, label in enumerate(self.labels)}

        chosen_data = self._filter_subject(loaded["dataset"], self.subject)
        self.data = self._filter_granularity(chosen_data, self.granularity)
        present_labels = {item["label"] for item in self.data}
        self.index_to_label = [label for label in self.labels if label in present_labels]
        self.label_to_index = {label: idx for idx, label in enumerate(self.index_to_label)}

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
        # Keep the raw dataset on CPU. Training code moves mini-batches to the target device,
        # and frequency features can be transferred after preprocessing if needed.
        if not isinstance(map_location, torch.device):
            map_location = torch.device(map_location)

        load_location = torch.device("cpu") if map_location.type in {"cuda", "mps"} else map_location

        try:
            return torch.load(path, map_location=load_location, weights_only=True)
        except pickle.UnpicklingError:
            return torch.load(path, map_location=load_location, weights_only=False)

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
            and start <= self.global_label_to_index.get(item.get("label"), -1) < end
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

        freq_tensor = torch.as_tensor(feat).float()
        if self.device.type != "cpu":
            freq_tensor = freq_tensor.to(self.device)

        self.frequency_feat = freq_tensor
        self.use_frequency_feat = True

    def clear_frequency_feat(self) -> None:
        self.frequency_feat = None
        self.use_frequency_feat = False

    def set_label_mode(self, mode: str) -> None:
        if mode not in ("index", "image"):
            raise ValueError("label mode must be 'index' or 'image'")
        self.use_image_label = mode == "image"
        
class ThingsEEGDataset(Dataset):
    def __init__(
        self,
        dataset_dir="./data/things_eeg",
        data_type = "train",
        subject = -1,
        transform: Callable[[Image.Image], Any] | None = None,
        device=torch.device("cpu"),
        max_subjects: int = 10,
        load_images: bool = True,
    ) -> None:
        
        self.max_subjects = max_subjects
        self.data_type = data_type
        
        if data_type not in ["train", "test"]:
            raise ValueError(f"Invalid data_type: {data_type}. Expected 'train' or 'test'.")
        
        if not (0 < subject <=   self.max_subjects or subject == -1):
            raise ValueError(f"Invalid subject index: {subject}. Must be between 1 and {self.max_subjects}, or -1 for all subjects.")
        
        # Get list of eeg data from each subject
        eeg_folder_list = [f for f in os.listdir(dataset_dir) if f.startswith("sub")]
        eeg_file_name = "preprocessed_eeg_training" if data_type == "train" else "preprocessed_eeg_test"

        # Image folders
        image_meta_data = os.path.join(dataset_dir, "image_metadata.npy")
        image_dir = os.path.join(dataset_dir, "training_images") if data_type == "train" else os.path.join(dataset_dir, "test_images")
        
        logger.info(f"dataset_dir={dataset_dir} device={device} found_subjects={len(eeg_folder_list)}")
        logger.info("Loadding EEG data from each subject...")
        self.eeg_data, self.number_of_repetitions, self.number_of_subjects_loaded = self._load_eeg_data(
            dataset_dir, eeg_folder_list, eeg_file_name, device, subject
        )
        
        logger.info("Loading image metadata...")
        self.image_meta_data = self._load_image_meta_data(image_meta_data)
        num_data = len(self.image_meta_data[f'{self.data_type}_img_concepts'])
        self.samples_per_subject = num_data * self.number_of_repetitions

        logger.info("Loading image data...")
        if load_images:
            self.image_data = self._load_image(image_dir, transform, device)
        else:
            self.image_data = None
            logger.info("Skipping image pixel loading (load_images=False).")
        
        n_concepts = len(np.unique(self.image_meta_data[f'{self.data_type}_img_concepts']))
        
        logger.info(
            "ThingsEEGDataset\n"
            "Total samples : {}\n"
            "Number of subjects loaded : {}\n"
            "Samples per subject without repetition : {}\n"
            "Number of repetitions : {}\n"
            "Number of images : {}\n"
            "Number of samples per image : {}",
            self.number_of_subjects_loaded,
            len(self.eeg_data),
            num_data,
            self.number_of_repetitions,
            n_concepts,
            int(num_data / n_concepts),
        )
        
        logger.info("EEG data loaded successfully.")

    
    @staticmethod
    def _load_eeg_data(dataset_dir, folder_list: list[str], file_name: str, device: torch.device, subject: int) -> list[dict[str, Any]]:
        # Load EEG data from numpy file and convert to torch tensor
        eeg_data = []
        number_of_repetitions = 0
        number_of_subjects_loaded = 0
        
        for subject_folder in folder_list:
            subject_path = os.path.join(dataset_dir, subject_folder)
            if subject != -1 and subject_folder != f"sub-{subject:02d}":
                continue
            
            if not os.path.isdir(subject_path):
                continue
            
            # Load training and test EEG data for the subject
            path = os.path.join(subject_path, f"{file_name}.npy")
            
            if not os.path.isfile(path):
                logger.warning(f"EEG data file not found for subject {subject_folder}: {path}")
                continue
                
            data = np.load(path, allow_pickle=True).item()
            # Training image conditions × Training EEG repetitions × EEG channels × EEG time points
            # 16540, 4, 17, 100
            # Flattern the repetitions dimension into 16540*4, 17, 100
            number_of_repetitions = data['preprocessed_eeg_data'].shape[1]
            data['preprocessed_eeg_data'] = data['preprocessed_eeg_data'].reshape(-1, *data['preprocessed_eeg_data'].shape[2:])

            eeg_data.append(data['preprocessed_eeg_data'])
            number_of_subjects_loaded+=1

        # Concatenate all subjects' data into a single tensor
        if eeg_data:
            eeg_data = torch.from_numpy(np.concatenate(eeg_data, axis=0)).float().to(device)
        else:
            logger.warning("No EEG data found for any subject.")
            eeg_data = None
            
        if number_of_repetitions == 0:
            logger.warning("No EEG data found, setting number_of_repetitions to 0.")
        
        return eeg_data, number_of_repetitions, number_of_subjects_loaded
    
    @staticmethod
    def _load_image_meta_data(image_meta_data_path: str) -> dict[str, Any]:
        if not os.path.isfile(image_meta_data_path):
            raise FileNotFoundError(f"Image metadata file not found: {image_meta_data_path}")
        
        meta_data = np.load(image_meta_data_path, allow_pickle=True).item()
        return meta_data

    @staticmethod
    def _load_image(image_dir: str, transform: Callable[[Image.Image], Any] | None = None, device: torch.device = torch.device("cpu")) -> None:
        
        images = {}
        
        for conept in os.listdir(image_dir):
            concept_dir = os.path.join(image_dir, conept)
            if not os.path.isdir(concept_dir):
                continue
            
            for img_file in os.listdir(concept_dir):
                img_path = os.path.join(concept_dir, img_file)
                if not os.path.isfile(img_path):
                    continue
                
                image = Image.open(img_path).convert('RGB')
                
                if image is None:
                    logger.warning(f"Failed to load image: {img_path}")
                    continue
                image = transform(image) if transform else image
                
                if not isinstance(image, torch.Tensor):
                    image = tf.to_tensor(image).float().to(device)
                
                if conept not in images:
                    images[conept] = {}                        
                
                images[conept][img_file] = image
        
        return images


    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} is out of range for dataset of size {len(self)}.")
        if self.number_of_repetitions <= 0:
            raise ValueError("number_of_repetitions must be positive to decode the image index.")

        subject_local_index = index % self.samples_per_subject
        subject_index = index // self.samples_per_subject
        data_index = subject_local_index // self.number_of_repetitions
        repetition_index = subject_local_index % self.number_of_repetitions
        
        image_concept = self.image_meta_data[f'{self.data_type}_img_concepts'][data_index]
        image_file = self.image_meta_data[f'{self.data_type}_img_files'][data_index]


        return (
            self.eeg_data[index],
            self.image_data[image_concept][image_file] if self.image_data is not None else None,
            subject_index, # index of the subject in the loaded dataset, from 0 to number_of_subjects_loaded-1
            repetition_index, # index of the repetition for the same image and subject, from 0 to number_of_repetitions-1
            data_index, # original index in the whole dataset both eeg and image before flattening the repetitions dimension
            image_concept,
            image_file,
        )
        
    def __len__(self) -> int:
        return len(self.eeg_data)