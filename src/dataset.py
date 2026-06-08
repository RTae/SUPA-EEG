import os
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
from loguru import logger
import numpy as np

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
        data_average: bool = False,
        eeg_t_start: float = -0.2,
        eeg_t_end: float = 1.0,
        eeg_suffix: str = "",
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
            dataset_dir, eeg_folder_list, eeg_file_name, device, subject,
            data_average=data_average,
            eeg_t_start=eeg_t_start,
            eeg_t_end=eeg_t_end,
            eeg_suffix=eeg_suffix,
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
    def _load_eeg_data(dataset_dir, folder_list: list[str], file_name: str, device: torch.device, subject: int, data_average: bool = False, eeg_t_start: float = -0.2, eeg_t_end: float = 1.0, eeg_suffix: str = "") -> list[dict[str, Any]]:
        # Load EEG data from numpy file and convert to torch tensor
        eeg_data = []
        number_of_repetitions = 0
        number_of_subjects_loaded = 0
        
        import re as _re
        folder_pattern = _re.compile(r"sub-\d{2}" + _re.escape(eeg_suffix) + r"$")
        for subject_folder in folder_list:
            subject_path = os.path.join(dataset_dir, subject_folder)
            # Match folder by suffix: "" -> sub-XX, "_63" -> sub-XX_63
            if not folder_pattern.match(subject_folder):
                continue
            if subject != -1 and subject_folder != f"sub-{subject:02d}{eeg_suffix}":
                continue
            
            if not os.path.isdir(subject_path):
                continue
            
            # Load training and test EEG data for the subject
            path = os.path.join(subject_path, f"{file_name}.npy")
            
            if not os.path.isfile(path):
                logger.warning(f"EEG data file not found for subject {subject_folder}: {path}")
                continue
                
            data = np.load(path, allow_pickle=True).item()
            raw = data['preprocessed_eeg_data']

            # Drop stimulus/trigger channel if present (last channel named 'stim')
            ch_names = list(data.get('ch_names', []))
            if ch_names and ch_names[-1].lower() == 'stim':
                raw = raw[..., :-1, :]   # drop last channel: (N, reps, 64, T) -> (N, reps, 63, T)

            # Crop to post-stimulus window using the stored time axis
            times = data['times']   # shape (n_timepoints,) in seconds
            t_start_idx = int(np.searchsorted(times, eeg_t_start))
            t_end_idx   = int(np.searchsorted(times, eeg_t_end))
            raw = raw[..., t_start_idx:t_end_idx]

            number_of_repetitions = raw.shape[1]

            if data_average:
                # average over repetitions dimension
                # (n_images, n_reps, 17, 100) → mean → (n_images, 17, 100)
                processed = raw.mean(axis=1)
                logger.info(
                    f"{subject_folder}: averaged {number_of_repetitions} reps "
                    f"shape {raw.shape} → {processed.shape}"
                )
            else:
                # current behaviour: flatten reps into samples
                # (n_images, n_reps, 17, 100) → reshape → (n_images*n_reps, 17, 100)
                processed = raw.reshape(-1, *raw.shape[2:])

            eeg_data.append(processed)
            number_of_subjects_loaded+=1

        # Concatenate all subjects' data into a single tensor
        if eeg_data:
            eeg_data = torch.from_numpy(np.concatenate(eeg_data, axis=0)).float().to(device)
        else:
            logger.warning("No EEG data found for any subject.")
            eeg_data = None
            
        if number_of_repetitions == 0:
            logger.warning("No EEG data found, setting number_of_repetitions to 0.")

        # when averaged: set number_of_repetitions to 1
        # so __getitem__ index arithmetic still works correctly
        if data_average:
            number_of_repetitions = 1

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