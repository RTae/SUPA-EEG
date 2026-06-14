"""
Run this on the training server to pre-compute and save EEG + image embeddings.
Outputs three small files that can be transferred locally to generate the retrieval figure.

Usage:
    python scripts/save_embeddings.py --subject 7
    python scripts/save_embeddings.py --subject 7 --checkpoint outputs/2026-06-06/inter/supaeeg_loso_sub07.pt
    python scripts/save_embeddings.py --subject 7 --n_concepts 10 --seed 42  # subset for visualization

Outputs (in reports/embeddings/):
    eeg_embeddings_sub{N}.npy    shape (200, 512)  — averaged EEG embedding per test concept
    image_embeddings.npy         shape (200, 512)  — image gallery embeddings
    concept_order.npy            shape (200,)      — concept names in order
    concept_files.npy            shape (200,)      — image filenames in order
"""

import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import ThingsEEGDataset
from src.encoders.vision_encoder import InternViTFeatureLookup
from src.utilities import Config, make_model


def load_config() -> Config:
    cfg = OmegaConf.load("conf/config.yaml")
    config = Config()
    for field_name in config.__dataclass_fields__:
        if hasattr(cfg, field_name):
            setattr(config, field_name, getattr(cfg, field_name))
    return config


def run(subject: int, checkpoint_path: str, out_dir: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Subject: {subject:02d}")

    config = load_config()
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model = make_model(config, device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded: {checkpoint_path}")

    # Test dataset
    dataset = ThingsEEGDataset(
        dataset_dir=config.dataset_dir,
        data_type="test",
        subject=subject,
        load_images=False,
        data_average=False,
    )

    all_concepts = sorted(list(set(dataset.image_meta_data["test_img_concepts"])))
    concept_to_file = {}
    for c, f in zip(
        dataset.image_meta_data["test_img_concepts"],
        dataset.image_meta_data["test_img_files"],
    ):
        if c not in concept_to_file:
            concept_to_file[c] = f

    n_reps = dataset.number_of_repetitions
    n_concepts = len(all_concepts)

    # EEG embeddings — average over repetitions per concept
    print(f"Computing EEG embeddings for {n_concepts} concepts ({n_reps} reps each)...")
    eeg_embeddings = np.zeros((n_concepts, 512), dtype=np.float32)
    for i, concept in enumerate(all_concepts):
        indices = [i * n_reps + r for r in range(n_reps)]
        eeg_list = [dataset[idx][0] for idx in indices]
        eeg_batch = torch.stack(eeg_list).to(device)
        with torch.no_grad():
            zE = model.embed(eeg_batch)
            zE = torch.nn.functional.normalize(zE.mean(dim=0, keepdim=True), dim=1)
        eeg_embeddings[i] = zE.cpu().numpy()
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_concepts}")

    # Image gallery embeddings
    print("Computing image gallery embeddings...")
    feature_path = os.path.join(config.internvit_dir, "internvit_features.npy")
    lookup = InternViTFeatureLookup(feature_path=feature_path)
    gallery_files = [concept_to_file[c] for c in all_concepts]
    gallery_features = lookup.retrieve_batch(all_concepts, gallery_files)
    with torch.no_grad():
        zI = model.encode_image(gallery_features.to(device), subject_ids=None)
    image_embeddings = zI.cpu().numpy()

    # Save
    eeg_path = os.path.join(out_dir, f"eeg_embeddings_sub{subject:02d}.npy")
    img_path = os.path.join(out_dir, "image_embeddings.npy")
    concepts_path = os.path.join(out_dir, "concept_order.npy")
    files_path = os.path.join(out_dir, "concept_files.npy")

    np.save(eeg_path, eeg_embeddings)
    np.save(img_path, image_embeddings)
    np.save(concepts_path, np.array(all_concepts))
    np.save(files_path, np.array(gallery_files))

    print(f"\nSaved:")
    print(f"  {eeg_path}  {eeg_embeddings.shape}")
    print(f"  {img_path}  {image_embeddings.shape}")
    print(f"  {concepts_path}")
    print(f"  {files_path}")
    print(f"\nTransfer these 4 files to your local machine and run:")
    print(f"  python scripts/visualize_inter_retrieval.py --subject {subject} --embeddings_dir {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=7)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (default: outputs/2026-06-06/inter/supaeeg_loso_sub{N:02d}.pt)",
    )
    parser.add_argument("--out_dir", type=str, default="reports/embeddings")
    args = parser.parse_args()

    ckpt = args.checkpoint or f"outputs/2026-06-06/inter/supaeeg_loso_sub{args.subject:02d}.pt"
    run(args.subject, ckpt, args.out_dir)
