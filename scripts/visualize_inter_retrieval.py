"""
Generate a top-5 retrieval visualization grid for inter-subject (LOSO) evaluation.

Each row: GT image (red border) + Top-1 through Top-5 retrieved images.
Correct retrievals are highlighted with a green border.

Usage:
    python scripts/visualize_inter_retrieval.py
    python scripts/visualize_inter_retrieval.py --subject 7 --n_concepts 10 --seed 42
    python scripts/visualize_inter_retrieval.py --subject 1 --n_concepts 15 --output reports/inter_vis_sub01.png
"""

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_pil(img_dir: str, concept: str, filename: str) -> Image.Image:
    path = os.path.join(img_dir, concept, filename)
    return Image.open(path).convert("RGB")


def set_border(ax, color: str, lw: float = 4.0) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)


def run(subject: int, n_concepts: int, seed: int, output_path: str,
        embeddings_dir: str, test_img_dir: str) -> None:
    print(f"Subject: {subject:02d} | Concepts: {n_concepts}")

    # Load pre-computed embeddings from server
    eeg_emb = np.load(os.path.join(embeddings_dir, f"eeg_embeddings_sub{subject:02d}.npy"))
    img_emb = np.load(os.path.join(embeddings_dir, "image_embeddings.npy"))
    all_concepts = list(np.load(os.path.join(embeddings_dir, "concept_order.npy")))
    all_files = list(np.load(os.path.join(embeddings_dir, "concept_files.npy")))
    concept_to_file = dict(zip(all_concepts, all_files))

    print(f"EEG embeddings: {eeg_emb.shape} | Image embeddings: {img_emb.shape}")

    # Randomly select concepts to visualize
    rng = random.Random(seed)
    selected = rng.sample(all_concepts, n_concepts)

    rows = []
    for concept in selected:
        concept_idx = all_concepts.index(concept)
        zE = eeg_emb[concept_idx:concept_idx+1]  # (1, 512)

        sim = (zE @ img_emb.T)[0]  # (200,)
        top5_idx = np.argsort(-sim)[:5]

        top5_concepts = [all_concepts[i] for i in top5_idx]
        top5_files = [concept_to_file[c] for c in top5_concepts]
        top5_correct = [c == concept for c in top5_concepts]

        rows.append(
            dict(
                concept=concept,
                gt_file=concept_to_file[concept],
                top5_concepts=top5_concepts,
                top5_files=top5_files,
                top5_correct=top5_correct,
            )
        )

    # Plot
    n_rows = len(rows)
    n_cols = 6  # GT + Top 1-5
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 1.8, n_rows * 1.8), squeeze=False
    )

    col_labels = ["GT", "Top 1", "Top 2", "Top 3", "Top 4", "Top 5"]
    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=10, fontweight="bold", pad=6)

    for row_idx, row in enumerate(rows):
        # GT column
        gt_img = load_pil(test_img_dir, row["concept"], row["gt_file"])
        ax = axes[row_idx, 0]
        ax.imshow(gt_img)
        ax.set_xticks([])
        ax.set_yticks([])
        set_border(ax, color="red", lw=4)

        # Top-1 to Top-5
        for col_idx, (concept, fname, correct) in enumerate(
            zip(row["top5_concepts"], row["top5_files"], row["top5_correct"])
        ):
            img = load_pil(test_img_dir, concept, fname)
            ax = axes[row_idx, col_idx + 1]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if correct:
                set_border(ax, color="green", lw=4)
            else:
                set_border(ax, color="none")

    plt.suptitle(
        f"Inter-Subject (LOSO) Top-5 Retrieval — Subject {subject:02d}",
        fontsize=12,
        y=1.005,
    )
    plt.tight_layout(pad=0.3)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=7)
    parser.add_argument("--n_concepts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="reports/inter_retrieval_vis.png")
    parser.add_argument("--embeddings_dir", type=str, default="reports/embeddings",
                        help="Directory with pre-computed .npy files from save_embeddings.py")
    parser.add_argument("--test_img_dir", type=str, default="data/things_eeg/test_images")
    args = parser.parse_args()
    run(args.subject, args.n_concepts, args.seed, args.output,
        args.embeddings_dir, args.test_img_dir)
