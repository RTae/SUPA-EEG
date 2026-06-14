"""
Experiment 7: Wrong Prediction Analysis

Produces 4 figures from pre-computed embeddings (no model needed):
  reports/error_analysis_hard_easy.png   — per-concept Top-1 accuracy ranking
  reports/error_analysis_semantic_sim.png — semantic similarity of errors vs random baseline
  reports/error_analysis_confusion.png   — confusion heatmap for the 20 hardest concepts
  reports/error_analysis_failures.png    — GT + top-5 retrieved for worst-performing concepts

Usage:
    python scripts/analyze_errors.py
    python scripts/analyze_errors.py --subject 1 --n_failures 8 --test_img_dir data/things_eeg/test_images
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ── helpers (mirrors visualize_inter_retrieval.py) ────────────────────────────

def load_pil(img_dir: str, concept: str, filename: str) -> Image.Image:
    return Image.open(os.path.join(img_dir, concept, filename)).convert("RGB")


def set_border(ax, color: str, lw: float = 4.0) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)


# ── core ──────────────────────────────────────────────────────────────────────

def run(subject: int, n_failures: int, embeddings_dir: str, test_img_dir: str) -> None:
    # Load
    eeg_emb  = np.load(os.path.join(embeddings_dir, f"eeg_embeddings_sub{subject:02d}.npy"))
    img_emb  = np.load(os.path.join(embeddings_dir, "image_embeddings.npy"))
    concepts = list(np.load(os.path.join(embeddings_dir, "concept_order.npy")))
    files    = list(np.load(os.path.join(embeddings_dir, "concept_files.npy")))
    n        = len(concepts)

    print(f"EEG: {eeg_emb.shape}  IMG: {img_emb.shape}  Concepts: {n}")

    # Similarity matrix and per-concept stats
    sim      = eeg_emb @ img_emb.T                         # (N, N)
    top5_idx = np.argsort(-sim, axis=1)[:, :5]             # (N, 5)
    correct  = (top5_idx == np.arange(n)[:, None])         # (N, 5) bool
    top1_acc = correct[:, 0].astype(float)                 # 0 or 1 per concept
    top5_acc = correct.any(axis=1).astype(float)

    print(f"Overall  Top-1: {top1_acc.mean():.1%}  Top-5: {top5_acc.mean():.1%}")

    os.makedirs("reports", exist_ok=True)

    # ── A: Hard vs Easy ───────────────────────────────────────────────────────
    order      = np.argsort(top1_acc)
    n_show     = min(20, n // 2)
    hard_idx   = order[:n_show]
    easy_idx   = order[-n_show:][::-1]
    both_idx   = np.concatenate([hard_idx, easy_idx])
    both_names = [concepts[i].split("_", 1)[1] if "_" in concepts[i] else concepts[i]
                  for i in both_idx]
    both_acc   = top1_acc[both_idx]
    colors     = ["#d62728"] * n_show + ["#2ca02c"] * n_show

    fig, ax = plt.subplots(figsize=(7, n_show * 0.45 * 2 + 1))
    bars = ax.barh(range(len(both_idx)), both_acc, color=colors)
    ax.set_yticks(range(len(both_idx)))
    ax.set_yticklabels(both_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Top-1 Accuracy")
    ax.set_title(f"Hard (red) vs Easy (green) Concepts — Sub{subject:02d}")
    ax.axvline(top1_acc.mean(), color="grey", linestyle="--", linewidth=1, label="mean")
    ax.legend(fontsize=8)
    ax.axhline(n_show - 0.5, color="black", linewidth=0.8, linestyle=":")
    plt.tight_layout()
    out_a = "reports/error_analysis_hard_easy.png"
    plt.savefig(out_a, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_a}")

    # ── B: Semantic similarity of errors ─────────────────────────────────────
    wrong_mask = top1_acc == 0
    if wrong_mask.sum() == 0:
        print("No wrong predictions — skipping semantic similarity plot.")
    else:
        wrong_true_idx = np.where(wrong_mask)[0]
        wrong_pred_idx = top5_idx[wrong_true_idx, 0]
        error_sim = (img_emb[wrong_pred_idx] * img_emb[wrong_true_idx]).sum(axis=1)

        rng = np.random.default_rng(42)
        rand_a = rng.integers(0, n, size=2000)
        rand_b = rng.integers(0, n, size=2000)
        baseline_sim = (img_emb[rand_a] * img_emb[rand_b]).sum(axis=1)

        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(-0.1, 1.0, 30)
        ax.hist(baseline_sim, bins=bins, alpha=0.5, label="Random pairs", color="steelblue", density=True)
        ax.hist(error_sim,    bins=bins, alpha=0.7, label="Error pairs",  color="tomato",   density=True)
        ax.set_xlabel("Image-to-image cosine similarity")
        ax.set_ylabel("Density")
        ax.set_title(f"Semantic Similarity of Errors vs Random — Sub{subject:02d}")
        ax.legend()
        plt.tight_layout()
        out_b = "reports/error_analysis_semantic_sim.png"
        plt.savefig(out_b, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_b}")
        print(f"  Mean error sim: {error_sim.mean():.3f}  |  Mean random sim: {baseline_sim.mean():.3f}")

    # ── C: Confusion heatmap (20 hardest concepts) ────────────────────────────
    n_conf      = min(20, n)
    hard20_idx  = order[:n_conf]
    sub_sim     = sim[np.ix_(hard20_idx, hard20_idx)]
    sub_names   = [concepts[i].split("_", 1)[1] if "_" in concepts[i] else concepts[i]
                   for i in hard20_idx]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(sub_sim, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n_conf)); ax.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_conf)); ax.set_yticklabels(sub_names, fontsize=7)
    ax.set_title(f"EEG→Image Similarity — 20 Hardest Concepts (Sub{subject:02d})")
    plt.tight_layout()
    out_c = "reports/error_analysis_confusion.png"
    plt.savefig(out_c, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_c}")

    # ── D: Failure case visualization ─────────────────────────────────────────
    # Pick concepts where Top-1 is wrong; rank by max top-5 similarity (hardest first)
    wrong_concepts = np.where(top1_acc == 0)[0]
    if len(wrong_concepts) == 0:
        print("No wrong predictions — skipping failure visualization.")
        return

    # Sort wrong concepts by their best similarity score (ascending = hardest)
    best_sim_among_wrong = sim[wrong_concepts, wrong_concepts]
    fail_order  = wrong_concepts[np.argsort(best_sim_among_wrong)]
    fail_idx    = fail_order[:n_failures]

    n_rows = len(fail_idx)
    n_cols = 6  # GT + Top 1-5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.8, n_rows * 1.8), squeeze=False)

    col_labels = ["GT", "Top 1", "Top 2", "Top 3", "Top 4", "Top 5"]
    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=9, fontweight="bold", pad=4)

    for row_i, concept_i in enumerate(fail_idx):
        concept   = concepts[concept_i]
        gt_file   = files[concept_i]
        t5_conc   = [concepts[j] for j in top5_idx[concept_i]]
        t5_files  = [files[j]    for j in top5_idx[concept_i]]
        t5_corr   = [j == concept_i for j in top5_idx[concept_i]]

        # GT
        ax = axes[row_i, 0]
        try:
            ax.imshow(load_pil(test_img_dir, concept, gt_file))
        except FileNotFoundError:
            ax.text(0.5, 0.5, concept, ha="center", va="center", fontsize=6, wrap=True)
        ax.set_xticks([]); ax.set_yticks([])
        set_border(ax, "red")
        short = concept.split("_", 1)[1] if "_" in concept else concept
        ax.set_ylabel(short, fontsize=6, rotation=0, labelpad=40, va="center")

        # Top-5 retrievals
        for col_i, (tc, tf, ok) in enumerate(zip(t5_conc, t5_files, t5_corr)):
            ax = axes[row_i, col_i + 1]
            try:
                ax.imshow(load_pil(test_img_dir, tc, tf))
            except FileNotFoundError:
                ax.text(0.5, 0.5, tc, ha="center", va="center", fontsize=6, wrap=True)
            ax.set_xticks([]); ax.set_yticks([])
            set_border(ax, "green" if ok else "none")

    plt.suptitle(f"Failure Cases — {n_rows} Worst Concepts (Sub{subject:02d})", fontsize=11, y=1.002)
    plt.tight_layout(pad=0.3)
    out_d = "reports/error_analysis_failures.png"
    plt.savefig(out_d, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_d}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",        type=int, default=1)
    parser.add_argument("--n_failures",     type=int, default=8)
    parser.add_argument("--embeddings_dir", type=str, default="reports/embeddings")
    parser.add_argument("--test_img_dir",   type=str, default="data/things_eeg/test_images")
    args = parser.parse_args()
    run(args.subject, args.n_failures, args.embeddings_dir, args.test_img_dir)
