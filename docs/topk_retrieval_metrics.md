# Top-1 / Top-5 Retrieval Metrics

## Overview

Top-1 and Top-5 retrieval accuracy are the standard evaluation metrics for zero-shot
EEG-to-image decoding. They measure how often the correct image concept ranks first
(Top-1) or within the top five (Top-5) when an EEG query is matched against a gallery
of candidate embeddings.

These utilities are implemented in `src/trainer/metrics.py` and align with the
methodology in the SAMGA paper (Subject-Aware Multi-Granularity Alignment for
Zero-Shot EEG-to-Image Retrieval):

- Paper: https://arxiv.org/pdf/2601.21948
- Reference implementation: https://anonymous.4open.science/r/repo-7f41a8

---

## Two Active Protocols

Two distinct retrieval protocols are in use.

### 1. EEG-to-image retrieval (SUPAEEG)

Used in SUPAEEG evaluation via `utilities.evaluate()`. The similarity matrix is **square**:

```
sim_matrix: (N_concepts, N_concepts)
```

Row `i` is the mean ℓ2-normalised EEG embedding for concept `i`, averaged over all
test trials. Column `i` is the paired CLIP image embedding for that concept (concatenated
S1/S2/S3 features, ℓ2-normalised). Correctness is the **diagonal convention**: entry
`(i, i)` is the correct EEG–image pair.

```
utilities.evaluate(model, test_loader, feature_lookup, device)
    └── model.embed(eeg)                # (batch, 2304) ℓ2-normalised
    └── mean per concept                # (N_concepts, 2304) eeg_features
    └── feature_lookup.retrieve(...)    # (N_concepts, 2304) image_features (CLIP S1+S2+S3)
    └── retrieve_all(eeg, image)
            └── sk_cosine_similarity    # → (N_concepts, N_concepts) square matrix
            └── retrieve_topk(sim, 5)  # diagonal convention
            └── returns (top5_count, top1_count, total)
    └── returns (top1, top5)            # fractions in [0, 1]
```

The CLIP feature gallery (`data/vision_encoder/clip/visual_features_clip.pt`) contains
16,740 entries covering all 1,654 training concepts and all 200 test concepts.

### 2. Prototype/class-based retrieval (SemanticModel)

Used in `evaluate_semantic_embeddings()` for the object-classification `SemanticModel`.
The similarity matrix is **rectangular**:

```
sim_matrix: (N_trials, N_classes)
```

Each row is one EEG trial. Each column is one class prototype — the mean ℓ2-normalised
EEG embedding for that class, computed fresh from the current eval set. Correctness for
row `i` is `labels[i]`, the column index of the correct class.

```
evaluate_semantic_embeddings(model, eval_loader, device)
    └── _collect_semantic_embeddings    # (N_trials × D, N_trials)
    └── prototype construction          # mean ℓ2-norm per class → (N_classes × D)
    └── sk_cosine_similarity            # → (N_trials, N_classes)
    └── _label_retrieval_counts         # → (count_5, count_1)
    └── returns (top1, top5)            # fractions in [0, 1]
```

For WT split, fine granularity (40 classes, 20 test trials per class):
`sim_matrix.shape == (800, 40)`.

---

## API Reference

### `retrieve_all`

```python
top5_count, top1_count, total = retrieve_all(eeg_features, image_features)

top1 = top1_count / total
top5 = top5_count / total
```

- `eeg_features`: `(N, D)` ℓ2-normalised, one embedding per concept (mean over trials)
- `image_features`: `(N, D)` ℓ2-normalised, same concept order as `eeg_features`
- Returns **raw counts**, not fractions

### `retrieve_topk`

```python
from src.trainer.metrics import retrieve_topk
from sklearn.metrics.pairwise import cosine_similarity

sim = cosine_similarity(eeg_features, image_features)  # (N, N) square
top5_count, top1_count = retrieve_topk(sim, k=5)
```

Requires a **square** matrix. Raises `ValueError` if the matrix is not square.

### `evaluate_semantic_embeddings`

```python
top1, top5 = evaluate_semantic_embeddings(model, eval_loader, device)
# top1, top5 ∈ [0.0, 1.0]
```

No image gallery required — prototypes are derived directly from the eval-set EEG
embeddings. Returns normalized fractions.

---

## How Rankings Are Computed

Both protocols use the same double-argsort ranking:

```python
sorted_indices = np.argsort(-sim_matrix, axis=1)   # descending similarity order
rankings = np.argsort(sorted_indices, axis=1)       # rank of each column for each row
```

For protocol 1 (diagonal): `correct_rank[i] = rankings[i, i] + 1`  
For protocol 2 (label-based): `correct_rank[i] = rankings[i, labels[i]] + 1`

```
top1 = (correct_ranks == 1).sum() / N
top5 = (correct_ranks <= 5).sum() / N
```
 