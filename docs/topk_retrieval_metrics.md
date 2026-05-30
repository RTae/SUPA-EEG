# Top-1 / Top-5 Retrieval Metrics

## Overview

Top-1 and Top-5 retrieval accuracy are the standard evaluation metrics for zero-shot
EEG-to-image decoding. They measure how often the correct image concept ranks first
(Top-1) or within the top five (Top-5) when an EEG query is matched against a gallery
of candidate image embeddings.

These utilities were added to align our evaluation with the methodology in the SAMGA
paper (Subject-Aware Multi-Granularity Alignment for Zero-Shot EEG-to-Image Retrieval):

- Paper: https://arxiv.org/pdf/2601.21948
- Reference implementation: https://anonymous.4open.science/r/repo-7f41a8

The retrieval utilities are implemented in:
```
src/trainer/metrics.py
```
Two distinct retrieval protocols exist in the codebase. 

### Prototype/class-based retrieval (current)

Used today for the `SemanticModel`. The similarity matrix is **rectangular**:

```
sim_matrix: (N_trials, N_classes)
```

Each row is one EEG trial. Each column is one class prototype (the mean L2-normalized
EEG embedding for that class). Correctness for row `i` is determined by `labels[i]` —
the column index of the correct class.

### EEG-to-image retrieval (future)

Used in the SAMGA paper. The similarity matrix is **square**:

```
sim_matrix: (N_concepts, N_concepts)
```

Each row `i` is an EEG concept embedding. Column `i` is that concept's image embedding
(e.g., from a CLIP vision encoder). The correct retrieval for query `i` is always
column `i` — the **diagonal convention**.

This requires a separate gallery of image embeddings aligned to the EEG embeddings
by concept index.

## Current Pipeline

```
evaluate_semantic_embeddings
    └── _collect_semantic_embeddings   # runs model, returns (N_trials × D, N_trials)
    └── prototype construction         # mean L2-norm per class → (N_classes × D)
    └── sk_cosine_similarity           # → (N_trials, N_classes)
    └── _label_retrieval_counts        # → (count_5, count_1)
    └── returns (top1, top5, val_loss) # normalized fractions
```

### Prototype construction

For each class present in the eval set, all trial embeddings for that class are
averaged and L2-normalized to produce a single prototype vector. Prototypes are
computed fresh each eval call from the model's current outputs.

### Similarity matrix

```
sim_matrix[i, j] = cosine_similarity(eeg_trial_i, prototype_j)
shape: (N_trials, N_classes)
```

For WT split, fine granularity (40 classes, 20 test trials per class):
`sim_matrix.shape == (800, 40)`.

### How Top-1 / Top-5 are computed

For each trial `i`, the correct class is `mapped_labels[i]` — the column index of its
prototype in `sim_matrix`. Rankings are derived via double argsort (descending), and
`correct_ranks[i]` is the 1-indexed rank of the correct class for that trial.

```
top1 = (correct_ranks == 1).sum() / N_trials
top5 = (correct_ranks <= 5).sum() / N_trials
```

Both metrics are returned as **normalized fractions** in `[0.0, 1.0]`.

### Returned signature

```python
top1, top5, val_loss = evaluate_semantic_embeddings(model, eval_loader, device, triplet_margin)
# top1, top5 ∈ [0.0, 1.0]
# val_loss: batch-hard triplet loss on the eval set
```

---

## Future Retrieval

Once per-concept image embeddings (e.g., from CLIP) are available, use `retrieve_all`
and `retrieve_topk` for the paper-exact evaluation protocol.

```
retrieve_all(eeg_embeddings, image_embeddings)
    └── sk_cosine_similarity           # → (N_concepts, N_concepts) square matrix
    └── retrieve_topk(sim, k=5)        # diagonal convention
    └── returns (top5_count, top1_count, total)
```

### Requirements

- `eeg_embeddings`: `(N, D)` array — one L2-normalized EEG embedding per concept,
  averaged over test trials.
- `image_embeddings`: `(N, D)` array — one L2-normalized image embedding per concept
  (e.g., CLIP ViT mean over candidate images), in the **same concept order**.
- Both arrays must be aligned: `eeg_embeddings[i]` and `image_embeddings[i]` must
  correspond to the same concept.

### Diagonal convention

`sim_matrix[i, i]` is the similarity between the EEG query for concept `i` and the
correct image for concept `i`. All other entries in row `i` are distractors.

### Return values

`retrieve_all` returns **raw counts**, not fractions, matching the paper's API:

```python
top5_count, top1_count, total = retrieve_all(
    eeg_embeddings,
    image_embeddings
)

top1 = top1_count / total
top5 = top5_count / total
```

To call `retrieve_topk` directly with a precomputed similarity matrix:

```python
from trainer.metrics import retrieve_topk
from sklearn.metrics.pairwise import cosine_similarity

sim = cosine_similarity(eeg_embeddings, image_embeddings)  # (N, N) square
top5_count, top1_count = retrieve_topk(sim, k=5)
```
---
## Limitations
Current evaluation is **not yet full cross-modal EEG-to-image retrieval** since:
- no CLIP/image embedding gallery exists yet,
- and prototypes are computed directly from EEG evaluation embeddings. 

Current scores should be interpreted as class retrieval accuracy rather than paper-exact EEG-to-image retrieval performance. 