# LucidChart Pipeline Spec — SUPAEEG

Each section maps to a stage label. For every node: what image/icon to use, exact inputs, exact outputs.

## Visual Conventions

- **Stage label = rectangular container/group box.** Each stage (Offline Pre-processing, Input Data, Training, Evaluation) is drawn as a bounding rectangle that wraps all nodes belonging to that stage. The stage name sits on the box border or inside the top edge.
- Within the **Training** container, the EEG branch and Image branch are two parallel sub-containers, both feeding into a Loss sub-container.
- Arrows crossing container boundaries are drawn from the last node of the source stage to the first node of the destination stage.

---

## STAGE LABEL: Offline Pre-processing  *(run once, no gradients)*

---

### Node 1 — Raw Images
**Visual:** Grid of thumbnail photos (varied objects — animals, tools, food) representing the THINGS dataset  
**Input:** THINGS image dataset on disk — `training_images/` and `test_images/` folders, organized by concept  
**Output:** PIL RGB images, each resized to 448×448 for InternViT preprocessing  
**Annotation:** ~22,248 training images · 200 test concepts

---

### Node 2 — InternViT-6B-448px-V1-5 Encoder  *(frozen — lock icon overlay)*
**Visual:** ViT architecture diagram (patch grid → transformer stack) with a padlock icon  
**Model:** `OpenGVLab/InternViT-6B-448px-V1-5` — 6B parameter vision transformer  
**Preprocessing:** `CLIPImageProcessor` (resize to 448×448, normalize)  
**Input:** Preprocessed pixel tensor `(B, 3, 448, 448)`  
**Output:** Hidden states from 5 tapped layers: [20, 24, 28, 32, 36]  
**Annotation:** 48 transformer layers total · all parameters frozen · no backprop ever

---

### Node 3 — Multi-Level Feature Extraction  *(forward hooks on layers 20 / 24 / 28 / 32 / 36)*
**Visual:** Vertical stack of 48 numbered transformer blocks; arrows tapping off at blocks 20, 24, 28, 32, 36  
**Input:** InternViT hidden states from Node 2  
**Output (per image):** 5 CLS-token vectors, each `(3200,)` — stacked to `(5, 3200)`  
**Annotation:** Layers sampled from early-mid to late to capture multi-level visual semantics

---

### Node 4 — Visual Feature Bank  *(lookup table)*
**Visual:** Database cylinder icon or table grid showing rows of `(concept, filename) → tensor(5, 3200)`  
**Input:** All `(5, 3200)` tensors from Node 3  
**Output:** `.pt` files in `data/things_eeg/image_feature/internvit_multilevel_20_24_28_32_36/`  
**Shape per image:** `(5, 3200)` — 5 layer activations × 3200 hidden dim  
**Annotation:** Loaded once at training start via `InternViTFeatureLookup` · never modified during training

---

## STAGE LABEL: Input Data

---

### Node 5 — Raw EEG Recordings
**Visual:** EEG cap/headset illustration (top-down scalp view) alongside a multi-channel waveform plot (17 coloured lines over time)  
**Input:** `.npy` preprocessed EEG files per subject — `data/things_eeg/sub-XX/preprocessed_eeg_training.npy`  
**Output:** Tensor `(16540 × 4 repetitions, 17 channels, 100 timepoints)` per subject  
**Annotation:** 10 subjects · 17 channels · 100 time pts (≈ 400 ms post-stimulus) · 4 repetitions per image

---

### Node 6 — ThingsEEGDataset / DataLoader
**Visual:** Stacked-papers icon or a "batching" funnel  
**Source:** `src/dataset.py — ThingsEEGDataset`  
**Input:** EEG tensor (Node 5) + image metadata `.npy` + concept/file index  
**Output per batch:**
- `eeg` — `(batch, 17, 100)` float tensor
- `image_concepts` — list of concept strings, length `batch`
- `image_files` — list of filename strings, length `batch`
- `concept_indices`, `image_indices` — integer index tensors  

**Annotation:** Intra-subject: 1 subject's data · Inter-subject (LOSO): 9 subjects concatenated

---

## STAGE LABEL: Training  *(gradients flow here)*

---

### EEG Branch

---

### Node 7 — EEGNetEncoder
**Visual:** MLP layer stack with residual skip arrow  
**Source:** `src/encoders/eegnet_encoder.py`  
**Input:** EEG tensor `(batch, 17, 100)`

| Layer | Operation | Output shape |
|---|---|---|
| Flatten | Reshape (17×100) | `(batch, 1700)` |
| Linear | Linear(1700 → 1024) | `(batch, 1024)` |
| ResidualAdd | GELU → Linear(1024→1024) → Dropout(0.3), add residual | `(batch, 1024)` |
| LayerNorm | Normalize over 1024-d | `(batch, 1024)` |

**Output:** `(batch, 1024)`

---

### Node 8 — eeg_projector
**Visual:** Single linear layer box  
**Source:** `src/models/supaeeg.py — self.eeg_projector`  
**Input:** `(batch, 1024)` from EEGNetEncoder  
**Architecture:** `Linear(1024 → 512)`  
**Output:** `(batch, 512)`

---

### Node 9 — share_encoder  *(shared — EEG and image branches both pass through this)*
**Visual:** Single linear layer box with a "shared / chain-link" icon  
**Source:** `src/models/supaeeg.py — self.share_encoder`  
**Input:** `(batch, 512)`  
**Architecture:** `Linear(512 → 512)`  
**Output:** `(batch, 512)`  
**Annotation:** Same weights used for both EEG and image embeddings — forces a unified space

---

### Node 10 — L2-normalize → zE
**Visual:** Vector bar with ℓ2-norm symbol  
**Input:** `(batch, 512)` from share_encoder  
**Operation:** `F.normalize(x, dim=-1)`  
**Output:** `zE (batch, 512)` — unit-norm EEG embedding

---

### Image Branch  *(no gradient — features from frozen bank)*

---

### Node 11 — InternViTFeatureLookup
**Visual:** Database cylinder (same icon as Node 4) with a "lookup" arrow; padlock to indicate frozen  
**Source:** `src/encoders/vision_encoder.py — InternViTFeatureLookup.retrieve_batch()`  
**Input:** `image_concepts` + `image_files` lists from the batch (Node 6)  
**Output:** `image_layers (batch, 5, 3200)` — 5-layer features for each batch image  
**Annotation:** No gradient flows through this node

---

### Node 12 — Mean Pooling (layers dim)
**Visual:** Five stacked rectangles collapsing into one with a "μ" symbol  
**Input:** `(batch, 5, 3200)`  
**Operation:** `.mean(dim=1)` — average across the 5 InternViT layer activations  
**Output:** `(batch, 3200)`

---

### Node 13 — img_pre_projector
**Visual:** Single linear layer box  
**Source:** `src/models/supaeeg.py — self.img_pre_projector`  
**Input:** `(batch, 3200)`  
**Architecture:** `Linear(3200 → 1024)`  
**Output:** `(batch, 1024)`

---

### Node 14 — img_projector
**Visual:** Single linear layer box  
**Source:** `src/models/supaeeg.py — self.img_projector`  
**Input:** `(batch, 1024)`  
**Architecture:** `Linear(1024 → 512)`  
**Output:** `(batch, 512)`

---

### Node 15 — share_encoder  *(same weights as Node 9)*
**Visual:** Single linear layer box with chain-link icon (mirror of Node 9)  
**Input:** `(batch, 512)` from img_projector  
**Architecture:** `Linear(512 → 512)` — same weights as Node 9  
**Output:** `(batch, 512)`

---

### Node 16 — L2-normalize → zI
**Visual:** Vector bar with ℓ2-norm symbol  
**Input:** `(batch, 512)` from share_encoder  
**Operation:** `F.normalize(x, dim=-1)`  
**Output:** `zI (batch, 512)` — unit-norm image embedding

---

### Node 17 — Training Loss  *(two terms, two-stage schedule)*
**Visual:** Two sub-boxes feeding into a summation (Σ) node

**17a — Symmetric InfoNCE**  
- Visual: N×N similarity heatmap with bright diagonal (correct pairs)  
- Input: `zE (batch, 512)` and `zI (batch, 512)`  
- Operation: cosine similarity matrix → cross-entropy on both directions (τ = 0.07)  
- Active: Stage 1 and Stage 2  
- Output: Scalar `ℒ_InfoNCE`

**17b — MMD Loss**  
- Visual: Two distribution curves being compared  
- Input: `zE (batch, 512)` and `zI (batch, 512)`  
- Operation: Maximum Mean Discrepancy between EEG and image embedding distributions  
- Active: Stage 2 only  
- Output: Scalar `ℒ_MMD`

**Combined:**  
- Stage 1: `ℒ_total = ℒ_InfoNCE`  
- Stage 2: `ℒ_total = ℒ_InfoNCE + λ · ℒ_MMD`

---

## STAGE LABEL: Evaluation  *(test time — no gradients)*

---

### Node 18 — EEG Query Embedding
**Visual:** EEG waveform icon → same encoder stack (Nodes 7–10) → single vector  
**Input:** Test EEG `(batch, 17, 100)`  
**Process:** Same EEGNetEncoder → eeg_projector → share_encoder → L2-normalize (Nodes 7–10, inference mode)  
**Output:** `zE (batch, 512)`

---

### Node 19 — Image Gallery  *(pre-computed from Feature Bank)*
**Visual:** Grid of 200 concept thumbnails with 512-d embedding bars below each  
**Input:** InternViT features from Feature Bank for all 200 test concepts → image branch (Nodes 11–16)  
**Output:** Gallery matrix `(200, 512)` — L2-normalized embeddings  
**Annotation:** Fixed gallery — same 200 concepts for every query

---

### Node 20 — Cosine Similarity & Ranking
**Visual:** N×200 similarity matrix heatmap (queries on Y-axis, gallery on X-axis)  
**Input:** EEG query embeddings `(N_queries, 512)` + Gallery `(200, 512)`  
**Output:** Ranked list per query — 200 images sorted by cosine similarity descending

---

### Node 21 — Retrieval Accuracy
**Visual:** Horizontal bar chart or two bold numbers: Top-1 % and Top-5 %  
**Input:** Ranked lists from Node 20  
**Output:** `Top-1 Accuracy` and `Top-5 Accuracy` scalars (% of queries where correct image ranks 1st or within top 5)  
**Annotation:** Reported separately for Intra-subject and Inter-subject (LOSO) protocols

---

## Protocol Bracket  *(shown as a label/divider, not a processing node)*

| Protocol | Training data | Test data |
|---|---|---|
| **Intra-subject** | Subject k train split | Subject k test split |
| **Inter-subject (LOSO)** | All subjects except k | Subject k test split |

Wrap the entire Training + Evaluation stages in a bracket labelled with the active protocol.
