# SUPAEEG Pipeline Diagrams

Paste each code block into [mermaid.live](https://mermaid.live) to preview, then recreate in LucidChart.

**Two diagrams:**
- **Diagram 1** — high-level overview (LR, ~12 boxes). Good for a slide or README.
- **Diagram 2** — detailed LucidChart realization (TB overall, LR within Training). Matches the 21-node spec in `lucidchart_spec.md`; use this as the basis for the actual LucidChart build.

---

## Diagram 1 — High-Level Pipeline Overview

```mermaid
flowchart LR
    subgraph DATA["THINGS-EEG2 Dataset"]
        direction TB
        EEG["EEG Recordings\n17 channels × 100 time pts\n10 subjects"]
        IMGS["Images\n~22,248 training\n200 test concepts"]
    end

    subgraph OFFLINE["Offline Visual Feature Extraction  (run once, frozen)"]
        direction TB
        INTERN["InternViT-6B-448px-V1-5\n(frozen — never updated)"]
        FEAT["Multi-Level Feature Bank\n5 layers [20,24,28,32,36] → (5, 3200) per image\n(concept, file) → tensor on disk"]
        IMGS --> INTERN --> FEAT
    end

    subgraph MODEL["SUPAEEG Model  (trained)"]
        direction TB
        NET["EEGNetEncoder\nFlatten → MLP + ResidualAdd + LayerNorm\n(batch,17,100) → (batch,1024)"]
        PROJ["eeg_projector + share_encoder\nLinear(1024→512) → Linear(512→512)"]
        ZE["zE  (batch × 512)\nL2-normalized EEG embedding"]
        EEG --> NET --> PROJ --> ZE

        LOOKUP["InternViTFeatureLookup\n(batch, 5, 3200) → mean pool → (batch, 3200)"]
        IPROJ["img_pre_projector + img_projector + share_encoder\nLinear(3200→1024) → Linear(1024→512) → Linear(512→512)"]
        ZI["zI  (batch × 512)\nL2-normalized image embedding"]
        FEAT -- "batch lookup" --> LOOKUP --> IPROJ --> ZI
    end

    subgraph LOSS["Training Objective  (two-stage)"]
        direction TB
        NCE["Symmetric InfoNCE\nzE vs zI  (τ = 0.07)\nStage 1 + Stage 2"]
        MMD["MMD Loss\nStage 2 only"]
        TOTAL["Total Loss\nStage 1: InfoNCE\nStage 2: InfoNCE + MMD"]
        NCE & MMD --> TOTAL
    end

    subgraph EVAL["Evaluation — Zero-Shot Retrieval"]
        direction TB
        EMB["zE  (batch × 512)\nL2-normalized"]
        RET["Cosine Similarity\nvs InternViT image gallery  (200 × 512)\nTop-1 & Top-5 Accuracy"]
        EMB --> RET
    end

    ZE --> NCE
    ZI --> NCE
    ZE --> MMD
    ZI --> MMD
    ZE --> EMB
    FEAT -- "gallery" --> RET
    TOTAL -.->|"backprop"| MODEL
```

---

## Diagram 2 — Detailed LucidChart Realization (21 nodes)

Mirrors `lucidchart_spec.md` exactly. Stage labels = rectangular container boxes. Training container uses LR layout to show EEG and Image branches side-by-side. ★ = shared weights.

```mermaid
flowchart TB
    subgraph OFFLINE["① Offline Pre-processing  (run once, frozen)"]
        direction TB
        N1["Node 1 · THINGS Image Dataset\n~22,248 train · 200 test concepts"]
        N2["Node 2 · InternViT-6B-448px-V1-5\nFrozen · CLIPImageProcessor · 448×448"]
        N3["Node 3 · Multi-level Feature Extraction\nHooks at layers 20 · 24 · 28 · 32 · 36"]
        N4["Node 4 · Feature Bank on Disk\n(5 × 3200) per image"]
        N1 --> N2 --> N3 --> N4
    end

    subgraph INPUT["② Input Data"]
        direction TB
        N5["Node 5 · Raw EEG Recordings\n.npy files · 17 ch × 100 tp · 10 subjects"]
        N6["Node 6 · ThingsEEGDataset / DataLoader\nbatch → eeg (B,17,100) + concepts + files"]
        N5 --> N6
    end

    subgraph TRAINING["③ Training  (gradients flow)"]
        direction LR
        subgraph EEG["EEG Branch"]
            direction TB
            N7["Node 7 · EEGNetEncoder\nFlatten → (B,1700)\nLinear(1700→1024)\nResidualAdd + LayerNorm\n→ (B,1024)"]
            N8["Node 8 · eeg_projector\nLinear(1024→512)"]
            N9["Node 9 · share_encoder ★\nLinear(512→512)"]
            N10["Node 10 · L2-normalize → zE\n(B, 512)"]
            N7 --> N8 --> N9 --> N10
        end
        subgraph IMG["Image Branch"]
            direction TB
            N11["Node 11 · InternViTFeatureLookup\n→ (B, 5, 3200)"]
            N12["Node 12 · Mean pool dim=1\n→ (B, 3200)"]
            N13["Node 13 · img_pre_projector\nLinear(3200→1024)"]
            N14["Node 14 · img_projector\nLinear(1024→512)"]
            N15["Node 15 · share_encoder ★\nLinear(512→512)"]
            N16["Node 16 · L2-normalize → zI\n(B, 512)"]
            N11 --> N12 --> N13 --> N14 --> N15 --> N16
        end
        subgraph LOSS["Loss"]
            direction TB
            N17a["Node 17a · InfoNCE\nSymmetric · τ=0.07\nStage 1 + Stage 2"]
            N17b["Node 17b · MMD Loss\nStage 2 only"]
            N17c["Node 17c · Total Loss\nS1: InfoNCE\nS2: InfoNCE + MMD"]
            N17a & N17b --> N17c
        end
        N10 --> N17a & N17b
        N16 --> N17a & N17b
    end

    subgraph EVAL["④ Evaluation  (no gradients)"]
        direction TB
        N18["Node 18 · Test EEG → zE\nSame encoder path as Nodes 7–10"]
        N19["Node 19 · Image Gallery\n(200, 512) pre-computed"]
        N20["Node 20 · Cosine Similarity & Ranking\nN_queries × 200 matrix"]
        N21["Node 21 · Retrieval Accuracy\nTop-1 & Top-5 · Intra-subject & LOSO"]
        N18 & N19 --> N20 --> N21
    end

    OFFLINE --> INPUT
    N6 --> N7
    N4 -- "batch lookup" --> N11
    N4 -- "test gallery" --> N19
    N17c -.->|"backprop"| TRAINING
    N10 --> N18
```
