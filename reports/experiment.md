## Experiments
### 1. Seed Variance
**Goal:** Verify the model is stable and reduce result variance by running with different random seeds.
**Metric:** Mean ± std of Top-1 and Top-5 across 3 runs.

#### How to run:
1. Run model with differencent random seeds (e.g. `seed=42`, `seed=43`, `seed=44`) to get a sense of variability across runs.
```bash
bash ./scripts/exp_difference_seed.sh

# run with nohup to keep it running after closing the terminal
nohup bash ./scripts/exp_difference_seed.sh > exp_difference_seed.log 2>&1 &
tail -f exp_difference_seed.log
```

#### Results:
1. Inter (protocol=inter, epochs=200)

| Subject | Metric | Seed 42 | Seed 43 | Seed 44 | Avg |
|---------|--------|-----:|-----:|-----:|----:|
| sub-01  | Top-1  | 0.3250 | 0.2950 | 0.3200 | 0.3133 ± 0.016 |
| sub-01  | Top-5  | 0.6650 | 0.6350 | 0.6200 | 0.6400 ± 0.023 |
| sub-02  | Top-1  | 0.2500 | 0.1950 | 0.1950 | 0.2133 ± 0.032 |
| sub-02  | Top-5  | 0.5250 | 0.4950 | 0.4900 | 0.5033 ± 0.019 |
| sub-03  | Top-1  | 0.1250 | 0.1050 | 0.1200 | 0.1167 ± 0.010 |
| sub-03  | Top-5  | 0.2300 | 0.2150 | 0.2350 | 0.2267 ± 0.010 |
| sub-04  | Top-1  | 0.2200 | 0.2250 | 0.2150 | 0.2200 ± 0.005 |
| sub-04  | Top-5  | 0.4800 | 0.5200 | 0.5150 | 0.5050 ± 0.022 |
| sub-05  | Top-1  | 0.1500 | 0.1300 | 0.1550 | 0.1450 ± 0.013 |
| sub-05  | Top-5  | 0.3450 | 0.2750 | 0.3500 | 0.3233 ± 0.042 |
| sub-06  | Top-1  | 0.2300 | 0.2150 | 0.2200 | 0.2217 ± 0.008 |
| sub-06  | Top-5  | 0.4850 | 0.4300 | 0.4900 | 0.4683 ± 0.033 |
| sub-07  | Top-1  | 0.2250 | 0.2250 | 0.2600 | 0.2367 ± 0.020 |
| sub-07  | Top-5  | 0.5100 | 0.5200 | 0.5500 | 0.5267 ± 0.021 |
| sub-08  | Top-1  | 0.2400 | 0.2200 | 0.2000 | 0.2200 ± 0.020 |
| sub-08  | Top-5  | 0.4600 | 0.4250 | 0.4600 | 0.4483 ± 0.020 |
| sub-09  | Top-1  | 0.1200 | 0.1100 | 0.1100 | 0.1133 ± 0.006 |
| sub-09  | Top-5  | 0.3450 | 0.3500 | 0.3600 | 0.3517 ± 0.008 |
| sub-10  | Top-1  | 0.3100 | 0.3150 | 0.3200 | 0.3150 ± 0.005 |
| sub-10  | Top-5  | 0.5850 | 0.6050 | 0.5700 | 0.5867 ± 0.018 |
| **Avg All Subject** | Top-1  | 0.2195 | 0.2035 | 0.2115 | 0.2115 ± 0.008 |
| **Avg All Subject** | Top-5  | 0.4630 | 0.4470 | 0.4640 | 0.4580 ± 0.010 |

2. Intra (protocol=intra, epochs=100)

| Subject | Metric | Seed 42 | Seed 43 | Seed 44 | Avg |
|---------|--------|-----:|-----:|-----:|----:|
| sub-01  | Top-1  | 0.8500 | 0.8550 | 0.8650 | 0.8567 ± 0.008 |
| sub-01  | Top-5  | 0.9750 | 0.9900 | 1.0000 | 0.9883 ± 0.013 |
| sub-02  | Top-1  | 0.7750 | 0.7850 | 0.7800 | 0.7800 ± 0.005 |
| sub-02  | Top-5  | 0.9650 | 0.9550 | 0.9700 | 0.9633 ± 0.008 |
| sub-03  | Top-1  | 0.8700 | 0.8350 | 0.8200 | 0.8417 ± 0.026 |
| sub-03  | Top-5  | 0.9800 | 0.9850 | 0.9800 | 0.9817 ± 0.003 |
| sub-04  | Top-1  | 0.8500 | 0.8300 | 0.8400 | 0.8400 ± 0.010 |
| sub-04  | Top-5  | 0.9750 | 0.9850 | 0.9700 | 0.9767 ± 0.008 |
| sub-05  | Top-1  | 0.7250 | 0.7650 | 0.7450 | 0.7450 ± 0.020 |
| sub-05  | Top-5  | 0.9350 | 0.9500 | 0.9550 | 0.9467 ± 0.010 |
| sub-06  | Top-1  | 0.8700 | 0.8800 | 0.8550 | 0.8683 ± 0.013 |
| sub-06  | Top-5  | 0.9750 | 0.9900 | 0.9750 | 0.9800 ± 0.009 |
| sub-07  | Top-1  | 0.7800 | 0.8100 | 0.8150 | 0.8017 ± 0.019 |
| sub-07  | Top-5  | 0.9500 | 0.9750 | 0.9850 | 0.9700 ± 0.018 |
| sub-08  | Top-1  | 0.9100 | 0.8950 | 0.9000 | 0.9017 ± 0.008 |
| sub-08  | Top-5  | 0.9950 | 1.0000 | 1.0000 | 0.9983 ± 0.003 |
| sub-09  | Top-1  | 0.6550 | 0.6350 | 0.6550 | 0.6483 ± 0.012 |
| sub-09  | Top-5  | 0.8950 | 0.8700 | 0.8650 | 0.8767 ± 0.016 |
| sub-10  | Top-1  | 0.7950 | 0.7800 | 0.8150 | 0.7967 ± 0.018 |
| sub-10  | Top-5  | 0.9800 | 0.9700 | 0.9850 | 0.9783 ± 0.008 |
| **Avg All Subject** | **Top-1**  | 0.8080 | 0.8070 | 0.8085 | 0.8078 ± 0.001 |
| **Avg All Subject** | **Top-5**  | 0.9625 | 0.9670 | 0.9685 | 0.9660 ± 0.003 |

## 1. Seed Variance
**Goal:** Verify the model is stable and results are not lucky.
**What changes:** `seed` in config (42, 43, 44).
**What's fixed:** Everything else — same architecture, data, hyperparameters.
**Metric:** Mean ± std of Top-1 and Top-5 across 3 runs.
### 2. EEG Encoder Ablation
**Goal:** Find the best architecture for encoding EEG signals.
**What changes:** The `eeg_encoder` block inside `SUPAEEG`.
**What's fixed:** Image encoder, shared encoder, training config.

| Variant | Architecture Detail |
|---|---|
| `EEGProject` *(current)* | Flatten (17×100→1700) → Linear → GELU + Linear Residual → LayerNorm |
| `EEGNet` | Temporal conv → Depthwise spatial conv over channels → Separable conv → ELU + pooling |
| `TSConv` | Temporal conv block → Spatial filtering → Temporal aggregation |
| `EEGConformer` | Conv patch embed → Transformer encoder → aggregation (CNN + self-attention hybrid) |
| `ATM` | Attention-based temporal mixing across channels and time |
### 3. Image Backbone Ablation
**Goal:** Find which vision model's features align best with EEG representations.
**What changes:** Pre-extracted image features and `image_input_dim`.
**What's fixed:** EEG encoder, shared encoder, router/layer config.

| Variant                 | Type                | Notes                                 |
| ----------------------- | ------------------- | ------------------------------------- |
| `InternViT` *(current)* | Supervised ViT 6B   | Multilayer features, used with router |
| `RN50`                  | CLIP ResNet-50      | Convolutional, smallest CLIP model    |
| `RN101`                 | CLIP ResNet-101     | Deeper convolutional CLIP             |
| `ViT-B-16`              | CLIP ViT Base       | Standard ViT, 16px patches            |
| `ViT-H-14`              | CLIP ViT Huge       | Large ViT, 14px patches               |
| `ViT-bigG-14`           | CLIP ViT bigG       | Largest CLIP model                    |
| `DINOv2`                | Self-supervised ViT | No language supervision               |
| `EVA-02`                | EVA-CLIP ViT        | Strong open-source CLIP variant       |
### 4. Image Layer Selection Ablation
**Goal:** Validate that `SubjectAwareRouter` and multi-layer blending adds value over fixed single-layer features.
**What changes:** How InternViT layer features are combined.
**What's fixed:** Everything else — same backbone (InternViT), EEG encoder, shared encoder.

| Variant              | Description                                                         |
| -------------------- | ------------------------------------------------------------------- |
| `Router` *(current)* | Learned per-subject softmax blending of layers [20, 24, 28, 32, 36] |
| `Uniform`            | Simple average of all 5 layers, no learned weights                  |
| `Single layer 20`    | Early layer only — low-level visual features                        |
| `Single layer 28`    | Middle layer only — mid-level features                              |
| `Single layer 36`    | Late layer only — high-level semantic features                      |

### 5. Shared Encoder Ablation
**Goal:** Validate that weight sharing between EEG and image paths drives cross-modal alignment.
**What changes:** The `share_encoder` module applied to both modalities after their respective projectors.
**What's fixed:** EEG encoder, image encoder, training config.

| Variant              | Description                                                        |
| -------------------- | ------------------------------------------------------------------ |
| `linear` *(current)* | Single `nn.Linear(512→512)` shared by both EEG and image           |
| `none`               | Removed — both paths go directly to L2 normalize                   |
| `separate`           | Two independent `nn.Linear(512→512)`, one per modality, no sharing |
| `transformer`        | Small Transformer block (1–2 layers, 512-dim token)                |
| `jepa`               | Split 512-dim into sub-tokens → ViT-style CLS encoder              |
### 6. EEG Channels
**Goal:** Test whether denser electrode coverage improves EEG representations.
**What changes:** `eeg_suffix`, `n_channels` in config.
**What's fixed:** Architecture, image encoder, training config.

| Variant             | Description                                          |
| ------------------- | ---------------------------------------------------- |
| `17-ch` *(current)* | Default electrode setup, `eeg_suffix=""`             |
| `63-ch`             | Denser coverage, `eeg_suffix="_63"`, `n_channels=63` |
### 7. Higher Sampling Rate (1000Hz)
**What changes:** Use the raw 1000Hz EEG instead of the 100Hz downsampled version.  
**Impact on architecture:** n_timepoints increases from 100 → 1000, so the EEG encoder input grows 10×. This may require adjusting eeg_feature_dim or using strided convolutions.

| Variant             | Sampling Rate   | n_timepoints |
| ------------------- | --------------- | ------------ |
| `100Hz` _(current)_ | Downsampled     | 100          |
| `1000Hz`            | Full resolution | 1000         |

**Worth doing because:** downsampling to 100Hz discards high-frequency neural signals (gamma band, 30–100Hz) that may carry visual information.

### 8. Wrong Prediction Analysis

When the model predicts incorrectly, you can analyze **what it got wrong and why**:

- **Semantic similarity of errors** — are wrong predictions visually/semantically similar to the correct image? (e.g. predicted "cat" when correct was "dog"). Use the image embeddings to compute similarity between the predicted concept and true concept.
- **Confusion matrix on Top-5** — which concepts are most often confused with each other? Plot as a heatmap.
- **Hard vs. easy concepts** — rank concepts by per-concept accuracy. Are low-level visual concepts (e.g. simple shapes) easier than high-level semantic ones?
- **Subject-level error patterns** — do all subjects fail on the same concepts or different ones? This can reveal subject-specific neural representations.
- **Visualize failure cases** — show the true image alongside the top-5 retrieved images for the worst-performing concepts.

## INTRA-SUBJECT
| Method             | Metric    | Sub01 | Sub02 | Sub03 | Sub04 | Sub05 | Sub06 | Sub07 | Sub08 | Sub09 | Sub10 |  Avg  |
|--------------------|-----------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| NICE               | Top-1     | 13.2% | 13.5% | 14.5% | 20.6% | 10.1% | 16.5% | 17.0% | 22.9% | 15.4% | 17.4% | 16.1% |
| NICE               | Top-5     | 39.5% | 40.3% | 42.7% | 52.7% | 31.5% | 44.0% | 42.1% | 56.1% | 41.6% | 45.8% | 43.6% |
| ATM                | Top-1     | 25.6% | 22.0% | 25.0% | 31.4% | 12.9% | 21.3% | 30.5% | 38.8% | 34.4% | 29.1% | 27.1% |
| ATM                | Top-5     | 60.4% | 54.5% | 62.4% | 60.9% | 43.0% | 51.1% | 61.5% | 72.0% | 51.5% | 63.5% | 58.1% |
| UBP                | Top-1     | 41.2% | 51.2% | 51.2% | 51.1% | 42.2% | 57.5% | 49.0% | 58.6% | 45.1% | 61.5% | 50.9% |
| UBP                | Top-5     | 70.5% | 80.9% | 82.0% | 76.9% | 72.8% | 83.5% | 79.9% | 85.8% | 76.2% | 88.2% | 79.7% |
| NeuroBridge        | Top-1     | 50.0% | 63.2% | 61.6% | 61.4% | 54.8% | 69.7% | 62.7% | 71.2% | 64.0% | 73.6% | 63.2% |
| NeuroBridge        | Top-5     | 77.6% | 90.6% | 91.1% | 90.0% | 85.0% | 92.9% | 88.8% | 95.1% | 91.0% | 97.1% | 89.9% |
| Shallow            | Top-1     | 75.4% | 87.3% | 82.9% | 79.1% | 74.9% | 90.2% | 79.0% | 86.9% | 81.3% | 89.3% | 82.6% |
| Shallow            | Top-5     | 94.3% | 99.0% | 98.3% | 96.5% | 96.4% | 99.2% | 97.3% | 99.4% | 97.8% | 99.2% | 97.7% |
| SAMGA              | Top-1     | 85.2% | 94.4% | 91.8% | 89.7% | 86.3% | 97.2% | 89.2% | 94.8% | 88.7% | 96.4% | 91.3% |
| SAMGA              | Top-5     | 95.5% | 99.8% | 98.2% | 98.7% | 98.4% | 99.9% | 98.7% | 99.8% | 99.2% | 99.8% | 98.8% |
| Baseline           | Top-1     | 86.5% | 84.5% | 83.0% | 78.5% | 76.5% | 89.5% | 85.5% | 92.0% | 83.0% | 91.5% | 85.0% |
| Baseline           | Top-5     | 99.0% | 98.0% | 99.5% | 98.5% | 95.5% | 98.5% | 98.0% | 100.0% | 98.0% | 99.5% | 98.5% |
| Smooth aug + DC removal + eval every epoch | Top-1 | 88.5% | 88.0% | 85.5% | 80.5% | 78.5% | 90.5% | 82.5% | 92.0% | 86.0% | 95.5% | 86.8% |
| Smooth aug + DC removal + eval every epoch | Top-5 | 99.0% | 100.0% | 99.0% | 99.5% | 96.5% | 98.0% | 97.5% | 100.0% | 98.5% | 100.0% | 98.8% |
| Smooth aug + DC removal + eval every epoch + Select time window | Top-1 | 92.0% | 88.5% | 88.0% | 83.0% | 81.0% | 91.0% | 84.0% | 93.0% | 88.0% | 94.5% | 88.3% |
| Smooth aug + DC removal + eval every epoch + Select time window | Top-5 | 100.0% | 99.5% | 99.5% | 99.5% | 97.0% | 99.5% | 99.0% | 99.5% | 99.5% | 99.5% | 99.3% |

## INTER-SUBJECT
| Method              | Metric | Sub01 | Sub02 | Sub03 | Sub04 | Sub05 | Sub06 | Sub07 | Sub08 | Sub09 | Sub10 | Avg |
|---------------------|--------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|----:|
| NICE                | Top-1 |  7.6% |  5.9% |  6.0% |  6.3% |  4.4% |  5.6% |  5.6% |  6.3% |  5.7% |  8.4% |  6.2% |
| NICE                | Top-5 | 22.8% | 20.5% | 22.3% | 20.7% | 18.3% | 22.2% | 19.7% | 22.0% | 17.6% | 28.3% | 21.4% |
| ATM                 | Top-1 | 10.5% |  7.1% | 11.9% | 14.7% |  7.0% | 11.1% | 16.1% | 15.0% |  4.9% | 20.5% | 11.9% |
| ATM                 | Top-5 | 26.8% | 24.8% | 33.8% | 39.4% | 23.9% | 35.8% | 43.5% | 40.3% | 22.7% | 46.5% | 34.1% |
| UBP                 | Top-1 | 11.5% | 15.5% |  9.8% | 13.0% |  8.8% | 11.7% | 10.2% | 12.2% | 15.5% | 16.0% | 12.4% |
| UBP                 | Top-5 | 29.7% | 40.0% | 27.0% | 32.3% | 33.8% | 31.0% | 23.8% | 32.2% | 40.5% | 43.5% | 33.4% |
| NeuroBridge         | Top-1 | 23.2% | 21.2% | 13.2% | 17.0% | 14.5% | 25.0% | 15.3% | 20.1% | 13.7% | 27.2% | 19.0% |
| NeuroBridge         | Top-5 | 52.4% | 49.3% | 36.5% | 45.3% | 37.7% | 55.0% | 45.1% | 44.9% | 36.5% | 56.3% | 45.9% |
| Shallow             | Top-1 | 24.6% | 31.3% | 11.4% | 19.9% | 19.0% | 24.1% | 18.6% | 17.6% | 23.3% | 34.6% | 22.4% |
| Shallow             | Top-5 | 54.7% | 61.5% | 31.1% | 48.8% | 45.5% | 49.8% | 51.6% | 46.7% | 54.9% | 63.2% | 50.8% |
| SAMGA               | Top-1 | 36.3% | 42.2% | 24.7% | 34.0% | 30.0% | 35.2% | 35.1% | 28.7% | 28.1% | 49.6% | 34.4% |
| SAMGA               | Top-5 | 69.8% | 71.7% | 50.7% | 66.7% | 64.3% | 67.7% | 61.0% | 59.5% | 57.0% | 79.4% | 64.8% |
| Baseline            | Top-1 | 25.5% | 10.5% | 11.0% | 11.5% | 7.5%  | 11.0% | 17.0% | 18.5% | 5.0%  | 25.5% | 14.3% |
| Baseline            | Top-5 | 56.5% | 24.5% | 25.0% | 32.0% | 21.5% | 37.5% | 48.5% | 42.5% | 23.0% | 54.0% | 36.5% |
| With Subject Aware + Average signal | Top-1 | 29.0% | 19.5% | 9.5%  | 16.0% | 13.5% | 18.5% | 22.0% | 16.0% | 13.0% | 27.5% | 18.4% |
| With Subject Aware + Average signal | Top-5 | 62.5% | 43.5% | 24.5% | 40.5% | 29.5% | 43.5% | 49.5% | 41.5% | 39.5% | 59.5% | 43.4% |
| Smooth aug + DC removal + eval every epoch + Average signal + Full-window | Top-1 | 33.0% | 22.0% | 10.5% | 20.0% | 13.5% | 24.5% | 23.5% | 23.0% | 11.5% | 32.0% | 21.4% |
| Smooth aug + DC removal + eval every epoch + Average signal + Full-window | Top-5 | 67.0% | 54.0% | 24.5% | 48.0% | 31.5% | 48.5% | 48.0% | 46.0% | 31.5% | 55.5% | 45.5% |