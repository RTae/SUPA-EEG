# EEG data

 Each EEG data sample has a size of **(nchannels, fs * T )**, where 
 - **nchannels** is the number of EEG electrodes, which is 62 in our dataset
   - The reference electrode name is in `data/mode/montage_ch_names.json`
   - Also the corresponding 3D coordinates of the electrodes are in `data/mode/montage.fif`
 - **fs** is the sampling frequency of the device, which is 1000 Hz in our dataset
 - **T** is the time window size, which in our dataset is the duration of the image stimulus presentation, i.e., 500 ms

## What do electrode channels tell us?

The electrode channels correspond to specific locations on the scalp where the EEG signals are recorded. Each channel represents the electrical activity of the brain at that particular location. The naming convention follows the **10-20 system**: the letter indicates the brain region, odd numbers = left hemisphere, even numbers = right hemisphere, and Z = midline.

| Region prefix | Electrodes (examples) | Brain area | Associated with |
|---|---|---|---|
| **FP, AF** | FP1, FP2, AF3, AF4 | Prefrontal / Anterior frontal | Decision-making, attention, working memory |
| **F** | F3, FZ, F4, F7, F8 | Frontal | Motor planning, executive function, language (left) |
| **FC, C** | FC3, CZ, C4 | Central (motor cortex) | Motor execution, somatosensory processing |
| **T** | T7, T8 | Temporal | Auditory processing, language comprehension, memory |
| **CP, P** | CP3, PZ, P4 | Parietal | Spatial awareness, sensory integration, attention |
| **PO, O** | PO3, OZ, O2 | Parieto-occipital / Occipital | **Visual processing** |
| **CB** | CB1, CB2 | Cerebellum (near) | Motor coordination |

For our EEG-ImageNet dataset (visual stimulus → brain response), the **occipital and parieto-occipital channels** (O1, OZ, O2, PO3–PO8) are especially important because they capture activity from the **visual cortex**, which processes the image stimuli.

## What does each timestep tell us?

Each timestep is a **single voltage sample** at a given electrode. With fs = 1000 Hz, each timestep = **1 millisecond**. The temporal patterns encode **event-related potentials (ERPs)** — characteristic brain responses to the visual stimulus:

| Time range (ms) | ERP component | What it reflects |
|---|---|---|
| ~50–100 | **C1** | Initial activation of primary visual cortex (V1) |
| ~80–120 | **P1** | Early visual processing, spatial attention |
| ~130–200 | **N1 / N170** | Object/face recognition, feature extraction |
| ~200–300 | **P2 / N2** | Higher-level categorization, stimulus matching |
| ~300–500 | **P3 / P300** | Stimulus evaluation, decision-making, memory updating |

In our dataset's default window `(40, 440)`, we capture **40–440 ms** post-stimulus, covering the key visual processing components from C1 through P300. The models learn to distinguish image categories based on **both** where (channels) and **when** (timesteps) discriminative neural activity occurs. 

## EEG Features
The raw EEG data can be transformed into various features that capture different aspects of the brain's response to visual stimuli:
### 1. Time-domain features
- **Event-Related Potentials (ERPs)**: Average voltage across trials for each channel, capturing characteristic peaks (e.g., P1, N1) associated with visual processing stages.
- **Peak amplitudes and latencies**: The height and timing of ERP components can indicate the strength and speed of neural responses to different image categories.
### 2. Frequency-domain features
- **Power spectral density (PSD)**: Measures the power of different frequency bands (e.g., alpha, beta, gamma) which can reflect cognitive states and processing demands.
- **Band-specific power**: Changes in specific frequency bands (e.g., increased gamma power) can indicate enhanced visual processing or attention to certain image features.
### 3. Time-frequency features
- **Wavelet transforms**: Capture how frequency content changes over time, revealing dynamic neural responses to visual stimuli.
- **Event-related spectral perturbations (ERSP)**: Measure changes in power across frequencies and time, indicating how the brain's oscillatory activity is modulated by different image categories.
### 4. Connectivity features
- **Functional connectivity**: Measures the correlation or coherence between channels, indicating how different brain regions interact during visual processing.
- **Effective connectivity**: Captures directional influences between channels, revealing how information flows through the brain in response to visual stimuli.
### 5. Spatial features
- **Topographical maps**: Visual representations of voltage distributions across the scalp, highlighting which regions are most active for different image categories.
- **Source localization**: Estimating the underlying brain sources generating the observed EEG signals, providing insights into which cortical areas are involved in processing specific visual features