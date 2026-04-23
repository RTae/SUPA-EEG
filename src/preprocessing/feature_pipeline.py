import numpy as np


def normalize(eeg_data: np.ndarray) -> np.ndarray:
    return (eeg_data - eeg_data.mean(axis=-1, keepdims=True)) / (
        eeg_data.std(axis=-1, keepdims=True) + 1e-6
    )


def extract_features(
    eeg_data: np.ndarray,
    feature_fn,
    info,
    window_fn=None,
    aggregate: str = "flatten",
) -> np.ndarray:
    """
    Minimal pipeline:
    - normalize
    - optional windowing
    - feature extraction per chunk
    - aggregation
    """

    eeg_data = normalize(eeg_data)

    # No windowing
    if window_fn is None:
        return feature_fn(eeg_data, info)

    # Windowing
    windows = window_fn(eeg_data)

    feats = []
    for w in range(windows.shape[1]):
        chunk = windows[:, w]
        feats.append(feature_fn(chunk, info))

    feats = np.stack(feats, axis=1)  # (n_samples, n_windows, feat_dim)

    if aggregate == "flatten":
        return feats.reshape(feats.shape[0], -1)

    elif aggregate == "mean":
        return feats.mean(axis=1)

    else:
        raise ValueError(f"Unsupported aggregation: {aggregate}")