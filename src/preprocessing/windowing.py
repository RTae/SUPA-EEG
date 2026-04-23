import numpy as np


def sliding_window(eeg_data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Args:
        eeg_data: (n_samples, n_channels, n_times)

    Returns:
        (n_samples, n_windows, n_channels, window_size)
    """
    n_samples, n_channels, n_times = eeg_data.shape

    windows = []
    for start in range(0, n_times - window_size + 1, stride):
        end = start + window_size
        windows.append(eeg_data[:, :, start:end])

    return np.stack(windows, axis=1)