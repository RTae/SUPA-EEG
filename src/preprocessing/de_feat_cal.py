import mne
import numpy as np
from src.utilities import FREQ_BANDS


def de_feature_chunk(data_chunk: np.ndarray, info: mne.Info) -> np.ndarray:
    """
    Pure DE feature extractor (no windowing, no caching)

    Args:
        data_chunk: (n_samples, n_channels, n_times)

    Returns:
        (n_samples, n_features)
    """
    epochs = mne.EpochsArray(data=data_chunk, info=info, verbose=False)

    feat_list = []
    for f_min, f_max in FREQ_BANDS.values():
        spectrum = epochs.compute_psd(fmin=f_min, fmax=f_max, verbose=False)
        psd = spectrum.get_data() + 1e-10

        band_power = np.mean(psd, axis=-1)
        de = 0.5 * np.log(2 * np.pi * np.e * band_power)

        feat_list.append(de)

    return np.concatenate(feat_list, axis=1)