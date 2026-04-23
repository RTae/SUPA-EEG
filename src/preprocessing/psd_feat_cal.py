import mne
import numpy as np
from src.utilities import FREQ_BANDS


def psd_feature_chunk(data_chunk: np.ndarray, info: mne.Info) -> np.ndarray:
    epochs = mne.EpochsArray(data=data_chunk, info=info, verbose=False)

    feat_list = []
    for f_min, f_max in FREQ_BANDS.values():
        spectrum = epochs.compute_psd(fmin=f_min, fmax=f_max, verbose=False)
        psd = spectrum.get_data()

        band_power = np.mean(psd, axis=-1)
        feat_list.append(band_power)

    return np.concatenate(feat_list, axis=1)