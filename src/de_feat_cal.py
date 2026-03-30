import os
from pathlib import Path

import mne
import numpy as np

from utilities import DEFAULT_DATA_DIR, FREQ_BANDS


def de_feat_cal(
    eeg_data: np.ndarray,
    subject: int,
    granularity: str,
    cache_dir: str | Path | None = None,
) -> np.ndarray:
    """Compute differential-entropy features per frequency band.

    Results are cached to *cache_dir* so subsequent calls with the same
    subject / granularity return instantly.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_DATA_DIR / "de_feat"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"{subject}_{granularity}_de.npy"
    if cache_path.exists():
        return np.load(cache_path)

    channel_names = [f"EEG{i}" for i in range(1, 63)]
    info = mne.create_info(ch_names=channel_names, sfreq=1000, ch_types="eeg")
    epochs = mne.EpochsArray(data=eeg_data, info=info)

    de_feat_list = []
    for f_min, f_max in FREQ_BANDS.values():
        spectrum = epochs.compute_psd(fmin=f_min, fmax=f_max)
        psd = spectrum.get_data() + 1e-10
        diff_entropy = np.sum(np.log(psd), axis=-1)
        de_feat_list.append(diff_entropy)

    de_feat = np.concatenate(de_feat_list, axis=1)
    np.save(cache_path, de_feat)
    return de_feat
