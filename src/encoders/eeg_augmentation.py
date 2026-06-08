"""EEG smooth augmentation — gaussian smoothing along time axis.

Applied during training only with probability p.
Reduces high-frequency noise and trial-to-trial variability,
acting as a regulariser that improves cross-subject generalisation.
Matches SAMGA --eeg_aug smooth configuration.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def smooth_eeg(
    eeg: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
    p: float = 0.3,
) -> torch.Tensor:
    """Apply Gaussian smoothing along time axis with probability p.

    Args:
        eeg:         (batch, n_channels, n_timepoints)
        kernel_size: size of gaussian kernel (odd number)
        sigma:       gaussian standard deviation
        p:           probability of applying augmentation per sample

    Returns:
        (batch, n_channels, n_timepoints) — same shape as input
    """
    if not torch.is_floating_point(eeg):
        eeg = eeg.float()

    if eeg.ndim != 3:
        raise ValueError(
            "smooth_eeg expects (batch, n_channels, n_timepoints), "
            f"got shape {tuple(eeg.shape)}"
        )

    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    # If disabled, return early (and do not advance RNG state).
    if p <= 0.0:
        return eeg

    batch, n_channels, n_timepoints = eeg.shape

    # build gaussian kernel: (1, 1, kernel_size)
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, dtype=eeg.dtype, device=eeg.device)
    kernel = torch.exp(-(x ** 2) / (2.0 * float(sigma) ** 2))
    kernel = (kernel / kernel.sum()).view(1, 1, kernel_size)

    # apply kernel per channel by flattening (batch * channels) into the batch dim
    eeg_flat = eeg.reshape(batch * n_channels, 1, n_timepoints)
    smoothed_flat = F.conv1d(eeg_flat, kernel, padding=half)
    smoothed = smoothed_flat.reshape(batch, n_channels, n_timepoints)

    if p >= 1.0:
        return smoothed

    # per-sample mask: (batch, 1, 1) — True = apply smooth
    mask = (torch.rand(batch, 1, 1, device=eeg.device) < p)
    return torch.where(mask, smoothed, eeg)
