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

    batch, n_channels, n_timepoints = eeg.shape

    # build gaussian kernel: (1, 1, kernel_size)
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, dtype=eeg.dtype, device=eeg.device)
    kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size)

    # apply per channel via grouped conv1d
    # reshape: (batch, n_channels, T) → (1, batch * n_channels, T)
    # so that C_in / groups = 1 per group
    eeg_flat = eeg.reshape(1, batch * n_channels, n_timepoints)
    kernel_expanded = kernel.expand(batch * n_channels, 1, -1)
    smoothed_flat = F.conv1d(
        eeg_flat,
        kernel_expanded,
        padding=half,
        groups=batch * n_channels,
    )
    smoothed = smoothed_flat.reshape(batch, n_channels, n_timepoints)

    # per-sample mask: (batch, 1, 1) — True = apply smooth
    mask = (torch.rand(batch, 1, 1, device=eeg.device) < p)
    return torch.where(mask, smoothed, eeg)
