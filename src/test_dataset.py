"""Unit tests for src/dataset.py.

Tests cover the data_average logic in ThingsEEGDataset._load_eeg_data
without touching the filesystem — the static method is exercised directly
with in-memory numpy arrays via a minimal monkey-patch.
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch

from src.dataset import ThingsEEGDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_npy(raw: np.ndarray) -> dict:
    """Wrap a raw array in the dict format _load_eeg_data expects."""
    return {"preprocessed_eeg_data": raw}


def _run_load(raw: np.ndarray, data_average: bool):
    """
    Call _load_eeg_data with a single fake subject backed by `raw`.
    Returns (eeg_tensor, number_of_repetitions, n_subjects_loaded).
    """
    dataset_dir = "/fake"
    folder_list = ["sub-01"]
    file_name = "preprocessed_eeg_training"
    device = torch.device("cpu")

    npy_item = _make_npy(raw)

    def fake_load(path, allow_pickle):
        m = MagicMock()
        m.item.return_value = npy_item
        return m

    with patch("os.path.isdir", return_value=True), \
         patch("os.path.isfile", return_value=True), \
         patch("numpy.load", side_effect=fake_load):

        return ThingsEEGDataset._load_eeg_data(
            dataset_dir, folder_list, file_name, device, subject=-1,
            data_average=data_average,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataAverageFalse:
    """data_average=False: existing flatten behaviour is unchanged."""

    def test_output_shape(self):
        raw = np.random.randn(10, 4, 17, 100).astype(np.float32)
        eeg, n_reps, n_subj = _run_load(raw, data_average=False)
        assert eeg.shape == (40, 17, 100)

    def test_number_of_repetitions(self):
        raw = np.random.randn(10, 4, 17, 100).astype(np.float32)
        _, n_reps, _ = _run_load(raw, data_average=False)
        assert n_reps == 4

    def test_subjects_loaded(self):
        raw = np.random.randn(10, 4, 17, 100).astype(np.float32)
        _, _, n_subj = _run_load(raw, data_average=False)
        assert n_subj == 1

    def test_values_preserved(self):
        """Values must match manual reshape, not a mean."""
        raw = np.random.randn(10, 4, 17, 100).astype(np.float32)
        eeg, _, _ = _run_load(raw, data_average=False)
        expected = raw.reshape(-1, 17, 100)
        np.testing.assert_array_equal(eeg, expected)


class TestDataAverageTrue:
    """data_average=True: repetitions are averaged, n_reps forced to 1."""

    def test_output_shape(self):
        raw = np.random.randn(10, 4, 17, 100).astype(np.float32)
        eeg, n_reps, n_subj = _run_load(raw, data_average=True)
        assert eeg.shape == (10, 17, 100)

    def test_number_of_repetitions_is_1(self):
        raw = np.random.randn(10, 4, 17, 100).astype(np.float32)
        _, n_reps, _ = _run_load(raw, data_average=True)
        assert n_reps == 1

    def test_subjects_loaded(self):
        raw = np.random.randn(10, 4, 17, 100).astype(np.float32)
        _, _, n_subj = _run_load(raw, data_average=True)
        assert n_subj == 1

    def test_values_are_mean(self):
        """Values must equal mean over the repetitions axis."""
        raw = np.random.randn(10, 4, 17, 100).astype(np.float32)
        eeg, _, _ = _run_load(raw, data_average=True)
        expected = raw.mean(axis=1)
        np.testing.assert_allclose(eeg, expected, rtol=1e-5)

    def test_quarter_the_samples_vs_flatten(self):
        raw = np.random.randn(16, 4, 17, 100).astype(np.float32)
        avg_eeg, _, _ = _run_load(raw, data_average=True)
        flat_eeg, _, _ = _run_load(raw, data_average=False)
        assert avg_eeg.shape[0] * 4 == flat_eeg.shape[0]


class TestDataAverageVariousRepCounts:
    """data_average works for datasets with rep counts other than 4."""

    @pytest.mark.parametrize("n_reps", [1, 2, 8, 80])
    def test_averaged_shape(self, n_reps: int):
        raw = np.random.randn(5, n_reps, 17, 100).astype(np.float32)
        eeg, n_reps_out, _ = _run_load(raw, data_average=True)
        assert eeg.shape == (5, 17, 100)
        assert n_reps_out == 1

    @pytest.mark.parametrize("n_reps", [1, 2, 8, 80])
    def test_flattened_shape(self, n_reps: int):
        raw = np.random.randn(5, n_reps, 17, 100).astype(np.float32)
        eeg, n_reps_out, _ = _run_load(raw, data_average=False)
        assert eeg.shape == (5 * n_reps, 17, 100)
        assert n_reps_out == n_reps
