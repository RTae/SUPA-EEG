"""Unit tests for src/trainer/metrics.py retrieval helpers.

These tests cover the three pure-numpy evaluation functions:
  - retrieve_topk        (paper-style: square N×N diagonal matching)
  - retrieve_all         (paper-style: wraps retrieve_topk with cosine similarity)
  - _label_retrieval_counts  (our label/prototype-based rectangular retrieval)

Torch is stubbed out so the tests run without a GPU runtime.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub torch and trainer.loss so metrics.py can be imported without torch.
# The functions under test are pure numpy; torch is only used in other
# functions within the same file (evaluate_classifier, etc.).
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    return m


_torch = _make_stub("torch")
_torch.no_grad = lambda: (lambda fn: fn)  # passthrough: @torch.no_grad() decorator
_torch.Tensor = object
_torch.device = object               # used in type annotations
_torch.nn = _make_stub("torch.nn")
_torch.nn.Module = object
_torch_nn_func = _make_stub("torch.nn.functional")
_torch_utils = _make_stub("torch.utils")
_torch_utils_data = _make_stub("torch.utils.data")
_torch_utils_data.DataLoader = object
_torch_utils.data = _torch_utils_data

for _name, _mod in [
    ("torch", _torch),
    ("torch.nn", _torch.nn),
    ("torch.nn.functional", _torch_nn_func),
    ("torch.utils", _torch_utils),
    ("torch.utils.data", _torch_utils_data),
]:
    sys.modules.setdefault(_name, _mod)

# Stub trainer.loss so the relative import inside metrics.py doesn't need torch.
_loss_stub = _make_stub("trainer.loss")
_loss_stub.batch_hard_triplet_loss = None
sys.modules.setdefault("trainer.loss", _loss_stub)

# Load metrics.py directly from its file path to bypass trainer/__init__.py,
# which pulls in loss.py and other torch-dependent modules.
_src_dir = Path(__file__).parent.parent / "src"
_metrics_path = _src_dir / "trainer" / "metrics.py"
_spec = importlib.util.spec_from_file_location("trainer.metrics", _metrics_path)
_metrics_mod = importlib.util.module_from_spec(_spec)
_metrics_mod.__package__ = "trainer"   # needed for the relative import inside metrics
sys.modules["trainer.metrics"] = _metrics_mod
_spec.loader.exec_module(_metrics_mod)

retrieve_topk = _metrics_mod.retrieve_topk
retrieve_all = _metrics_mod.retrieve_all
_label_retrieval_counts = _metrics_mod._label_retrieval_counts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalised(rng: np.random.Generator, shape: tuple) -> np.ndarray:
    x = rng.standard_normal(shape).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# retrieve_topk  (paper-style: square N×N, diagonal = correct pair)
# ---------------------------------------------------------------------------

class TestRetrieveTopk:
    def test_perfect_retrieval(self):
        # Identical features → every diagonal entry is the global maximum → rank 1.
        rng = np.random.default_rng(0)
        feats = _normalised(rng, (10, 64))
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(feats, feats)
        count_k, count_1 = retrieve_topk(sim, k=5)
        assert count_1 == 10
        assert count_k == 10

    def test_constructed_ranks(self):
        # Simple 5×5 matrix: every row is [5, 4, 3, 2, 1].
        # Diagonal entry of row i is the (i+1)-th largest value, so diagonal rank = i+1.
        #   Row 0: col 0 = 5 → rank 1
        #   Row 1: col 1 = 4 → rank 2
        #   Row 2: col 2 = 3 → rank 3
        #   Row 3: col 3 = 2 → rank 4
        #   Row 4: col 4 = 1 → rank 5
        N = 5
        sim = np.tile(np.arange(N, 0, -1, dtype=np.float32), (N, 1))

        count_k5, count_1 = retrieve_topk(sim, k=5)
        assert count_1 == 1    # only row 0 has diagonal rank 1
        assert count_k5 == 5   # all rows have diagonal rank ≤ 5

        count_k3, _ = retrieve_topk(sim, k=3)
        assert count_k3 == 3   # rows 0, 1, 2 have diagonal rank ≤ 3

    def test_rectangle_raises(self):
        sim = np.ones((8, 40), dtype=np.float32)
        with pytest.raises(ValueError, match="square"):
            retrieve_topk(sim, k=5)

    def test_1d_raises(self):
        sim = np.ones(10, dtype=np.float32)
        with pytest.raises(ValueError):
            retrieve_topk(sim, k=1)


# ---------------------------------------------------------------------------
# retrieve_all  (paper-style: same concept ordering, diagonal match)
# ---------------------------------------------------------------------------

class TestRetrieveAll:
    def test_perfect_retrieval(self):
        rng = np.random.default_rng(1)
        feats = _normalised(rng, (20, 64))
        count_5, count_1, total = retrieve_all(feats, feats)
        assert total == 20
        assert count_1 == 20
        assert count_5 == 20

    def test_independent_features_reduce_top1(self):
        # Independent random EEG and image features: top-1 should drop well below N.
        rng = np.random.default_rng(42)
        N = 50
        eeg = _normalised(rng, (N, 64))
        img = _normalised(rng, (N, 64))
        count_5, count_1, total = retrieve_all(eeg, img)
        assert total == N
        assert count_1 < N  # astronomically unlikely to be N with random features

    def test_return_contract(self):
        rng = np.random.default_rng(7)
        feats = _normalised(rng, (10, 32))
        count_5, count_1, total = retrieve_all(feats, feats)
        assert isinstance(count_5, int)
        assert isinstance(count_1, int)
        assert isinstance(total, int)
        assert count_5 >= count_1  # top-5 always ≥ top-1


# ---------------------------------------------------------------------------
# _label_retrieval_counts  (label-indexed rectangular retrieval)
# ---------------------------------------------------------------------------

class TestLabelRetrievalCounts:
    def test_all_rank_1(self):
        # Correct class always has the highest similarity.
        N_trials, N_classes = 8, 4
        labels = np.array([i % N_classes for i in range(N_trials)])
        sim = np.zeros((N_trials, N_classes), dtype=np.float32)
        sim[np.arange(N_trials), labels] = 1.0
        count_k, count_1 = _label_retrieval_counts(sim, labels, k=5)
        assert count_1 == N_trials
        assert count_k == N_trials

    def test_all_rank_last(self):
        # Correct class always has the lowest similarity → rank = N_classes (last).
        N_trials, N_classes = 6, 5
        labels = np.zeros(N_trials, dtype=int)  # correct = col 0
        sim = np.ones((N_trials, N_classes), dtype=np.float32) * 2.0
        sim[np.arange(N_trials), labels] = 0.0  # correct col is minimum
        # k = N_classes - 1: correct item is rank N_classes, so count = 0
        count_k, count_1 = _label_retrieval_counts(sim, labels, k=N_classes - 1)
        assert count_1 == 0
        assert count_k == 0

    def test_mixed_known_ranks(self):
        # 3 classes, 6 trials with manually chosen scores; ranks are deterministic.
        #
        # Trial | correct col | scores       | rank of correct
        #   0   |      0      | [3, 2, 1]    |   1   ✓top1, ✓top5
        #   1   |      1      | [3, 1, 2]    |   3   ✗top1, ✓top5
        #   2   |      2      | [3, 2, 1]    |   3   ✗top1, ✓top5
        #   3   |      0      | [2, 3, 1]    |   2   ✗top1, ✓top5, ✓top2
        #   4   |      1      | [1, 2, 3]    |   2   ✗top1, ✓top5, ✓top2
        #   5   |      2      | [3, 2, 1]    |   3   ✗top1, ✓top5, ✗top2
        sim = np.array([
            [3, 2, 1],
            [3, 1, 2],
            [3, 2, 1],
            [2, 3, 1],
            [1, 2, 3],
            [3, 2, 1],
        ], dtype=np.float32)
        labels = np.array([0, 1, 2, 0, 1, 2])

        _, count_1 = _label_retrieval_counts(sim, labels, k=1)
        count_2, _ = _label_retrieval_counts(sim, labels, k=2)
        count_5, _ = _label_retrieval_counts(sim, labels, k=5)

        assert count_1 == 1  # only trial 0
        assert count_2 == 3  # trials 0, 3, 4 have rank ≤ 2
        assert count_5 == 6  # all 6 have rank ≤ 5 (only 3 classes exist)

    def test_single_class_single_trial(self):
        # 1 class, 1 trial: rank is always 1.
        sim = np.array([[0.8]], dtype=np.float32)
        labels = np.array([0])
        count_k, count_1 = _label_retrieval_counts(sim, labels, k=1)
        assert count_1 == 1
        assert count_k == 1
