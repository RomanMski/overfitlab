"""Behaviour tests for combinatorially symmetric cross-validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stressfold import probability_of_backtest_overfitting as pbo


def _noise(n_periods=600, n_trials=20, seed=0, scale=0.01):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, size=(n_periods, n_trials))


def test_pbo_is_unbiased_under_the_null():
    """With no trial better than any other, selection carries no information.

    The in-sample winner's out-of-sample rank is then uniform, so PBO has to
    average one half. A single dataset is far too noisy to assert on, which is
    the point of averaging over independent ones here.
    """

    values = []
    for seed in range(12):
        matrix = _noise(n_periods=500, n_trials=20, seed=500 + seed)
        values.append(pbo(matrix, n_splits=10).pbo)
    mean = float(np.mean(values))

    assert 0.35 < mean < 0.65, f"null PBO averaged {mean:.3f}, expected about 0.5"


def test_a_persistent_edge_gives_low_pbo():
    matrix = _noise(seed=1)
    matrix[:, 3] += 0.0018  # a real edge, present throughout the history

    result = pbo(matrix, n_splits=12)

    assert result.pbo < 0.15
    # The winner should be the trial that actually carries the edge.
    assert int(np.bincount(result.selected_trial).argmax()) == 3


def test_cscv_is_invariant_to_the_order_of_the_blocks():
    """A documented limitation, asserted so it cannot regress silently.

    CSCV enumerates every way of choosing half the blocks, and that set of
    subsets does not change when the blocks are relabelled. The statistic is
    therefore blind to chronology by construction, which is a real constraint
    on what it can be used for rather than an implementation detail.
    """

    matrix = _noise(n_periods=600, n_trials=20, seed=2)
    reordered = np.concatenate(
        [matrix[index * 50 : (index + 1) * 50] for index in [11, 3, 7, 0, 9, 1, 5, 2, 10, 4, 8, 6]]
    )

    assert pbo(matrix, n_splits=12).pbo == pbo(reordered, n_splits=12).pbo


def test_a_regime_flip_is_not_what_cscv_measures():
    """An edge that reverses halfway is invisible to CSCV, near the null.

    This follows directly from the order invariance above. Detecting a change
    of regime needs a chronological protocol such as walk-forward, and reading
    a mid-range PBO as evidence against regime change would be a mistake.
    """

    matrix = _noise(n_periods=600, n_trials=20, seed=2)
    half = matrix.shape[0] // 2
    matrix[:half, 5] += 0.005
    matrix[half:, 5] -= 0.005

    result = pbo(matrix, n_splits=12)

    assert 0.35 < result.pbo < 0.65


def test_degradation_and_loss_are_reported():
    matrix = _noise(seed=3)
    result = pbo(matrix, n_splits=10)

    assert result.is_performance.shape == result.oos_performance.shape
    assert 0.0 <= result.probability_of_loss <= 1.0
    # In-sample selection beats out-of-sample reality under a null.
    assert result.mean_is() > result.mean_oos()


def test_accepts_a_dataframe_and_matches_the_array():
    matrix = _noise(seed=4)
    frame = pd.DataFrame(matrix, columns=[f"cfg_{i}" for i in range(matrix.shape[1])])

    assert pbo(frame, n_splits=10).pbo == pbo(matrix, n_splits=10).pbo


def test_is_deterministic():
    matrix = _noise(seed=5)
    assert pbo(matrix, n_splits=10).pbo == pbo(matrix, n_splits=10).pbo


def test_combinations_are_sampled_when_the_split_count_is_large():
    matrix = _noise(n_periods=900, n_trials=8, seed=6)

    result = pbo(matrix, n_splits=20, max_combinations=500, random_state=3)

    assert result.combinations_were_sampled is True
    assert result.n_combinations == 500
    # Sampling is seeded, so the estimate must still reproduce exactly.
    repeat = pbo(matrix, n_splits=20, max_combinations=500, random_state=3)
    assert result.pbo == repeat.pbo


def test_full_enumeration_is_flagged_as_such():
    result = pbo(_noise(seed=7), n_splits=10)
    assert result.combinations_were_sampled is False
    assert result.n_combinations == 252  # C(10, 5)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_splits": 11}, "even"),
        ({"n_splits": 2}, "at least 4"),
    ],
)
def test_split_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        pbo(_noise(seed=8), **kwargs)


def test_rejects_a_single_trial():
    with pytest.raises(ValueError, match="at least two trials"):
        pbo(_noise(n_trials=1, seed=9))


def test_rejects_missing_values():
    matrix = _noise(seed=10)
    matrix[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        pbo(matrix)


def test_rejects_a_history_too_short_for_the_blocks():
    with pytest.raises(ValueError, match="cannot be cut"):
        pbo(_noise(n_periods=20, seed=11), n_splits=16)
