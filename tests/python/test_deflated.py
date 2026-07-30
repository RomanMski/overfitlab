"""Behaviour tests for the probabilistic and deflated Sharpe ratios."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from stressfold import (
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


def test_probabilistic_sharpe_matches_a_hand_computation():
    """Pin the formula against a value worked out by hand.

    With SR = 0.1, T = 101, no skew and normal kurtosis the variance term is
    1 + (2 / 4) * 0.01 = 1.005, and the statistic is 0.1 * sqrt(100) / sqrt(1.005).
    """

    expected = norm.cdf(1.0 / math.sqrt(1.005))
    got = probabilistic_sharpe_ratio(0.1, n_observations=101)

    assert got == pytest.approx(expected, abs=1e-12)


def test_probabilistic_sharpe_rises_with_sample_length():
    short = probabilistic_sharpe_ratio(0.1, n_observations=50)
    long = probabilistic_sharpe_ratio(0.1, n_observations=5_000)
    assert long > short


def test_negative_skew_and_fat_tails_reduce_confidence():
    clean = probabilistic_sharpe_ratio(0.15, n_observations=500)
    ugly = probabilistic_sharpe_ratio(
        0.15, n_observations=500, skewness=-1.5, kurtosis_=9.0
    )
    assert ugly < clean


def test_impossible_moment_combinations_return_nan_rather_than_a_number():
    got = probabilistic_sharpe_ratio(
        5.0, n_observations=100, skewness=10.0, kurtosis_=1.0
    )
    assert math.isnan(got)


def test_expected_maximum_rises_with_the_number_of_trials():
    bars = [
        expected_maximum_sharpe(n_trials=n, sharpe_variance=0.01)
        for n in (2, 10, 100, 1_000, 10_000)
    ]
    assert bars == sorted(bars)
    assert all(bar > 0 for bar in bars)


def test_a_single_trial_has_no_selection_bias_to_remove():
    assert expected_maximum_sharpe(n_trials=1, sharpe_variance=0.25) == 0.0


def test_identical_trials_leave_nothing_to_deflate():
    """Zero spread across trials means the maximum carries no inflation."""

    assert expected_maximum_sharpe(n_trials=500, sharpe_variance=0.0) == 0.0


def test_deflating_one_trial_equals_the_plain_probabilistic_ratio():
    rng = np.random.default_rng(0)
    series = rng.normal(0.0006, 0.01, size=800)

    result = deflated_sharpe_ratio(series)

    assert result.n_trials == 1
    assert result.benchmark_sharpe == 0.0
    assert result.deflated_sharpe == pytest.approx(result.probabilistic_sharpe)


def test_more_trials_deflate_the_same_series_further():
    """Hold the audited series fixed and vary only how much was searched."""

    rng = np.random.default_rng(1)
    periods = 1_000
    edge = rng.normal(0.0006, 0.01, size=periods)

    scores = []
    for extra in (4, 49, 499):
        noise = rng.normal(0.0, 0.01, size=(periods, extra))
        matrix = np.column_stack([edge, noise])
        result = deflated_sharpe_ratio(matrix, selected=0)
        scores.append(result.deflated_sharpe)

    assert scores == sorted(scores, reverse=True)


def test_a_large_search_over_noise_is_not_convincing():
    """The headline case: an impressive Sharpe produced by searching alone."""

    rng = np.random.default_rng(2)
    noise = rng.normal(0.0, 0.01, size=(1_000, 200))

    result = deflated_sharpe_ratio(noise, periods_per_year=252)

    # The winner looks strong on an annualised basis and ignoring selection.
    assert result.annualised_sharpe > 1.0
    assert result.probabilistic_sharpe > 0.95
    # Accounting for the search, it is close to worthless.
    assert result.deflated_sharpe < 0.8
    assert result.benchmark_sharpe > 0.0


def test_a_genuine_edge_survives_a_modest_search():
    rng = np.random.default_rng(3)
    periods = 2_000
    edge = rng.normal(0.0025, 0.01, size=periods)
    noise = rng.normal(0.0, 0.01, size=(periods, 19))

    result = deflated_sharpe_ratio(np.column_stack([edge, noise]), selected=0)

    assert result.deflated_sharpe > 0.95


def test_annualisation_is_reporting_only():
    rng = np.random.default_rng(4)
    series = rng.normal(0.0005, 0.01, size=500)

    plain = deflated_sharpe_ratio(series)
    annual = deflated_sharpe_ratio(series, periods_per_year=252)

    assert plain.deflated_sharpe == annual.deflated_sharpe
    assert plain.annualised_sharpe is None
    assert annual.annualised_sharpe == pytest.approx(
        annual.observed_sharpe * math.sqrt(252)
    )


def test_sharpe_of_a_constant_series_is_zero_not_infinite():
    assert sharpe_ratio(np.full(100, 0.01)) == 0.0


def test_rejects_missing_values():
    values = np.zeros((100, 3))
    values[5, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        deflated_sharpe_ratio(values)
