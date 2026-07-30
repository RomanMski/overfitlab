"""Behaviour tests for synthetic path generation and the structure sweep."""

from __future__ import annotations

import numpy as np
import pytest

from stressfold import (
    iid_bootstrap,
    moving_block_bootstrap,
    path_stress,
    stationary_bootstrap,
)


def autocorrelation(values: np.ndarray, lag: int = 1) -> float:
    centred = values - values.mean()
    denominator = float(np.sum(centred**2))
    if denominator == 0:
        return 0.0
    return float(np.sum(centred[lag:] * centred[:-lag]) / denominator)


def ar1_market(n: int = 1500, phi: float = 0.35, seed: int = 0) -> np.ndarray:
    """A market with genuine short-horizon momentum to exploit."""

    rng = np.random.default_rng(seed)
    shocks = rng.normal(scale=0.01, size=n)
    series = np.empty(n)
    series[0] = shocks[0]
    for i in range(1, n):
        series[i] = phi * series[i - 1] + shocks[i]
    return series


def momentum_strategy(market: np.ndarray) -> np.ndarray:
    """Go with yesterday's direction. Profitable only if returns persist."""

    return np.sign(market[:-1]) * market[1:]


def always_long(market: np.ndarray) -> np.ndarray:
    return market


# -- generators ------------------------------------------------------------


@pytest.mark.parametrize(
    "generator",
    [
        lambda data: iid_bootstrap(data, 6, seed=1),
        lambda data: moving_block_bootstrap(data, 6, block_size=10, seed=1),
        lambda data: stationary_bootstrap(data, 6, expected_block=10.0, seed=1),
    ],
)
def test_generators_preserve_shape_and_draw_only_observed_values(generator):
    data = ar1_market(400)
    paths = generator(data)

    assert paths.shape == (6, data.size)
    # Resampling reorders observations, it never invents new ones.
    assert np.isin(paths, data).all()


def test_generators_are_deterministic_for_a_fixed_seed():
    data = ar1_market(300)
    assert np.array_equal(
        stationary_bootstrap(data, 4, expected_block=8.0, seed=3),
        stationary_bootstrap(data, 4, expected_block=8.0, seed=3),
    )
    assert not np.array_equal(
        stationary_bootstrap(data, 4, expected_block=8.0, seed=3),
        stationary_bootstrap(data, 4, expected_block=8.0, seed=4),
    )


def test_iid_bootstrap_destroys_autocorrelation():
    data = ar1_market(4000, phi=0.4, seed=2)
    assert autocorrelation(data) > 0.25

    paths = iid_bootstrap(data, 40, seed=5)
    resampled = np.array([autocorrelation(path) for path in paths])
    assert abs(resampled.mean()) < 0.05


def test_block_bootstraps_retain_most_of_the_autocorrelation():
    data = ar1_market(4000, phi=0.4, seed=2)
    original = autocorrelation(data)

    for paths in (
        moving_block_bootstrap(data, 40, block_size=40, seed=5),
        stationary_bootstrap(data, 40, expected_block=40.0, seed=5),
    ):
        resampled = np.array([autocorrelation(path) for path in paths])
        # Some dependence is lost at the joins, so this is a floor not equality.
        assert resampled.mean() > 0.6 * original


def test_generator_validation():
    data = ar1_market(100)
    with pytest.raises(ValueError, match="n_paths"):
        iid_bootstrap(data, 0)
    with pytest.raises(ValueError, match="block_size"):
        moving_block_bootstrap(data, 2, block_size=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        moving_block_bootstrap(data, 2, block_size=500)
    with pytest.raises(ValueError, match="expected_block"):
        stationary_bootstrap(data, 2, expected_block=0.5)
    with pytest.raises(ValueError, match="at least 8"):
        iid_bootstrap([0.1, 0.2], 2)
    with pytest.raises(ValueError, match="non-finite"):
        iid_bootstrap([0.1] * 10 + [np.nan], 2)
    with pytest.raises(ValueError, match="seed"):
        iid_bootstrap(data, 2, seed=-1)


# -- the sweep -------------------------------------------------------------


def test_a_real_timing_edge_collapses_once_ordering_is_destroyed():
    market = ar1_market(2000, phi=0.35, seed=7)
    result = path_stress(
        momentum_strategy,
        market,
        block_sizes=(1, 40),
        n_paths=120,
        seed=11,
    )

    shuffled, blocked = result.levels[0], result.levels[1]
    assert shuffled["block_size"] == 1.0

    # The edge exists on the real path.
    assert result.observed_annualised > 1.0
    # It disappears when the momentum it trades is shuffled away.
    assert shuffled["median_annualised"] < 0.3 * result.observed_annualised
    # And largely returns once runs are kept intact.
    assert blocked["median_annualised"] > shuffled["median_annualised"]
    # Which is exactly what high structure dependence means.
    assert result.structure_dependence() > 0.7
    assert not result.errors


def test_a_long_only_result_survives_shuffling_and_is_flagged():
    """Buy and hold earns from drift, not from timing, so shuffling changes nothing."""

    rng = np.random.default_rng(3)
    market = rng.normal(loc=0.0006, scale=0.01, size=2000)
    result = path_stress(
        always_long, market, block_sizes=(1, 40), n_paths=120, seed=13
    )

    shuffled = result.levels[0]
    assert result.observed_annualised > 0.5
    # Reordering cannot change a mean or a standard deviation.
    assert shuffled["median_annualised"] == pytest.approx(
        result.observed_annualised, rel=0.15
    )
    # So the sweep correctly reports that none of this comes from timing.
    assert result.structure_dependence() < 0.3


def test_percentile_places_the_real_result_in_the_synthetic_distribution():
    market = ar1_market(1500, phi=0.35, seed=17)
    result = path_stress(
        momentum_strategy, market, block_sizes=(1,), n_paths=200, seed=19
    )
    # A genuine edge should beat almost every shuffled market.
    assert result.levels[0]["percentile"] > 90.0


def test_failing_strategies_are_recorded_rather_than_swallowed():
    market = ar1_market(200)
    calls = {"n": 0}

    def flaky(values: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise RuntimeError("boom")
        return values

    result = path_stress(flaky, market, block_sizes=(1,), n_paths=30, seed=2)
    assert result.errors
    assert any("boom" in message for message in result.errors)
    # The surviving paths still produce a level.
    assert result.levels[0]["n_paths"] < 30


def test_summary_text_reports_the_gradient():
    market = ar1_market(1200, seed=23)
    result = path_stress(
        momentum_strategy, market, block_sizes=(1, 20), n_paths=60, seed=29
    )
    text = result.summary_text()

    assert "Observed Sharpe" in text
    assert "nothing, pure noise" in text
    assert "runs of 20 periods" in text
    assert "Structure dependence" in text


def test_sweep_validation():
    market = ar1_market(100)
    with pytest.raises(ValueError, match="n_paths"):
        path_stress(always_long, market, n_paths=0)
    with pytest.raises(ValueError, match="periods_per_year"):
        path_stress(always_long, market, periods_per_year=0)
    with pytest.raises(ValueError, match="block_sizes"):
        path_stress(always_long, market, block_sizes=(0,))
