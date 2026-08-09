"""Behaviour tests for synthetic path generation and the structure sweep."""

from __future__ import annotations

import numpy as np
import pytest

from overfitlab import (
    block_permutation,
    iid_bootstrap,
    moving_block_bootstrap,
    path_stress,
    stationary_bootstrap,
)
from overfitlab.synthetic import PathStressResult


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
    assert result.levels[0]["p_value"] < 0.1


def test_an_order_invariant_strategy_sits_in_the_middle_rather_than_anywhere():
    """Ties must be split, not resolved by floating point rounding.

    Buy and hold scores identically on every arrangement, so the synthetic
    scores differ from the real one only in the last bits of a sum taken in a
    different order. Counting strictly below made this report 23 on one real
    series and 11 on another. The only truthful answer is the middle.
    """

    market = ar1_market(1500, phi=0.35, seed=23)
    result = path_stress(
        lambda values: values, market, block_sizes=(1, 20), n_paths=200, seed=5
    )
    for level in result.levels:
        assert level["percentile"] == pytest.approx(50.0, abs=1e-9)
        # Every arrangement matches it, so nothing here is rare.
        assert level["p_value"] == pytest.approx(1.0, abs=1e-9)


def test_a_small_but_rare_effect_is_not_dismissed_by_the_summary():
    """Volatility targeting on SPY had dependence 0.21 and a p-value of 0.002.

    The ratio is small because the strategy is long most of the time and keeps
    that exposure under shuffling. Branching on the ratio alone printed that it
    was not earning anything from ordering, while the arrangements almost never
    reached it. Both numbers have to speak.
    """

    result = PathStressResult(
        observed_sharpe=0.0384,
        periods_per_year=252,
        n_paths=400,
        levels=(
            {
                "block_size": 1.0,
                "n_paths": 400.0,
                "median_sharpe": 0.0303,
                "median_annualised": 0.49,
                "p95_annualised": 0.58,
                "mean_annualised": 0.49,
                "percentile": 97.2,
                "p_value": 0.0025,
            },
        ),
    )
    assert result.structure_dependence() == pytest.approx(0.21, abs=0.01)
    assert result.shuffled_p_value() == pytest.approx(0.0025)
    text = result.summary_text()
    assert "the arrangements rarely match it" in text
    assert "not earning this from ordering" not in text


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
    assert "order only, marginals kept" in text
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


def test_non_finite_strategy_output_is_recorded_not_averaged_in():
    """A NaN path used to poison the median for its whole level.

    The path was counted as a success, so the level still reported a result and
    every quantile for it came back NaN.
    """

    market = ar1_market(400)
    calls = {"n": 0}

    def sometimes_nan(values: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        out = values.copy()
        if calls["n"] % 4 == 0:
            out[0] = np.nan
        return out

    result = path_stress(
        sometimes_nan, market, block_sizes=(1,), n_paths=40, seed=3
    )
    level = result.levels[0]

    assert np.isfinite(level["median_annualised"])
    assert np.isfinite(level["p95_annualised"])
    # The bad paths are dropped and reported rather than silently included.
    assert level["n_paths"] < 40
    assert any("non-finite" in message for message in result.errors)


def test_a_strategy_that_returns_nothing_usable_fails_loudly():
    market = ar1_market(400)
    with pytest.raises(ValueError, match="non-finite"):
        path_stress(lambda values: np.full(values.size, np.nan), market, n_paths=4)


def test_permutation_preserves_every_observation_exactly():
    """The documented claim is that only the ordering changes.

    That is true of a permutation and false of a bootstrap. Sampling with
    replacement drops some observations and duplicates others, so its paths
    have different moments from the source, and a strategy could fail on them
    for reasons unrelated to sequence.
    """

    from overfitlab import block_permutation

    rng = np.random.default_rng(0)
    market = rng.standard_t(df=3, size=900) * 0.01

    for block in (1, 5, 60):
        paths = block_permutation(market, 30, block_size=block, seed=2)
        for path in paths:
            assert np.array_equal(np.sort(path), np.sort(market))
        assert np.allclose(paths.mean(axis=1), market.mean())
        assert np.allclose(paths.std(axis=1, ddof=1), market.std(ddof=1))


def test_the_bootstrap_does_not_preserve_them_which_is_why_it_is_not_default():
    rng = np.random.default_rng(0)
    market = rng.standard_t(df=3, size=900) * 0.01
    paths = iid_bootstrap(market, 50, seed=2)

    # Some observations are dropped and others repeated.
    assert any(len(set(path)) < len(set(market)) for path in paths)
    # So the path means scatter around the source mean rather than matching it.
    assert paths.mean(axis=1).std() > 0


def test_default_scheme_is_the_permutation():
    from overfitlab import generate_datasets

    rng = np.random.default_rng(1)
    market = rng.normal(scale=0.01, size=600)
    datasets = generate_datasets(market, block_sizes=(1, 20), n_paths=10, seed=3)

    for paths in datasets.values():
        for path in paths:
            assert np.array_equal(np.sort(path), np.sort(market))


def test_permutation_still_destroys_ordering_at_block_one():
    from overfitlab import block_permutation

    market = ar1_market(3000, phi=0.4, seed=5)
    assert autocorrelation(market) > 0.25

    paths = block_permutation(market, 40, block_size=1, seed=7)
    resampled = np.array([autocorrelation(path) for path in paths])
    assert abs(resampled.mean()) < 0.05


def test_unknown_scheme_is_rejected():
    from overfitlab import generate_datasets

    with pytest.raises(ValueError, match="scheme must be"):
        generate_datasets(ar1_market(200), scheme="wishful")


def _fixture():
    import json
    import pathlib

    path = pathlib.Path(__file__).parents[1] / "fixtures" / "block-order.json"
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", _fixture(), ids=lambda case: case["name"])
def test_block_cutting_matches_the_shared_fixture(case):
    """One specification, checked by both languages.

    A cross language test catches the two implementations drifting apart. It
    does not catch them agreeing on the same wrong thing, which is how the
    bootstrap error survived in both. The fixture is written by hand for that
    reason rather than generated from either implementation.
    """

    from overfitlab import apply_block_order

    result = apply_block_order(case["source"], case["block_size"], case["order"])
    assert list(result) == pytest.approx(case["expected"])


@pytest.mark.parametrize(
    "n, block",
    [(100, 7), (100, 3), (101, 10), (13, 5), (9, 4), (8, 1), (50, 49), (12, 5)],
)
def test_multiset_survives_awkward_block_sizes(n, block):
    """The ragged final block is where this would break."""

    from overfitlab import block_permutation

    rng = np.random.default_rng(n * 31 + block)
    market = rng.normal(size=n)
    paths = block_permutation(market, 20, block_size=block, seed=3)

    assert paths.shape == (20, n)
    for path in paths:
        assert np.array_equal(np.sort(path), np.sort(market))


def test_repeated_values_keep_their_multiplicity():
    from overfitlab import block_permutation

    market = np.array([0.01, 0.01, -0.02, 0.01, -0.02, 0.03, 0.01, 0.0])
    paths = block_permutation(market, 20, block_size=3, seed=1)

    source_counts = np.unique(market, return_counts=True)
    for path in paths:
        assert np.array_equal(np.unique(path, return_counts=True)[1], source_counts[1])


def test_a_single_block_is_rejected_rather_than_silently_doing_nothing():
    from overfitlab import block_permutation

    market = np.arange(20, dtype=float)
    with pytest.raises(ValueError, match="single block"):
        block_permutation(market, 5, block_size=20)
    with pytest.raises(ValueError, match="single block"):
        block_permutation(market, 5, block_size=40)


def test_apply_block_order_rejects_a_bad_ordering():
    from overfitlab import apply_block_order

    with pytest.raises(ValueError, match="permutation of"):
        apply_block_order([1, 2, 3, 4, 5, 6], 2, [0, 0, 1])


def test_write_datasets_round_trips_to_disk(tmp_path):
    """This was broken by a rename and no test noticed, so now one does."""

    import csv
    import json

    from overfitlab import write_datasets

    rng = np.random.default_rng(0)
    market = rng.normal(scale=0.01, size=300)
    manifest = write_datasets(
        market, str(tmp_path), block_sizes=(1, 20), n_paths=6, seed=2
    )

    assert manifest["source_periods"] == 300
    assert manifest["scheme"] == "permutation"
    assert [entry["file"] for entry in manifest["files"]] == [
        "block-001.csv",
        "block-020.csv",
    ]

    written = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert written == manifest

    with (tmp_path / "block-001.csv").open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    assert header == [f"path_{index + 1}" for index in range(6)]
    assert len(body) == 300

    # Every written column has to be a rearrangement of the source.
    columns = np.array([[float(cell) for cell in row] for row in body]).T
    for column in columns:
        assert np.allclose(np.sort(column), np.sort(market))


def test_write_datasets_rejects_an_unknown_scheme(tmp_path):
    from overfitlab import write_datasets

    with pytest.raises(ValueError, match="scheme must be"):
        write_datasets(np.zeros(50) + 0.01, str(tmp_path), scheme="wishful")


def test_a_multiset_only_strategy_is_exactly_invariant():
    """The sharpest available check that the generator only reorders.

    Buy and hold depends solely on the multiset of returns, so a permutation
    cannot change its result. Under a bootstrap it moved, which is what
    revealed the original error. Any deviation beyond floating point here means
    the generator is doing something other than reordering.
    """

    rng = np.random.default_rng(1)
    market = rng.normal(loc=0.0004, scale=0.011, size=1500)

    result = path_stress(
        always_long, market, block_sizes=(1, 5, 20, 60), n_paths=100, seed=4
    )

    for level in result.levels:
        assert level["median_annualised"] == pytest.approx(
            result.observed_annualised, abs=1e-12
        )
    assert result.structure_dependence() == pytest.approx(0.0, abs=1e-12)


def trend_with_positions(market: np.ndarray):
    """Reports its exposure as well as its returns, so costs can be charged."""

    positions = np.sign(market[:-1])
    return positions * market[1:], positions


def hold_with_positions(market: np.ndarray):
    return market, np.ones(market.size)


def test_zero_cost_leaves_the_result_untouched():
    market = ar1_market(1200, phi=0.3, seed=2)
    free = path_stress(trend_with_positions, market, block_sizes=(1,), n_paths=40, seed=1)
    charged = path_stress(
        trend_with_positions, market, block_sizes=(1,), n_paths=40, seed=1, cost_bps=0.0
    )
    assert free.observed_sharpe == charged.observed_sharpe


def test_costs_only_bite_where_there_is_turnover():
    """Buy and hold trades once and then never again."""

    rng = np.random.default_rng(5)
    market = rng.normal(loc=0.0005, scale=0.01, size=1500)

    free = path_stress(hold_with_positions, market, block_sizes=(1,), n_paths=30, seed=2)
    charged = path_stress(
        hold_with_positions, market, block_sizes=(1,), n_paths=30, seed=2, cost_bps=25.0
    )
    # One entry from flat, spread over 1500 periods, is close to nothing.
    assert charged.observed_annualised == pytest.approx(free.observed_annualised, abs=0.02)


def test_costs_erode_a_strategy_that_trades_constantly():
    market = ar1_market(2000, phi=0.3, seed=7)

    free = path_stress(trend_with_positions, market, block_sizes=(1,), n_paths=60, seed=3)
    charged = path_stress(
        trend_with_positions, market, block_sizes=(1,), n_paths=60, seed=3, cost_bps=10.0
    )
    assert charged.observed_annualised < free.observed_annualised
    # And the shuffled markets get worse, because there the costs buy nothing.
    assert charged.levels[0]["median_annualised"] < free.levels[0]["median_annualised"]


def test_asking_for_costs_without_positions_is_an_error():
    """Silently reporting a gross number as though it were net would be worse."""

    market = ar1_market(400)
    with pytest.raises(ValueError, match="only its returns"):
        path_stress(momentum_strategy, market, block_sizes=(1,), n_paths=5, cost_bps=5.0)


def test_apply_costs_validation():
    from overfitlab import apply_costs

    with pytest.raises(ValueError, match="same periods"):
        apply_costs([0.01, 0.02, 0.03], [1.0, 1.0], 5.0)
    with pytest.raises(ValueError, match="non-negative"):
        apply_costs([0.01, 0.02], [1.0, 1.0], -1.0)


def test_apply_costs_charges_each_change_in_position():
    from overfitlab import apply_costs

    returns = np.zeros(4)
    positions = np.array([1.0, 1.0, -1.0, 0.0])
    # Traded amounts: 1 to open, 0, 2 to flip, 1 to close.
    net = apply_costs(returns, positions, 100.0)  # 100 bps = 1%
    assert net == pytest.approx([-0.01, 0.0, -0.02, -0.01])


def test_the_bootstrap_injects_variance_into_a_deterministic_quantity():
    """The argument the paper actually makes, asserted.

    Buy and hold's Sharpe is a function of the multiset of returns, so under a
    permutation it is deterministic. The bootstrap is not biased here, its mean
    sits within a couple of standard errors of the source. What it does is
    scatter a quantity that cannot move, by an amount comparable to the
    quantity itself, and any strategy tested that way inherits that noise.
    """

    rng = np.random.default_rng(1)
    market = rng.normal(loc=0.0004, scale=0.011, size=2500)
    source = market.mean() / market.std(ddof=1)

    def sharpes(paths):
        return paths.mean(axis=1) / paths.std(axis=1, ddof=1)

    permuted = sharpes(block_permutation(market, 2000, block_size=1, seed=2))
    bootstrapped = sharpes(iid_bootstrap(market, 2000, seed=2))

    # Deterministic under permutation.
    assert permuted.std() < 1e-12
    assert permuted.mean() == pytest.approx(source, abs=1e-12)

    # Not biased under the bootstrap, within three standard errors.
    standard_error = bootstrapped.std() / np.sqrt(bootstrapped.size)
    assert abs(bootstrapped.mean() - source) < 3 * standard_error

    # But scattered by an amount comparable to the quantity itself.
    assert bootstrapped.std() > 0.5 * abs(source)
