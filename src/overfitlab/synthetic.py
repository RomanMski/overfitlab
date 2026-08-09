"""Alternative histories, and what a strategy does on markets that never happened.

A backtest is one draw. The strategy was tuned on the one price path that
history handed you, and the usual defence, a train and test split, still cuts
up that same single path.

This module builds new paths from the one you have and reruns the strategy on
each. The interesting control is *how much structure the resampling keeps*:

``block_size = 1``
    A plain permutation. Every observation appears exactly once, so the mean,
    the variance, the skew and every extreme value are identical to the source
    and only the arrangement changes. No autocorrelation, no momentum, no mean
    reversion and no volatility clustering survives. This is not a noise
    series. It is the same returns in a different order.

    The default scheme permutes rather than resamples for exactly this reason.
    A bootstrap draws with replacement, so its paths drop some observations and
    duplicate others, and their moments differ from the source. A strategy
    could then fail on them for reasons unrelated to sequence. The bootstrap
    functions remain available for anyone who wants bootstrap inference.

``block_size = 20``
    Runs of twenty periods are kept intact and only their order is shuffled.
    Anything the strategy exploits over a horizon shorter than a block mostly
    survives.

Sweeping the block size therefore tells you *what* the strategy depends on,
rather than only whether it works. That gradient is the point of this module.

The generators are the standard resampling schemes for dependent data. The
default is a block permutation rather than one of them, because this module
asks whether ordering matters and a permutation is the only scheme that holds
the marginal distribution exactly fixed while the ordering changes. That is a
trade rather than a free improvement. The geometric block lengths of the
stationary bootstrap of Politis and Romano (1994) leave the resampled series
stationary, which permuting fixed blocks does not, so for interval estimation
the ranking reverses. Both are implemented and ``scheme`` selects between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "PathStressResult",
    "iid_bootstrap",
    "moving_block_bootstrap",
    "path_stress",
    "stationary_bootstrap",
]

Strategy = Callable[[np.ndarray], np.ndarray]


def _as_returns(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size < 8:
        raise ValueError(f"{name} needs at least 8 observations, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _rng(seed: int) -> np.random.Generator:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return np.random.default_rng(seed)


def iid_bootstrap(
    returns: Sequence[float] | np.ndarray, n_paths: int, *, seed: int = 0
) -> np.ndarray:
    """Resample observations independently, destroying every ordering.

    This is the strongest null available here. Nothing a timing rule could act
    on survives it.
    """

    data = _as_returns(returns, name="returns")
    generator = _rng(seed)
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1")
    picks = generator.integers(0, data.size, size=(n_paths, data.size))
    return data[picks]


def moving_block_bootstrap(
    returns: Sequence[float] | np.ndarray,
    n_paths: int,
    *,
    block_size: int,
    seed: int = 0,
) -> np.ndarray:
    """Resample overlapping blocks of fixed length, keeping their contents intact."""

    data = _as_returns(returns, name="returns")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    if block_size > data.size:
        raise ValueError("block_size cannot exceed the number of observations")
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1")

    generator = _rng(seed)
    n_blocks = int(np.ceil(data.size / block_size))
    starts = generator.integers(0, data.size - block_size + 1, size=(n_paths, n_blocks))
    offsets = np.arange(block_size)
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(n_paths, -1)
    return data[indices[:, : data.size]]


def stationary_bootstrap(
    returns: Sequence[float] | np.ndarray,
    n_paths: int,
    *,
    expected_block: float,
    seed: int = 0,
) -> np.ndarray:
    """Politis and Romano's stationary bootstrap.

    Block lengths are geometric with mean ``expected_block``, and blocks wrap
    around the end of the series. Randomising the length is what keeps the
    resampled series stationary, which a fixed block length does not.
    """

    data = _as_returns(returns, name="returns")
    if not np.isfinite(expected_block) or expected_block < 1:
        raise ValueError("expected_block must be at least 1")
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1")

    generator = _rng(seed)
    n = data.size
    restart_probability = 1.0 / float(expected_block)
    indices = np.empty((n_paths, n), dtype=np.int64)
    indices[:, 0] = generator.integers(0, n, size=n_paths)
    restarts = generator.random((n_paths, n)) < restart_probability
    fresh = generator.integers(0, n, size=(n_paths, n))
    for step in range(1, n):
        carried = (indices[:, step - 1] + 1) % n
        indices[:, step] = np.where(restarts[:, step], fresh[:, step], carried)
    return data[indices]


def apply_block_order(
    returns: Sequence[float] | np.ndarray,
    block_size: int,
    order: Sequence[int],
) -> np.ndarray:
    """Cut into consecutive blocks and concatenate them in ``order``.

    Separated from the random draw so the block arithmetic can be checked
    against a fixed expected output, including the ragged final block, without
    a generator in the way. The browser implements the same function and both
    are tested against one shared fixture.
    """

    # Deliberately not _as_returns. This is a mechanical rearrangement rather
    # than a statistical procedure, so the minimum length that a return series
    # needs does not apply and the fixture cases are short on purpose.
    data = np.asarray(returns, dtype=float).reshape(-1)
    if data.size == 0:
        raise ValueError("returns is empty")
    if not np.all(np.isfinite(data)):
        raise ValueError("returns contains non-finite values")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    starts = list(range(0, data.size, block_size))
    if sorted(order) != list(range(len(starts))):
        raise ValueError(
            f"order must be a permutation of 0..{len(starts) - 1}, got {list(order)}"
        )
    return np.concatenate([data[s : s + block_size] for s in (starts[i] for i in order)])


def block_permutation(
    returns: Sequence[float] | np.ndarray,
    n_paths: int,
    *,
    block_size: int,
    seed: int = 0,
) -> np.ndarray:
    """Cut the series into consecutive blocks and permute their order.

    Unlike the bootstrap schemes this samples without replacement, so every
    observation appears exactly once in every generated path. The multiset of
    returns is identical to the source, which means the mean, the variance, the
    skewness and every extreme value are preserved exactly rather than in
    expectation. Only the arrangement changes.

    That is what makes it the right tool for asking whether a result depends on
    ordering. A bootstrap answers a different question and its paths have
    different moments from the source, so a strategy could fail on them for
    reasons that have nothing to do with sequence.

    ``block_size = 1`` is a plain permutation of the observations. A trailing
    partial block is kept whole and permuted with the rest, so nothing is
    dropped and nothing is duplicated.

    The number of blocks is ``ceil(n / block_size)``, and it controls how much
    stress the result carries. Few blocks means few possible arrangements and
    most local dependence surviving, so a large block size is close to a copy
    of the source and should be read as the low stress end of a sweep rather
    than as a test in its own right.

    Preserving the multiplicity of every observation is an exact statement
    about the data. Empirical statistics that do not depend on order are
    therefore mathematically unchanged, though a recomputed mean or standard
    deviation can differ in the last bits because floating point addition is
    not associative and the summation order changes.
    """

    data = _as_returns(returns, name="returns")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    if block_size >= data.size:
        raise ValueError(
            f"block_size {block_size} leaves a single block of {data.size} "
            "observations, so every generated path would be a copy of the "
            "source. Use a block size smaller than the series."
        )
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1")

    generator = _rng(seed)
    edges = list(range(0, data.size, block_size))
    blocks = [data[start : start + block_size] for start in edges]

    out = np.empty((n_paths, data.size), dtype=float)
    for path in range(n_paths):
        out[path] = np.concatenate(
            [blocks[index] for index in generator.permutation(len(blocks))]
        )
    return out

# Indicative round trip costs in basis points of notional traded, for setting a
# starting value. These are ranges seen in public venue schedules and execution
# studies, not a quote for your account. Spread, commission, fees and market
# impact all vary by venue, size, time of day and how you route. Use your own
# fills if you have them. If you do not, run the sweep at two or three values
# and see whether the answer changes rather than trusting any single number.
INDICATIVE_COST_BPS = {
    "equity_large_cap": 2.0,
    "equity_small_cap": 20.0,
    "futures_liquid": 1.0,
    "crypto_major": 10.0,
    "crypto_alt": 40.0,
    "government_bond": 5.0,
    "corporate_bond": 50.0,
}


def apply_costs(
    returns: Sequence[float] | np.ndarray,
    positions: Sequence[float] | np.ndarray,
    cost_bps: float,
) -> np.ndarray:
    """Charge ``cost_bps`` of notional on every change in position.

    ``positions`` is the exposure held during each period, so the traded amount
    in period ``t`` is ``abs(positions[t] - positions[t - 1])``. The first
    period is charged for opening the initial position from flat.

    This is a linear cost model. It is proportional to size and it ignores
    market impact, so it understates the cost of a large order and of anything
    trading illiquid instruments. It is a floor rather than an estimate.
    """

    net = np.asarray(returns, dtype=float).reshape(-1)
    held = np.asarray(positions, dtype=float).reshape(-1)
    if held.size != net.size:
        raise ValueError(
            f"positions has {held.size} entries and returns has {net.size}; "
            "they must describe the same periods"
        )
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be finite and non-negative")

    traded = np.abs(np.diff(held, prepend=0.0))
    return net - traded * (cost_bps / 10_000.0)


def _sharpe(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    deviation = float(np.std(values, ddof=1))
    if deviation <= 1e-15 * max(float(np.max(np.abs(values))), 1.0):
        return 0.0
    return float(np.mean(values) / deviation)


@dataclass(frozen=True, slots=True)
class PathStressResult:
    """What a strategy did on markets that never happened."""

    observed_sharpe: float
    periods_per_year: int
    n_paths: int
    levels: tuple[dict[str, float], ...]
    errors: tuple[str, ...] = field(default=())

    @property
    def observed_annualised(self) -> float:
        return self.observed_sharpe * float(np.sqrt(self.periods_per_year))

    def frame(self):
        """Return the per-level summary as a pandas DataFrame."""

        import pandas as pd

        return pd.DataFrame(list(self.levels))

    def structure_dependence(self) -> float:
        """How much of the result needs market structure to exist.

        One means the result collapses entirely once ordering is destroyed.
        Zero means it survives shuffling untouched, so whatever it earns comes
        from the marginal distribution rather than from the arrangement.

        A high value says the result depends on the ordering of this series
        under this resampling scheme. It does not by itself establish a timing
        edge, because volatility targeting, path-dependent sizing, lookback
        warm-up and ordinary backtest bugs all produce the same signature.
        Treat it as descriptive. It is unstable when the observed Sharpe is
        near zero and it carries no confidence interval.
        """

        if not self.levels:
            return float("nan")
        shuffled = min(self.levels, key=lambda level: level["block_size"])
        if abs(self.observed_sharpe) < 1e-12:
            return float("nan")
        return float(
            1.0 - shuffled["median_sharpe"] / self.observed_sharpe
        )

    def shuffled_p_value(self) -> float:
        """How often the fully shuffled markets matched the real result.

        The companion to structure dependence rather than a substitute for it.
        Dependence measures how large the gap is and this measures how often
        chance closes it, and the two can disagree. A strategy that is long
        most of the time keeps most of its Sharpe under shuffling, so its
        dependence is small, while still beating almost every arrangement.
        """

        if not self.levels:
            return float("nan")
        shuffled = min(self.levels, key=lambda level: level["block_size"])
        return float(shuffled.get("p_value", float("nan")))

    def summary_text(self) -> str:
        lines = [
            f"Observed Sharpe {self.observed_annualised:.3f} annualised,"
            f" from {self.n_paths} synthetic paths per level",
            "",
            "  block  keeps                       median  p95     beats   p",
        ]
        for level in self.levels:
            block = int(level["block_size"])
            keeps = (
                "order only, marginals kept"
                if block == 1
                else f"runs of {block} periods"
            )
            lines.append(
                f"  {block:>5}  {keeps:<26}"
                f"  {level['median_annualised']:>6.2f}"
                f"  {level['p95_annualised']:>6.2f}"
                f"  {level['percentile']:>5.1f}%"
                f"  {level['p_value']:>5.3f}"
            )
        dependence = self.structure_dependence()
        p_value = self.shuffled_p_value()
        lines += [
            "",
            f"Structure dependence {dependence:.2f}, shuffled p-value {p_value:.3f}",
        ]
        # These two answer different questions and a strategy can score low on
        # one and high on the other, so neither is allowed to speak alone.
        large = dependence >= 0.5
        rare = p_value <= 0.05
        if large and rare:
            lines.append(
                "  Most of the result disappears once ordering is destroyed and"
                " the arrangements rarely match it. Consistent with a timing"
                " edge, and also with volatility targeting, path-dependent"
                " sizing or a lookback bug."
            )
        elif rare:
            lines.append(
                "  The strategy keeps most of its result under shuffling, so"
                " the ordering is not where the size of it comes from, but"
                " the arrangements rarely match it. A small edge on top of"
                " exposure the shuffling cannot remove looks like this."
            )
        elif large:
            lines.append(
                "  The gap is wide but the arrangements reach it often enough"
                " that chance is not ruled out. Usually a Sharpe near zero,"
                " where dependence is a ratio of two small numbers."
            )
        else:
            lines.append(
                "  The strategy does nearly as well on shuffled returns and the"
                " arrangements match it often, so it is not earning this from"
                " ordering."
            )
        return "\n".join(lines)


def _strategy_returns(
    strategy: Strategy, series: np.ndarray, cost_bps: float
) -> np.ndarray:
    """Run the strategy and charge costs if it reported its positions.

    A strategy may return just its returns, or a ``(returns, positions)`` pair.
    Costs need turnover, and turnover needs positions, so a strategy that only
    reports returns cannot be charged. Asking for a cost without positions
    raises rather than silently reporting a gross number as though it were net.
    """

    produced = strategy(series)
    if isinstance(produced, tuple):
        raw, positions = produced
        net = np.asarray(raw, dtype=float).reshape(-1)
        if cost_bps > 0:
            net = apply_costs(net, positions, cost_bps)
        return net

    net = np.asarray(produced, dtype=float).reshape(-1)
    if cost_bps > 0:
        raise ValueError(
            "cost_bps was set but the strategy returned only its returns. "
            "Costs are charged on changes in position, so return a "
            "(returns, positions) pair to use them."
        )
    return net


def _rank_statistics(values: np.ndarray, observed: float) -> tuple[float, float]:
    """Locate the real result inside the synthetic distribution.

    Ties are the reason this is not one line. A strategy whose result does not
    depend on ordering scores the same on every generated path, and those
    scores then differ from the real one only in the last bits, because the
    returns are summed in a different order. Counting strictly below turns that
    rounding into a percentile anywhere between 0 and 100, so buy and hold on
    real data reported 23 on one series and 11 on another when the only
    truthful answer is 50. Anything within a tolerance of the real result is
    treated as the tie it is and split evenly.

    The p-value is the usual conservative permutation form. The real series is
    itself one of the arrangements being compared, which is what the extra
    count in the numerator and denominator accounts for.
    """

    tolerance = 1e-9 * max(abs(observed), 1.0)
    below = int(np.count_nonzero(values < observed - tolerance))
    tied = int(np.count_nonzero(np.abs(values - observed) <= tolerance))
    percentile = 100.0 * (below + 0.5 * tied) / values.size
    at_least = int(np.count_nonzero(values >= observed - tolerance))
    p_value = (1.0 + at_least) / (values.size + 1.0)
    return percentile, p_value


def path_stress(
    strategy: Strategy,
    market_returns: Sequence[float] | np.ndarray,
    *,
    block_sizes: Sequence[int] = (1, 5, 20, 60),
    n_paths: int = 200,
    periods_per_year: int = 252,
    seed: int = 0,
    cost_bps: float = 0.0,
) -> PathStressResult:
    """Rerun ``strategy`` on synthetic markets built from ``market_returns``.

    ``strategy`` maps a market return series to the returns it would have
    earned on that series. It is called once on the real history and then once
    per synthetic path, so it must be deterministic and reasonably fast.

    The block sizes are swept from most destructive to least. Reading the
    resulting gradient is the point: a timing edge should die at
    ``block_size = 1`` and recover as blocks lengthen. A result that survives
    the shuffle was never about timing.

    This always permutes and takes no ``scheme``. It used to accept one, which
    was a mistake. The whole interpretation offered here rests on the marginal
    distribution being held fixed while the ordering varies, and the bootstrap
    schemes do not hold it fixed, so a caller could ask for one and receive a
    percentile, a p-value and a structure dependence that all read as though it
    had. Nothing distinguished the two in the output. The bootstraps remain
    exported as :func:`iid_bootstrap`, :func:`moving_block_bootstrap` and
    :func:`stationary_bootstrap` for interval estimation, which is what they
    are for, and :func:`generate_datasets` still takes a ``scheme`` because it
    scores nothing and labels what it produced.
    """

    data = _as_returns(market_returns, name="market_returns")
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be at least 1")
    blocks = sorted({int(size) for size in block_sizes})
    if not blocks or blocks[0] < 1:
        raise ValueError("block_sizes must all be at least 1")

    observed_returns = _strategy_returns(strategy, data, cost_bps)
    if observed_returns.size < 2 or not np.all(np.isfinite(observed_returns)):
        raise ValueError(
            "the strategy returned too few or non-finite values on the real "
            "series, so there is nothing to compare the synthetic paths against"
        )
    observed = _sharpe(observed_returns)
    annualiser = float(np.sqrt(periods_per_year))
    errors: list[str] = []
    levels: list[dict[str, float]] = []

    for position, block in enumerate(blocks):
        paths = generate_datasets(
            data,
            block_sizes=(block,),
            n_paths=n_paths,
            seed=seed * 7919 + position,
            scheme="permutation",
        )[block]

        scores: list[float] = []
        for index in range(paths.shape[0]):
            try:
                result = _strategy_returns(strategy, paths[index], cost_bps)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                errors.append(f"block {block} path {index}: {type(exc).__name__}: {exc}")
                continue
            # A strategy that returns NaN would otherwise poison the median and
            # every quantile for this level, and the level would still be
            # reported as though it had succeeded.
            if result.size < 2:
                errors.append(f"block {block} path {index}: returned {result.size} values")
                continue
            if not np.all(np.isfinite(result)):
                bad = int(np.count_nonzero(~np.isfinite(result)))
                errors.append(
                    f"block {block} path {index}: {bad} non-finite values returned"
                )
                continue
            scores.append(_sharpe(result))
        if not scores:
            errors.append(f"block {block}: every path failed")
            continue

        values = np.asarray(scores, dtype=float)
        percentile, p_value = _rank_statistics(values, observed)
        levels.append(
            {
                "block_size": float(block),
                "n_paths": float(values.size),
                "median_sharpe": float(np.median(values)),
                "median_annualised": float(np.median(values) * annualiser),
                "p95_annualised": float(np.quantile(values, 0.95) * annualiser),
                "mean_annualised": float(np.mean(values) * annualiser),
                # How much of the synthetic distribution the real result beats.
                "percentile": percentile,
                "p_value": p_value,
            }
        )

    return PathStressResult(
        observed_sharpe=observed,
        periods_per_year=int(periods_per_year),
        n_paths=int(n_paths),
        levels=tuple(levels),
        errors=tuple(errors),
    )


def generate_datasets(
    returns: Sequence[float] | np.ndarray,
    *,
    block_sizes: Sequence[int] = (1, 5, 20, 60),
    n_paths: int = 100,
    seed: int = 0,
    scheme: str = "permutation",
) -> dict[int, np.ndarray]:
    """Build alternative versions of a return series and hand them back.

    Returns a mapping from block length to an array of shape
    ``(n_paths, len(returns))``. Nothing is scored here and no strategy is
    involved. Take the arrays, run whatever you like on them, and compare.

    Each block length answers a different question. At 1 no ordering survives.
    At 60 runs of sixty periods stay intact. Sweeping the range is more
    informative than picking one value.

    What survives besides the ordering depends on ``scheme`` and the difference
    matters. Under the default permutation every value in the original appears
    exactly once in every generated series, so these are arrangements of your
    history rather than new observations, and the mean, the variance, the skew
    and every extreme are exactly those of the source. A model that performs
    the same on them was not using time structure, because nothing else changed.

    Under ``"stationary"`` or ``"moving"`` the blocks are drawn with
    replacement. Values are duplicated and omitted, so those series are not
    arrangements of your history and their moments differ from the source. A
    model can then fail on them for reasons that have nothing to do with
    ordering, which is why they are not the default and why
    :func:`path_stress` does not offer them at all. They are here for callers
    who want bootstrap inference and know that is what they are asking for.

    In every case these are built from the data you supplied and can say
    nothing about behaviour it never contained.
    """

    data = _as_returns(returns, name="returns")
    blocks = sorted({int(size) for size in block_sizes})
    if not blocks or blocks[0] < 1:
        raise ValueError("block_sizes must all be at least 1")
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1")

    out: dict[int, np.ndarray] = {}
    for position, block in enumerate(blocks):
        level_seed = seed * 1_000_003 + position
        if scheme == "permutation":
            out[block] = block_permutation(
                data, n_paths, block_size=block, seed=level_seed
            )
        elif scheme == "stationary":
            out[block] = (
                iid_bootstrap(data, n_paths, seed=level_seed)
                if block == 1
                else stationary_bootstrap(
                    data, n_paths, expected_block=float(block), seed=level_seed
                )
            )
        elif scheme == "moving":
            out[block] = (
                iid_bootstrap(data, n_paths, seed=level_seed)
                if block == 1
                else moving_block_bootstrap(
                    data, n_paths, block_size=block, seed=level_seed
                )
            )
        else:
            raise ValueError(
                f"scheme must be permutation, stationary or moving; got {scheme!r}"
            )
    return out


def write_datasets(
    returns: Sequence[float] | np.ndarray,
    directory: str,
    *,
    block_sizes: Sequence[int] = (1, 5, 20, 60),
    n_paths: int = 100,
    seed: int = 0,
    scheme: str = "permutation",
) -> dict[str, object]:
    """Write the generated series to CSV files and return a manifest.

    One file per block length, each column a generated path. The manifest
    records the settings and a hash of the input so a result can be traced back
    to the series it came from.

    Values are written with ``repr``, which is the shortest string that reads
    back as the identical double. This matters more here than it looks. The
    permutation preserves the multiset exactly, and the whole point of these
    files is that a model runs on them, so rounding on the way out would break
    the claim at the only place a user actually sees it. An earlier version
    wrote ten significant figures and quietly lost the last few bits.
    """

    import csv
    import hashlib
    import json
    import os

    data = _as_returns(returns, name="returns")
    datasets = generate_datasets(
        data,
        block_sizes=block_sizes,
        n_paths=n_paths,
        seed=seed,
        scheme=scheme,
    )
    os.makedirs(directory, exist_ok=True)

    digest = hashlib.blake2b(data.tobytes(), digest_size=8).hexdigest()
    files = []
    for block, paths in datasets.items():
        name = f"block-{block:03d}.csv"
        with open(os.path.join(directory, name), "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([f"path_{index + 1}" for index in range(paths.shape[0])])
            for row in paths.T:
                writer.writerow([repr(float(value)) for value in row])
        files.append({"file": name, "block_size": block, "n_paths": int(paths.shape[0])})

    # The exactness claim belongs to the permutation and to nothing else, so
    # the manifest must not repeat it for a scheme that draws with replacement.
    if scheme == "permutation":
        note = (
            "Every value in the source appears exactly once in every generated "
            "series. These are reorderings, not new observations. The mean, the "
            "variance, the skew and the extremes are identical to the source "
            "and only the order changes, so they carry no information the "
            "source did not already contain."
        )
    else:
        note = (
            f"Generated with the {scheme} bootstrap, which draws with "
            "replacement. A generated series can therefore duplicate some "
            "observations and omit others, and its mean, variance, skew and "
            "extremes will differ from the source. Use scheme='permutation' if "
            "you need the distribution held fixed while only the order changes."
        )

    manifest = {
        "source_periods": int(data.size),
        "source_fingerprint": digest,
        "seed": int(seed),
        "scheme": scheme,
        "preserves_multiset": scheme == "permutation",
        "files": files,
        "note": note,
    }
    with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
