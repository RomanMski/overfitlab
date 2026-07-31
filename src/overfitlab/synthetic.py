"""Alternative histories, and what a strategy does on markets that never happened.

A backtest is one draw. The strategy was tuned on the one price path that
history handed you, and the usual defence, a train and test split, still cuts
up that same single path.

This module builds new paths from the one you have and reruns the strategy on
each. The interesting control is *how much structure the resampling keeps*:

``block_size = 1``
    Every ordering is destroyed. Returns are drawn independently, so no
    autocorrelation, no momentum, no mean reversion, nothing a timing rule can
    exploit. Volatility clustering is gone too. A strategy that still scores
    well here is not trading time structure at all, and is usually just long.

``block_size = 20``
    Runs of twenty periods are kept intact and only their order is shuffled.
    Anything the strategy exploits over a horizon shorter than a block mostly
    survives.

Sweeping the block size therefore tells you *what* the strategy depends on,
rather than only whether it works. That gradient is the point of this module.

The generators are the standard resampling schemes for dependent data, with
the stationary bootstrap of Politis and Romano (1994) as the default because
its geometric block lengths leave the resampled series stationary, which fixed
blocks do not.
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

        One means the strategy collapses entirely once ordering is destroyed,
        which is what a genuine timing edge looks like. Zero means it does just
        as well on independently shuffled returns, so whatever it earns does
        not come from timing.
        """

        if not self.levels:
            return float("nan")
        shuffled = min(self.levels, key=lambda level: level["block_size"])
        if abs(self.observed_sharpe) < 1e-12:
            return float("nan")
        return float(
            1.0 - shuffled["median_sharpe"] / self.observed_sharpe
        )

    def summary_text(self) -> str:
        lines = [
            f"Observed Sharpe {self.observed_annualised:.3f} annualised,"
            f" from {self.n_paths} synthetic paths per level",
            "",
            "  block  keeps                       median  p95     the real result beats",
        ]
        for level in self.levels:
            block = int(level["block_size"])
            keeps = (
                "nothing, pure noise"
                if block == 1
                else f"runs of {block} periods"
            )
            lines.append(
                f"  {block:>5}  {keeps:<26}"
                f"  {level['median_annualised']:>6.2f}"
                f"  {level['p95_annualised']:>6.2f}"
                f"  {level['percentile']:>5.1f}% of them"
            )
        dependence = self.structure_dependence()
        lines += ["", f"Structure dependence {dependence:.2f}"]
        if dependence < 0.5:
            lines.append(
                "  The strategy does nearly as well on shuffled returns, so it is"
                " not earning this from timing."
            )
        else:
            lines.append(
                "  Most of the result disappears once ordering is destroyed, which"
                " is what a timing edge should do."
            )
        return "\n".join(lines)


def path_stress(
    strategy: Strategy,
    market_returns: Sequence[float] | np.ndarray,
    *,
    block_sizes: Sequence[int] = (1, 5, 20, 60),
    n_paths: int = 200,
    periods_per_year: int = 252,
    seed: int = 0,
    stationary: bool = True,
) -> PathStressResult:
    """Rerun ``strategy`` on synthetic markets built from ``market_returns``.

    ``strategy`` maps a market return series to the returns it would have
    earned on that series. It is called once on the real history and then once
    per synthetic path, so it must be deterministic and reasonably fast.

    The block sizes are swept from most destructive to least. Reading the
    resulting gradient is the point: a timing edge should die at
    ``block_size = 1`` and recover as blocks lengthen. A result that survives
    the shuffle was never about timing.
    """

    data = _as_returns(market_returns, name="market_returns")
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be at least 1")
    blocks = sorted({int(size) for size in block_sizes})
    if not blocks or blocks[0] < 1:
        raise ValueError("block_sizes must all be at least 1")

    observed = _sharpe(np.asarray(strategy(data), dtype=float).reshape(-1))
    annualiser = float(np.sqrt(periods_per_year))
    errors: list[str] = []
    levels: list[dict[str, float]] = []

    for position, block in enumerate(blocks):
        level_seed = seed * 1_000_003 + position
        if block == 1:
            paths = iid_bootstrap(data, n_paths, seed=level_seed)
        elif stationary:
            paths = stationary_bootstrap(
                data, n_paths, expected_block=float(block), seed=level_seed
            )
        else:
            paths = moving_block_bootstrap(
                data, n_paths, block_size=block, seed=level_seed
            )

        scores: list[float] = []
        for index in range(paths.shape[0]):
            try:
                result = np.asarray(strategy(paths[index]), dtype=float).reshape(-1)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                errors.append(f"block {block} path {index}: {type(exc).__name__}: {exc}")
                continue
            scores.append(_sharpe(result))
        if not scores:
            errors.append(f"block {block}: every path failed")
            continue

        values = np.asarray(scores, dtype=float)
        levels.append(
            {
                "block_size": float(block),
                "n_paths": float(values.size),
                "median_sharpe": float(np.median(values)),
                "median_annualised": float(np.median(values) * annualiser),
                "p95_annualised": float(np.quantile(values, 0.95) * annualiser),
                "mean_annualised": float(np.mean(values) * annualiser),
                # How much of the synthetic distribution the real result beats.
                "percentile": float(100.0 * np.mean(values < observed)),
            }
        )

    return PathStressResult(
        observed_sharpe=observed,
        periods_per_year=int(periods_per_year),
        n_paths=int(n_paths),
        levels=tuple(levels),
        errors=tuple(errors),
    )
