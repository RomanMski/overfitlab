"""Probability of backtest overfitting, by combinatorially symmetric cross-validation.

The method is due to Bailey, Borwein, Lopez de Prado and Zhu, "The Probability
of Backtest Overfitting" (Journal of Computational Finance, 2016).

The question it answers is narrow and worth stating precisely. Given a set of
strategy configurations that you already backtested, and their period-by-period
returns, how often does the configuration that looked best on one half of the
history fall below the median of its peers on the other half?

If that happens about half the time, your selection procedure carries no
information: you would have done as well choosing at random. That is the null
this statistic is built around, and PBO near 0.5 is the signature of a search
that found nothing but noise.

What it is not
--------------
PBO is a property of a *selection procedure over a set of trials*, not of a
strategy. It cannot tell you a strategy is profitable, it says nothing about
transaction costs or capacity, and it assumes the trials you supply are the
trials you actually ran. Feeding it a curated subset of your search is the
statistical equivalent of reporting the best fold.

It is also blind to chronology. CSCV enumerates every way of choosing half the
blocks, and that set of subsets is unchanged when the blocks are relabelled, so
the statistic is exactly invariant to the order of the history. A strategy that
worked until some structural break and stopped afterwards will not be flagged
by PBO, because a symmetric split averages the good and bad regimes on both
sides. Detecting that needs a chronological protocol such as walk-forward, and
a mid-range PBO must not be read as evidence against regime change.

Finally, one PBO number is noisier than it looks. Across independent datasets
drawn from the same null, the estimate has a standard deviation of roughly 0.17
for a few dozen trials over several hundred periods. Treat 0.45 and 0.55 as the
same answer, and be wary of any write-up that reports PBO to three decimals as
though it were precise.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

__all__ = ["PBOResult", "probability_of_backtest_overfitting"]

_MAX_COMBINATIONS = 20_000


@dataclass(frozen=True)
class PBOResult:
    """The outcome of a CSCV run."""

    pbo: float
    n_trials: int
    n_periods: int
    n_splits: int
    n_combinations: int
    combinations_were_sampled: bool
    logits: np.ndarray
    relative_ranks: np.ndarray
    selected_trial: np.ndarray
    is_performance: np.ndarray
    oos_performance: np.ndarray
    degradation_slope: float
    degradation_intercept: float
    probability_of_loss: float

    def summary(self) -> dict[str, Any]:
        return {
            "pbo": self.pbo,
            "n_trials": self.n_trials,
            "n_periods": self.n_periods,
            "n_splits": self.n_splits,
            "n_combinations": self.n_combinations,
            "combinations_were_sampled": self.combinations_were_sampled,
            "median_logit": float(np.median(self.logits)),
            "mean_is_performance": float(np.mean(self.is_performance)),
            "mean_oos_performance": float(np.mean(self.oos_performance)),
            "degradation_slope": self.degradation_slope,
            "probability_of_loss": self.probability_of_loss,
        }

    def __str__(self) -> str:
        sampled = (
            f" (sampled from {comb(self.n_splits, self.n_splits // 2):,})"
            if self.combinations_were_sampled
            else ""
        )
        lines = [
            f"Probability of backtest overfitting: {self.pbo:.3f}",
            f"  {self.n_trials} trials over {self.n_periods} periods,"
            f" {self.n_splits} blocks, {self.n_combinations:,} splits{sampled}",
            f"  selected config averaged {self.mean_is():.3f} in sample"
            f" and {self.mean_oos():.3f} out of sample",
            f"  out-of-sample performance degrades with a slope of"
            f" {self.degradation_slope:.3f}",
            f"  the selected config lost money out of sample in"
            f" {self.probability_of_loss:.1%} of splits",
        ]
        return "\n".join(lines)

    def mean_is(self) -> float:
        return float(np.mean(self.is_performance))

    def mean_oos(self) -> float:
        return float(np.mean(self.oos_performance))


def _as_matrix(returns: Any) -> tuple[np.ndarray, list[str]]:
    if isinstance(returns, pd.DataFrame):
        names = [str(column) for column in returns.columns]
        matrix = returns.to_numpy(dtype=float)
    else:
        matrix = np.asarray(returns, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(
                "returns must be two-dimensional, with periods on the rows and "
                "one column per trial"
            )
        names = [f"trial_{index}" for index in range(matrix.shape[1])]
    if not np.isfinite(matrix).all():
        raise ValueError(
            "returns contains NaN or infinite values. CSCV compares every trial "
            "on the same periods, so gaps have to be resolved before the audit "
            "rather than dropped inside it."
        )
    return matrix, names


def _block_moments(
    matrix: np.ndarray, n_splits: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-block sums, sums of squares and lengths.

    Every CSCV combination is a union of whole blocks, so a Sharpe ratio over
    any combination can be rebuilt from these three arrays. That turns the
    inner loop into one matrix product instead of tens of thousands of passes
    over the returns.
    """

    n_periods = matrix.shape[0]
    # Contiguous blocks preserve the order of the series inside each block.
    edges = np.linspace(0, n_periods, n_splits + 1).astype(int)
    sums = np.empty((n_splits, matrix.shape[1]), dtype=float)
    sums_of_squares = np.empty_like(sums)
    lengths = np.empty(n_splits, dtype=float)
    for index in range(n_splits):
        block = matrix[edges[index] : edges[index + 1]]
        sums[index] = block.sum(axis=0)
        sums_of_squares[index] = np.square(block).sum(axis=0)
        lengths[index] = block.shape[0]
    return sums, sums_of_squares, lengths


def _sharpe_from_moments(
    sums: np.ndarray, sums_of_squares: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """Sharpe ratio per trial, using the sample standard deviation."""

    counts = counts[:, None]
    means = sums / counts
    # Sample variance, guarding the degenerate single-observation case.
    variances = (sums_of_squares - counts * np.square(means)) / np.maximum(
        counts - 1.0, 1.0
    )
    deviations = np.sqrt(np.maximum(variances, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(deviations > 0.0, means / deviations, 0.0)
    return sharpe


def probability_of_backtest_overfitting(
    returns: Any,
    *,
    n_splits: int = 16,
    max_combinations: int = _MAX_COMBINATIONS,
    random_state: int = 0,
) -> PBOResult:
    """Estimate the probability of backtest overfitting for a set of trials.

    Parameters
    ----------
    returns:
        A ``(n_periods, n_trials)`` table of period-by-period returns, one
        column per configuration you tried. Pass **every** configuration you
        evaluated. Dropping the ones that did badly is exactly the bias this
        function exists to measure.
    n_splits:
        Number of contiguous blocks the history is cut into. Must be even.
        The paper uses 16, which gives 12,870 symmetric splits.
    max_combinations:
        If the full set of splits exceeds this, a deterministic random sample
        of that many is used instead, and the result records that it happened.
    random_state:
        Seed for the sampling described above. Unused when every split is
        enumerated.

    Returns
    -------
    PBOResult
        ``pbo`` is the fraction of splits where the in-sample winner ranked at
        or below the median of the other trials out of sample.
    """

    matrix, _ = _as_matrix(returns)
    n_periods, n_trials = matrix.shape

    if n_trials < 2:
        raise ValueError(
            "CSCV compares a selected trial against its peers, so at least two "
            f"trials are required; got {n_trials}"
        )
    if n_splits % 2 != 0:
        raise ValueError(f"n_splits must be even; got {n_splits}")
    if n_splits < 4:
        raise ValueError(f"n_splits must be at least 4; got {n_splits}")
    if n_periods < n_splits * 2:
        raise ValueError(
            f"{n_periods} periods cannot be cut into {n_splits} usable blocks. "
            "Provide a longer history or reduce n_splits."
        )

    sums, sums_of_squares, lengths = _block_moments(matrix, n_splits)

    total = comb(n_splits, n_splits // 2)
    sampled = total > max_combinations
    if sampled:
        rng = np.random.default_rng(random_state)
        # Draw distinct partitions. Sampling each row independently can repeat a
        # partition, which silently gives it extra weight in the average.
        seen: set[tuple[int, ...]] = set()
        attempts = 0
        limit = max_combinations * 50
        while len(seen) < max_combinations and attempts < limit:
            chosen = rng.choice(n_splits, size=n_splits // 2, replace=False)
            seen.add(tuple(sorted(int(value) for value in chosen)))
            attempts += 1
        masks = np.zeros((len(seen), n_splits), dtype=bool)
        for row, chosen in enumerate(sorted(seen)):
            masks[row, list(chosen)] = True
    else:
        masks = np.zeros((total, n_splits), dtype=bool)
        for row, chosen in enumerate(combinations(range(n_splits), n_splits // 2)):
            masks[row, list(chosen)] = True

    weights = masks.astype(float)
    complement = (~masks).astype(float)

    is_sharpe = _sharpe_from_moments(
        weights @ sums, weights @ sums_of_squares, weights @ lengths
    )
    oos_sharpe = _sharpe_from_moments(
        complement @ sums, complement @ sums_of_squares, complement @ lengths
    )

    selected = np.argmax(is_sharpe, axis=1)
    rows = np.arange(masks.shape[0])

    # Rank the selected trial among all trials on the complementary half.
    # Ranks run 1..n_trials with the best performer last, so a relative rank
    # above one half means the winner stayed in the upper half out of sample.
    ranks = rankdata(oos_sharpe, axis=1)
    selected_rank = ranks[rows, selected]
    relative_rank = selected_rank / (n_trials + 1.0)
    logits = np.log(relative_rank / (1.0 - relative_rank))

    pbo = float(np.mean(logits <= 0.0))

    is_selected = is_sharpe[rows, selected]
    oos_selected = oos_sharpe[rows, selected]
    if np.ptp(is_selected) > 0:
        slope, intercept = np.polyfit(is_selected, oos_selected, 1)
    else:
        slope, intercept = 0.0, float(np.mean(oos_selected))

    return PBOResult(
        pbo=pbo,
        n_trials=n_trials,
        n_periods=n_periods,
        n_splits=n_splits,
        n_combinations=int(masks.shape[0]),
        combinations_were_sampled=bool(sampled),
        logits=logits,
        relative_ranks=relative_rank,
        selected_trial=selected,
        is_performance=is_selected,
        oos_performance=oos_selected,
        degradation_slope=float(slope),
        degradation_intercept=float(intercept),
        probability_of_loss=float(np.mean(oos_selected < 0.0)),
    )
