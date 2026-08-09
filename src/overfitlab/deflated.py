"""Probabilistic and deflated Sharpe ratios.

Both are due to Bailey and Lopez de Prado: "The Sharpe Ratio Efficient
Frontier" (Journal of Risk, 2012) for the probabilistic form, and "The Deflated
Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and
Non-Normality" (Journal of Portfolio Management, 2014) for the deflated form.

The problem being solved
------------------------
A Sharpe ratio computed from a finite sample is an estimate, and it is a biased
one once you have picked the best of several trials. Two corrections are
needed, and they are separate.

The first is sampling error, made worse by non-normal returns. A Sharpe of 1.5
over 40 observations of a skewed, fat-tailed series is far weaker evidence than
the same number over 4,000. The probabilistic Sharpe ratio gives the
probability that the true Sharpe exceeds a chosen benchmark, given the observed
higher moments and the sample length.

The second is selection. If you tried 200 configurations, the best of them has
an inflated Sharpe even when none has any edge, because you took a maximum over
200 noisy draws. The deflated Sharpe ratio sets the benchmark to the Sharpe you
would *expect* the winner to show under that null, then asks whether the
observed one clears it.

Scale
-----
Every Sharpe here is per period, matching the frequency of the returns you
supply. Annualising before deflating would corrupt the result, because the
sample-length and higher-moment terms are defined on the observation scale.
``periods_per_year`` is therefore only ever used for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import e, sqrt
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

__all__ = [
    "DeflatedSharpeResult",
    "deflated_sharpe_ratio",
    "expected_maximum_sharpe",
    "probabilistic_sharpe_ratio",
    "sharpe_ratio",
]

# Euler-Mascheroni constant, as used in the expected-maximum expression.
_EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: Any, *, ddof: int = 1) -> float:
    """Per-period Sharpe ratio of a single return series."""

    values = np.asarray(returns, dtype=float).reshape(-1)
    if values.size < 2:
        raise ValueError("a Sharpe ratio needs at least two observations")
    deviation = values.std(ddof=ddof)
    # A constant series does not always produce a standard deviation of exactly
    # zero, because the mean it is measured against carries rounding error. An
    # exact comparison would then divide by something near 1e-18 and report an
    # enormous Sharpe for a series that never moved.
    scale = float(np.max(np.abs(values)))
    if deviation <= 1e-12 * max(scale, 1.0):
        return 0.0
    return float(values.mean() / deviation)


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_observations: int,
    benchmark_sharpe: float = 0.0,
    skewness: float = 0.0,
    kurtosis_: float = 3.0,
) -> float:
    """Probability that the true Sharpe ratio exceeds ``benchmark_sharpe``.

    ``kurtosis_`` is the raw fourth standardised moment, so a normal
    distribution takes the value 3 rather than 0.
    """

    if n_observations < 2:
        raise ValueError("the probabilistic Sharpe ratio needs at least two observations")
    variance_term = (
        1.0
        - skewness * observed_sharpe
        + ((kurtosis_ - 1.0) / 4.0) * observed_sharpe**2
    )
    if variance_term <= 0.0:
        # The estimator's variance expression has gone non-positive, which
        # happens for extreme skew and kurtosis combinations. Refusing is
        # honest; returning a probability here would be invented.
        return float("nan")
    numerator = (observed_sharpe - benchmark_sharpe) * sqrt(n_observations - 1.0)
    return float(norm.cdf(numerator / sqrt(variance_term)))


def expected_maximum_sharpe(
    *, n_trials: int, sharpe_variance: float
) -> float:
    """The Sharpe the best of ``n_trials`` reaches when none has any edge.

    This is the benchmark the deflated Sharpe ratio deflates against. It grows
    with the number of trials, which is the entire point: trying more things
    raises the bar the winner has to clear.
    """

    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if sharpe_variance < 0.0:
        raise ValueError("sharpe_variance cannot be negative")
    if n_trials == 1 or sharpe_variance == 0.0:
        # With a single trial nothing was selected, so there is no selection
        # bias to remove and the benchmark is zero.
        return 0.0
    deviation = sqrt(sharpe_variance)
    first = norm.ppf(1.0 - 1.0 / n_trials)
    second = norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float(
        deviation
        * ((1.0 - _EULER_MASCHERONI) * first + _EULER_MASCHERONI * second)
    )


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """The outcome of deflating a selected trial's Sharpe ratio."""

    deflated_sharpe: float
    probabilistic_sharpe: float
    observed_sharpe: float
    benchmark_sharpe: float
    n_trials: int
    n_observations: int
    skewness: float
    kurtosis: float
    sharpe_variance: float
    selected_trial: int
    periods_per_year: float | None

    @property
    def annualised_sharpe(self) -> float | None:
        if self.periods_per_year is None:
            return None
        return self.observed_sharpe * sqrt(self.periods_per_year)

    def summary(self) -> dict[str, Any]:
        payload = {
            "deflated_sharpe": self.deflated_sharpe,
            "probabilistic_sharpe": self.probabilistic_sharpe,
            "observed_sharpe": self.observed_sharpe,
            "benchmark_sharpe": self.benchmark_sharpe,
            "n_trials": self.n_trials,
            "n_observations": self.n_observations,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "selected_trial": self.selected_trial,
        }
        if self.periods_per_year is not None:
            payload["annualised_sharpe"] = self.annualised_sharpe
        return payload

    def __str__(self) -> str:
        annual = (
            f" ({self.annualised_sharpe:.2f} annualised)"
            if self.periods_per_year is not None
            else ""
        )
        return "\n".join(
            [
                f"Deflated Sharpe ratio: {self.deflated_sharpe:.3f}",
                f"  observed Sharpe {self.observed_sharpe:.4f} per period{annual}"
                f" over {self.n_observations} observations",
                f"  {self.n_trials} trials raise the bar to"
                f" {self.benchmark_sharpe:.4f} before any edge is credited",
                f"  ignoring selection entirely would have said"
                f" {self.probabilistic_sharpe:.3f}",
                f"  skew {self.skewness:+.3f}, kurtosis {self.kurtosis:.3f}",
            ]
        )


def deflated_sharpe_ratio(
    returns: Any,
    *,
    selected: int | None = None,
    periods_per_year: float | None = None,
) -> DeflatedSharpeResult:
    """Deflate the best trial's Sharpe ratio for the fact that you chose it.

    Parameters
    ----------
    returns:
        A ``(n_observations, n_trials)`` table of period-by-period returns, one
        column per configuration you tried, or a single series. As with
        :func:`~overfitlab.pbo.probability_of_backtest_overfitting`, pass
        every trial you ran. The count of trials is what sets the benchmark, so
        under-reporting it inflates the result.
    selected:
        Index of the trial to deflate. Defaults to the highest Sharpe, which is
        the one selection bias applies to.
    periods_per_year:
        Only used to report an annualised figure alongside the per-period one.

    Returns
    -------
    DeflatedSharpeResult
        ``deflated_sharpe`` is a probability, not a ratio: the probability that
        the selected trial's true Sharpe exceeds what the best of this many
        trials would reach with no edge at all.
    """

    if isinstance(returns, pd.DataFrame):
        matrix = returns.to_numpy(dtype=float)
    elif isinstance(returns, pd.Series):
        matrix = returns.to_numpy(dtype=float).reshape(-1, 1)
    else:
        matrix = np.asarray(returns, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError("returns must be one- or two-dimensional")
    if not np.isfinite(matrix).all():
        raise ValueError("returns contains NaN or infinite values")

    n_observations, n_trials = matrix.shape
    if n_observations < 2:
        raise ValueError("at least two observations are required")

    sharpes = np.array(
        [sharpe_ratio(matrix[:, index]) for index in range(n_trials)], dtype=float
    )
    index = int(np.argmax(sharpes)) if selected is None else int(selected)
    if not 0 <= index < n_trials:
        raise ValueError(f"selected must index one of {n_trials} trials")

    series = matrix[:, index]
    observed = float(sharpes[index])
    # Variance of the Sharpe ratios across trials, which is what the expected
    # maximum scales with. A single trial has no spread and no selection.
    variance = float(np.var(sharpes, ddof=1)) if n_trials > 1 else 0.0
    benchmark = expected_maximum_sharpe(
        n_trials=n_trials, sharpe_variance=variance
    )
    series_skew = float(skew(series, bias=False))
    series_kurtosis = float(kurtosis(series, fisher=False, bias=False))

    deflated = probabilistic_sharpe_ratio(
        observed,
        n_observations=n_observations,
        benchmark_sharpe=benchmark,
        skewness=series_skew,
        kurtosis_=series_kurtosis,
    )
    plain = probabilistic_sharpe_ratio(
        observed,
        n_observations=n_observations,
        benchmark_sharpe=0.0,
        skewness=series_skew,
        kurtosis_=series_kurtosis,
    )

    return DeflatedSharpeResult(
        deflated_sharpe=deflated,
        probabilistic_sharpe=plain,
        observed_sharpe=observed,
        benchmark_sharpe=benchmark,
        n_trials=n_trials,
        n_observations=n_observations,
        skewness=series_skew,
        kurtosis=series_kurtosis,
        sharpe_variance=variance,
        selected_trial=index,
        periods_per_year=periods_per_year,
    )
