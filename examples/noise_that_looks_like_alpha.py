"""Search 200 configurations over pure noise and watch a strategy appear.

Nothing in this script contains any signal. Every one of the 200 return series
is drawn from the same zero-mean distribution, so no configuration is better
than any other and the best of them is the luckiest, not the smartest.

Run it with::

    python examples/noise_that_looks_like_alpha.py
"""

from __future__ import annotations

import numpy as np

from stressfold import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)

PERIODS = 1_000
TRIALS = 200
PERIODS_PER_YEAR = 252


def main() -> None:
    rng = np.random.default_rng(2)
    trials = rng.normal(0.0, 0.01, size=(PERIODS, TRIALS))

    sharpes = np.array([sharpe_ratio(trials[:, i]) for i in range(TRIALS)])
    winner = int(np.argmax(sharpes))
    annual = sharpes[winner] * np.sqrt(PERIODS_PER_YEAR)

    print(f"{TRIALS} configurations, {PERIODS} periods, no signal anywhere.")
    print()
    print("What the backtest reports for the best configuration")
    print(f"  config #{winner}")
    print(f"  Sharpe {sharpes[winner]:.4f} per period, {annual:.2f} annualised")

    naive = probabilistic_sharpe_ratio(
        float(sharpes[winner]), n_observations=PERIODS
    )
    print(f"  probability the Sharpe beats zero, ignoring the search: {naive:.3f}")
    print()

    deflated = deflated_sharpe_ratio(trials, periods_per_year=PERIODS_PER_YEAR)
    print("What it is worth once the search is accounted for")
    print(f"  {deflated}".replace("\n", "\n  "))
    print()

    audit = probability_of_backtest_overfitting(trials, n_splits=16)
    print("How well the selection procedure itself holds up")
    print(f"  {audit}".replace("\n", "\n  "))
    print()

    print("Read together: an annualised Sharpe above 1 that a naive statistic is")
    print(f"{naive:.1%} confident about, produced entirely by trying things.")


if __name__ == "__main__":
    main()
