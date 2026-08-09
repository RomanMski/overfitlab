#!/usr/bin/env python3
"""Run the structure sweep on real market data rather than a simulation.

Everything else in this repository demonstrates the method on series with
known properties, where the right answer is fixed in advance. That is the
right way to show a tool does what it claims and it is not evidence about any
real market. This script closes that gap.

It downloads daily adjusted closes for five liquid ETFs, runs four strategies
through the sweep on each, and prints the table. Nothing here is fitted and
nothing is selected after the fact. The four strategies were chosen before the
data was pulled, because two of them have a known sign in the literature and
disagreeing with that would be evidence the tool is broken.

    python -m pip install yfinance
    python scripts/real_data.py

The download is cached, so a second run is offline and reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from overfitlab import path_stress

TICKERS = ("SPY", "QQQ", "TLT", "GLD", "EEM")
START = "2000-01-01"
END = "2026-08-01"
BLOCKS = (1, 5, 20, 60)
CACHE = Path("data/real_prices.csv")


def load_prices(cache: Path = CACHE) -> pd.DataFrame:
    """Adjusted closes, from the cache if it is there and the network if not."""

    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    try:
        import yfinance
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            f"{cache} is missing and yfinance is not installed, so there is no "
            "way to get the prices. Run python -m pip install yfinance."
        ) from exc

    frame = yfinance.download(
        list(TICKERS), start=START, end=END, auto_adjust=True, progress=False
    )["Close"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache)
    return frame


def buy_and_hold(market: np.ndarray) -> np.ndarray:
    return market


def one_day_trend(market: np.ndarray) -> np.ndarray:
    """Long after an up day. Known to lose at daily frequency in equities."""

    return np.sign(market[:-1]) * market[1:]


def trend_follower(lookback: int = 60):
    """Long when the trailing sum is positive. Time series momentum."""

    def strategy(market: np.ndarray) -> np.ndarray:
        signal = pd.Series(market).rolling(lookback).sum().to_numpy()
        return np.sign(signal[lookback - 1 : -1]) * market[lookback:]

    return strategy


def volatility_targeted(lookback: int = 20, target: float = 0.01):
    """Always long, sized by recent realised volatility.

    This has no view on direction at all, so anything it earns above buy and
    hold comes from the clustering of volatility rather than from timing. It
    is here because it is the case the sweep is easiest to misread on.
    """

    def strategy(market: np.ndarray) -> np.ndarray:
        vol = pd.Series(market).rolling(lookback).std().to_numpy()
        size = np.clip(target / vol[lookback - 1 : -1], 0.0, 3.0)
        return size * market[lookback:]

    return strategy


STRATEGIES = {
    "buy and hold": buy_and_hold,
    "1 day trend": one_day_trend,
    "60 day trend": trend_follower(),
    "volatility targeted": volatility_targeted(),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=400)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    frame = load_prices()
    header = "  ".join(f"b{block:<5}" for block in BLOCKS)
    rows: list[str] = []

    for ticker in TICKERS:
        series = frame[ticker].dropna()
        returns = series.pct_change().dropna().to_numpy()
        rows.append(
            f"\n{ticker}  {series.index[0].date()} to {series.index[-1].date()}"
            f"  {returns.size} days"
        )
        rows.append(f"  {'strategy':<21}{'real':>6}   {header}    D      p")
        for name, strategy in STRATEGIES.items():
            result = path_stress(
                strategy, returns, block_sizes=BLOCKS,
                n_paths=args.paths, seed=args.seed,
            )
            medians = "  ".join(
                f"{level['median_annualised']:6.2f}" for level in result.levels
            )
            rows.append(
                f"  {name:<21}{result.observed_annualised:6.2f}   {medians}"
                f"  {result.structure_dependence():6.2f}"
                f"  {result.shuffled_p_value():5.3f}"
            )

    report = "\n".join(rows)
    print(report)
    print(
        "\nreal is the Sharpe on the history that happened. The b columns are "
        "median Sharpe\nacross generated markets at that block length. D is "
        "structure dependence and p is\nhow often the fully shuffled markets "
        "matched the real result."
    )


if __name__ == "__main__":
    main()
