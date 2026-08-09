#!/usr/bin/env python3
"""Draw the two real cases where structure dependence and the p-value disagree.

Both panels come from scripts/real_data.py, so the figure and the table cannot
say different things.

    python scripts/make_real_figure.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from overfitlab import block_permutation
from overfitlab.synthetic import _rank_statistics, _sharpe

from real_data import load_prices, trend_follower, volatility_targeted

INK = "#20272E"
MUTED = "#66717C"
GRID = "#D8DEE4"
BLUE = "#2F5D7E"
GOLD = "#C18F36"
N_PATHS = 2000
SEED = 11


def configure() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.family": "serif",
        "font.serif": ["DejaVu Serif"], "font.size": 9.5,
        "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "xtick.labelsize": 9.0, "ytick.labelsize": 9.0, "legend.fontsize": 9.0,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.8,
        "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    })


def null_distribution(strategy, returns: np.ndarray) -> tuple[float, np.ndarray]:
    """Sharpe on the real series, and on N_PATHS fully shuffled versions."""

    annualiser = float(np.sqrt(252))
    observed = _sharpe(strategy(returns)) * annualiser
    paths = block_permutation(returns, N_PATHS, block_size=1, seed=SEED)
    scores = np.array([_sharpe(strategy(paths[i])) for i in range(N_PATHS)])
    return observed, scores * annualiser


def panel(ax, observed, scores, title, colour, note) -> None:
    ax.hist(scores, bins=44, color=colour, alpha=0.28, edgecolor=colour, linewidth=0.5)
    ax.axvline(observed, color=INK, linestyle="--", linewidth=1.8,
               label=f"the market that happened, {observed:.2f}")
    ax.axvline(float(np.median(scores)), color=MUTED, linewidth=1.2,
               label=f"median of the shuffles, {np.median(scores):.2f}")
    ax.set_title(title, loc="left", fontweight="bold", pad=10)
    ax.set_xlabel("annualised Sharpe")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_yticks([])
    ax.legend(loc="best")
    ax.text(0.02, 0.42, note, transform=ax.transAxes, fontsize=9.2, color=MUTED)


def main() -> None:
    configure()
    frame = load_prices()

    cases = []
    for ticker, strategy, label in (
        ("SPY", trend_follower(), "A 60 day trend follower on SPY"),
        ("QQQ", volatility_targeted(), "Volatility targeting on QQQ"),
    ):
        returns = frame[ticker].dropna().pct_change().dropna().to_numpy()
        observed, scores = null_distribution(strategy, returns)
        percentile, p_value = _rank_statistics(scores, observed)
        dependence = 1.0 - float(np.median(scores)) / observed
        cases.append((label, observed, scores, dependence, p_value))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), constrained_layout=True)
    for ax, colour, (label, observed, scores, dependence, p_value) in zip(
        axes, (BLUE, GOLD), cases
    ):
        # 2000 paths cannot resolve a p-value below 1/2001, so do not print one.
        shown = "below 0.001" if p_value < 0.001 else f"{p_value:.3f}"
        note = f"structure dependence {dependence:.2f}\np-value {shown}"
        panel(ax, observed, scores, label, colour, note)

    fig.suptitle(
        "Structure dependence says one thing and the shuffles say another",
        fontsize=11.6, fontweight="bold", x=0.005, ha="left",
    )

    output = Path("docs/images")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "real-data.png"
    fig.savefig(destination, dpi=200, bbox_inches="tight", pad_inches=0.14,
                metadata={"Software": "scripts/make_real_figure.py"})
    plt.close(fig)

    print(f"wrote {destination}")
    for label, observed, scores, dependence, p_value in cases:
        print(f"  {label:34s} observed {observed:5.2f}"
              f"  median {np.median(scores):5.2f}"
              f"  D {dependence:5.2f}  p {p_value:.4f}")


if __name__ == "__main__":
    main()
