#!/usr/bin/env python3
"""Build the image for a LinkedIn post from real noise trials.

Square, because LinkedIn shows that largest on a phone. Everything on it is
computed here rather than drawn, so the numbers cannot drift from the library.

    python scripts/make_post_image.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from overfitlab import deflated_sharpe_ratio

INK = "#20272E"
MUTED = "#66717C"
BLUE = "#2F5D7E"
GREY = "#C9CFC9"
PAPER = "#F5F3ED"

N_TRIALS = 60
PERIODS = 756
PERIODS_PER_YEAR = 252


def main() -> None:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
    })

    rng = np.random.default_rng(20260801)
    trials = rng.normal(scale=0.01, size=(PERIODS, N_TRIALS))
    result = deflated_sharpe_ratio(trials, periods_per_year=PERIODS_PER_YEAR)

    annualiser = np.sqrt(PERIODS_PER_YEAR)
    observed = result.observed_sharpe * annualiser
    bar = result.benchmark_sharpe * annualiser
    equity = np.cumsum(trials, axis=0)

    fig = plt.figure(figsize=(10, 10), facecolor=PAPER)

    fig.text(0.07, 0.925, "60 strategies.", fontsize=40, fontweight="bold", color=INK)
    fig.text(0.07, 0.868, "None of them has any edge.", fontsize=40,
             fontweight="bold", color=INK)

    ax = fig.add_axes([0.07, 0.30, 0.86, 0.50])
    ax.set_facecolor(PAPER)
    for column in range(N_TRIALS):
        if column == result.selected_trial:
            continue
        ax.plot(equity[:, column], color=GREY, linewidth=0.9)
    ax.plot(equity[:, result.selected_trial], color=BLUE, linewidth=3.0)
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.text(0.99, 0.03, "three years of daily returns, drawn from noise",
            transform=ax.transAxes, ha="right", fontsize=13, color=MUTED)

    fig.text(0.07, 0.220,
             f"The best one looks like an annualised Sharpe of {observed:.2f}.",
             fontsize=22, color=INK)
    fig.text(0.07, 0.165,
             f"The best of 60 coin flips reaches {bar:.2f}.",
             fontsize=22, color=INK)
    fig.text(0.07, 0.110, "So it is below what luck alone produces.",
             fontsize=22, color=INK)

    fig.text(0.07, 0.040, "OverfitLab", fontsize=16, fontweight="bold", color=INK)
    fig.text(0.235, 0.040, "github.com/RomanMski/overfitlab", fontsize=16, color=MUTED)

    output = Path("docs/images")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "post-card.png"
    fig.savefig(destination, dpi=120, facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {destination}")
    print(f"  observed {observed:.4f}  bar {bar:.4f}  deflated {result.deflated_sharpe:.4f}")


if __name__ == "__main__":
    main()
