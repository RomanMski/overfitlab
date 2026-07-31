#!/usr/bin/env python3
"""Build the README figure and the social preview from a real path stress run.

Both images are produced by the package they advertise, so neither can drift
away from what the tool actually does.

    python scripts/make_figures.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from overfitlab import path_stress

SEED = 20260731
INK = "#20272E"
MUTED = "#66717C"
GRID = "#D8DEE4"
BLUE = "#2F5D7E"
GOLD = "#C18F36"
PAPER = "#F5F3ED"
BLOCKS = (1, 5, 20, 60)


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


def market(n: int = 2500) -> np.ndarray:
    rng = np.random.default_rng(1)
    shocks = rng.normal(scale=0.011, size=n)
    series = np.empty(n)
    series[0] = shocks[0]
    for i in range(1, n):
        series[i] = 0.06 * series[i - 1] + shocks[i] + 0.0004
    return series


def momentum(values: np.ndarray) -> np.ndarray:
    return np.sign(values[:-1]) * values[1:]


def buy_and_hold(values: np.ndarray) -> np.ndarray:
    return values


def panel(ax, result, title, colour) -> None:
    blocks = [level["block_size"] for level in result.levels]
    medians = [level["median_annualised"] for level in result.levels]
    p95 = [level["p95_annualised"] for level in result.levels]
    positions = np.arange(len(blocks))

    ax.fill_between(positions, medians, p95, color=colour, alpha=0.16,
                    label="synthetic markets, median to 95th")
    ax.plot(positions, medians, "o-", color=colour, label="median on synthetic")
    ax.axhline(result.observed_annualised, color=INK, linestyle="--", linewidth=1.6,
               label="the market that happened")

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{int(b)}" for b in blocks])
    ax.set_xlabel("periods of market structure kept intact")
    ax.set_title(title, loc="left", fontweight="bold", pad=10)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.02, 0.06, f"structure dependence {result.structure_dependence():.2f}",
            transform=ax.transAxes, fontsize=9.2, color=MUTED)


def build(output: Path) -> tuple:
    data = market()
    trend = path_stress(momentum, data, block_sizes=BLOCKS, n_paths=300, seed=4)
    hold = path_stress(buy_and_hold, data, block_sizes=BLOCKS, n_paths=300, seed=4)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), constrained_layout=True,
                             sharey=True)
    panel(axes[0], trend, "A trend follower", BLUE)
    panel(axes[1], hold, "Buy and hold", GOLD)
    axes[0].set_ylabel("annualised Sharpe")
    axes[0].legend(loc="upper left")
    fig.suptitle(
        "Destroy the market's ordering and see what survives",
        fontsize=11.6, fontweight="bold", x=0.005, ha="left",
    )

    output.mkdir(parents=True, exist_ok=True)
    destination = output / "structure-sweep.png"
    fig.savefig(destination, dpi=200, bbox_inches="tight", pad_inches=0.14,
                metadata={"Software": "scripts/make_figures.py"})
    plt.close(fig)
    return trend, hold, destination


def social(output: Path, trend, hold) -> Path:
    """A 1200x630 card for link previews."""

    fig = plt.figure(figsize=(12, 6.3), facecolor=PAPER)
    fig.text(0.055, 0.82, "OverfitLab", fontsize=46, fontweight="bold", color=INK)
    fig.text(0.055, 0.735, "How much of your backtest is the search?",
             fontsize=19, color=MUTED)

    ax = fig.add_axes([0.055, 0.13, 0.89, 0.5])
    ax.set_facecolor(PAPER)
    positions = np.arange(len(BLOCKS))
    for result, colour, label in (
        (trend, BLUE, "a trend follower"),
        (hold, GOLD, "buy and hold"),
    ):
        medians = [level["median_annualised"] for level in result.levels]
        ax.plot(positions, medians, "o-", color=colour, linewidth=2.6,
                markersize=7, label=label)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"block {b}" for b in BLOCKS])
    ax.set_ylabel("annualised Sharpe on markets that never happened", fontsize=11)
    ax.grid(True, axis="y", color="#DCD8CE")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=12)
    ax.tick_params(labelsize=11)

    destination = output / "og.png"
    fig.savefig(destination, dpi=100, facecolor=PAPER,
                metadata={"Software": "scripts/make_figures.py"})
    plt.close(fig)
    return destination


def main() -> None:
    configure()
    trend, hold, figure = build(Path("docs/images"))
    card = social(Path("public"), trend, hold)
    print(f"wrote {figure}")
    print(f"wrote {card}")
    print(f"  trend follower  observed {trend.observed_annualised:.2f}"
          f"  dependence {trend.structure_dependence():.2f}")
    print(f"  buy and hold    observed {hold.observed_annualised:.2f}"
          f"  dependence {hold.structure_dependence():.2f}")


if __name__ == "__main__":
    main()
