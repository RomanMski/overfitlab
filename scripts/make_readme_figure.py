#!/usr/bin/env python3
"""Build the README header figure from a real StressFold search audit.

The figure is produced by the package it documents rather than drawn by hand.
It runs one 24-candidate grid search twice, against a target that carries real
signal and against a target that is pure noise, and plots the score each search
reported against the distribution of scores the same search reaches on shuffled
targets.

    python scripts/make_readme_figure.py --output docs/images
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from stressfold import SearchAuditConfig, audit_search

SEED = 20260730
INK = "#20272E"
MUTED = "#66717C"
GRID = "#D8DEE4"
BLUE = "#2F5D7E"
GOLD = "#C18F36"
GOLD_LIGHT = "#E6D1A4"


def configure() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.4,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.8,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.8,
            "legend.frameon": False,
        }
    )


def build_search() -> GridSearchCV:
    pipeline = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=0))
    grid = {
        "decisiontreeclassifier__max_depth": [1, 2, 3, 5, 8, None],
        "decisiontreeclassifier__min_samples_leaf": [1, 2, 5, 10],
    }
    return GridSearchCV(pipeline, grid, cv=4, scoring="roc_auc")


def signal_table(n: int = 300):
    values, target = make_classification(
        n_samples=n,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        class_sep=1.5,
        random_state=3,
    )
    columns = [f"feature_{index}" for index in range(values.shape[1])]
    return pd.DataFrame(values, columns=columns), pd.Series(target, name="outcome")


def noise_table(n: int = 300):
    rng = np.random.default_rng(0)
    columns = [f"feature_{index}" for index in range(8)]
    return (
        pd.DataFrame(rng.normal(size=(n, 8)), columns=columns),
        pd.Series(rng.integers(0, 2, size=n), name="outcome"),
    )


def run(permutations: int):
    config = SearchAuditConfig(
        task="binary",
        random_state=7,
        outer_repeats=2,
        permutation_repeats=permutations,
        noise_levels=(0.0,),
        verbose=True,
    )
    noise = audit_search(build_search(), *noise_table(), config=config)
    signal = audit_search(build_search(), *signal_table(), config=config)
    return noise, signal


def panel(ax, result, title, subtitle) -> None:
    nulls = result.permutation_frame()["null_best_score"].to_numpy(float)
    nulls = nulls[np.isfinite(nulls)]
    reported = result.reported_score
    summary = result.permutation_summary()

    lo = min(nulls.min(), reported) - 0.025
    hi = max(nulls.max(), reported) + 0.045
    ax.hist(
        nulls,
        bins=np.linspace(lo, hi, 26),
        color=GOLD_LIGHT,
        edgecolor=GOLD,
        linewidth=0.7,
        label=f"best score from {len(nulls)} searches on shuffled targets",
    )
    ax.axvline(reported, color=BLUE, linewidth=2.2, label="score the search reported")

    # Keep the label block on whichever side of the line has room, so it never
    # collides with the axis edge or with the histogram.
    ax.set_ylim(top=ax.get_ylim()[1] * 1.16)
    top = ax.get_ylim()[1]
    on_the_right = (reported - lo) / (hi - lo) > 0.55
    ax.annotate(
        f"reported {reported:.3f}\np = {summary['p_value']:.3f}",
        xy=(reported, top * 0.98),
        xytext=(-9 if on_the_right else 9, 0),
        textcoords="offset points",
        color=BLUE,
        fontsize=9.2,
        fontweight="bold",
        ha="right" if on_the_right else "left",
        va="top",
        linespacing=1.5,
    )
    ax.set_title(title, loc="left", fontweight="bold", pad=17)
    ax.text(
        0.0,
        1.03,
        subtitle,
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8.6,
        va="bottom",
    )
    ax.set_xlabel("ROC AUC reached by the best of 24 candidates")
    ax.set_xlim(lo, hi)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/images", type=Path)
    parser.add_argument("--permutations", default=200, type=int)
    args = parser.parse_args()

    configure()
    noise, signal = run(args.permutations)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), constrained_layout=True)
    panel(
        axes[0],
        noise,
        "The target is pure noise",
        "the reported score sits inside what shuffling already reaches",
    )
    panel(
        axes[1],
        signal,
        "The target carries real signal",
        "shuffling never comes close",
    )
    axes[0].set_ylabel("number of searches")
    fig.suptitle(
        "A tuned score is only worth what the same search cannot reach on shuffled targets",
        fontsize=11.4,
        fontweight="bold",
        x=0.005,
        ha="left",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncols=2,
        bbox_to_anchor=(0.5, -0.055),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "selection-null.png"
    fig.savefig(
        destination,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.14,
        metadata={"Software": "scripts/make_readme_figure.py"},
    )
    plt.close(fig)

    print(f"\nwrote {destination}")
    for name, result in (("noise", noise), ("signal", signal)):
        summary = result.permutation_summary()
        print(
            f"  {name:6} reported={result.reported_score:.4f}"
            f"  null_mean={summary['null_mean_best_score']:.4f}"
            f"  null_max={summary['null_max_best_score']:.4f}"
            f"  p={summary['p_value']:.4f}"
        )


if __name__ == "__main__":
    main()
