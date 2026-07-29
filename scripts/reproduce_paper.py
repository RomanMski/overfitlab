#!/usr/bin/env python3
"""Reproduce the controlled experiments and figures in the StressFold paper.

The script is deliberately independent of the StressFold package API.  It uses
only common scientific Python libraries and writes deterministic artifacts to
``paper/figures`` and ``paper/tables``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


SEED = 20260729

INK = "#20272E"
MUTED = "#66717C"
GRID = "#D8DEE4"
BLUE = "#2F5D7E"
BLUE_LIGHT = "#9DB8C9"
GOLD = "#C18F36"
GOLD_LIGHT = "#E6D1A4"
ORANGE = "#B65C32"
OPEN = "#F7F9FA"


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.2,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.8,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.75,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.2,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axes(ax: mpl.axes.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.17,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=INK,
    )


def save_figure(fig: mpl.figure.Figure, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path_without_suffix.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={
            "Title": path_without_suffix.name,
            "Author": "StressFold contributors",
            "Creator": "scripts/reproduce_paper.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        path_without_suffix.with_suffix(".png"),
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={"Software": "scripts/reproduce_paper.py"},
    )
    plt.close(fig)


def covariance_matrix(d: int = 8, rho: float = 0.55) -> np.ndarray:
    idx = np.arange(d)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def true_probability(x: np.ndarray) -> np.ndarray:
    eta = (
        1.60 * np.sin(x[:, 0])
        + 1.10 * x[:, 1]
        - 0.75 * x[:, 2] * x[:, 3]
        + 0.45 * (x[:, 4] ** 2 - 1.0)
    )
    return expit(eta)


def sample_generalization_dgp(
    rng: np.random.Generator, n: int, covariance: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rng.multivariate_normal(np.zeros(covariance.shape[0]), covariance, size=n)
    p = true_probability(x)
    y = rng.binomial(1, p).astype(int)
    return x, y, p


def expected_brier(p_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Expected Brier loss under Bernoulli(p_true), including Bayes noise."""
    return float(np.mean(p_true * (1.0 - p_true) + (p_true - p_pred) ** 2))


def empirical_brier(y: np.ndarray, p_pred: np.ndarray) -> float:
    return float(np.mean((y - p_pred) ** 2))


def mean_interval(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = 1.96 * float(values.std(ddof=1)) / math.sqrt(values.size)
    return mean, mean - half, mean + half


@dataclass(frozen=True)
class GeneralizationResult:
    depth_labels: list[str]
    train_mean: np.ndarray
    train_lo: np.ndarray
    train_hi: np.ndarray
    test_mean: np.ndarray
    test_lo: np.ndarray
    test_hi: np.ndarray
    noise_levels: np.ndarray
    regularized_skill: np.ndarray
    regularized_lo: np.ndarray
    regularized_hi: np.ndarray
    unpruned_skill: np.ndarray
    unpruned_lo: np.ndarray
    unpruned_hi: np.ndarray
    fixed_models: tuple[DecisionTreeClassifier, DecisionTreeClassifier]
    paired_models: tuple[DecisionTreeClassifier, DecisionTreeClassifier]
    fixed_audit_x: np.ndarray
    fixed_audit_p: np.ndarray
    covariance: np.ndarray


def run_generalization_experiment(rng: np.random.Generator) -> GeneralizationResult:
    covariance = covariance_matrix()
    depths: list[int | None] = [1, 2, 3, 5, 8, None]
    labels = ["1", "2", "3", "5", "8", "unpruned"]
    train_losses = np.empty((54, len(depths)))
    test_losses = np.empty_like(train_losses)

    for rep in range(train_losses.shape[0]):
        x_train, y_train, _ = sample_generalization_dgp(rng, 320, covariance)
        x_test, _, p_test = sample_generalization_dgp(rng, 2600, covariance)
        for j, depth in enumerate(depths):
            model = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=1,
                random_state=10_000 + rep * 17 + j,
            )
            model.fit(x_train, y_train)
            train_losses[rep, j] = empirical_brier(
                y_train, model.predict_proba(x_train)[:, 1]
            )
            test_losses[rep, j] = expected_brier(
                p_test, model.predict_proba(x_test)[:, 1]
            )

    train_stats = np.array([mean_interval(train_losses[:, j]) for j in range(len(depths))])
    test_stats = np.array([mean_interval(test_losses[:, j]) for j in range(len(depths))])

    x_train, y_train, _ = sample_generalization_dgp(rng, 420, covariance)
    x_audit, _, p_audit = sample_generalization_dgp(rng, 3200, covariance)
    regularized = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=6, random_state=31_415
    ).fit(x_train, y_train)
    unpruned = DecisionTreeClassifier(
        max_depth=None, min_samples_leaf=1, random_state=27_182
    ).fit(x_train, y_train)
    depth_four = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=6, random_state=16_180
    ).fit(x_train, y_train)

    noise_levels = np.array([0.00, 0.08, 0.16, 0.28, 0.42, 0.60, 0.85])
    chol = np.linalg.cholesky(covariance)
    prior = float(y_train.mean())
    skill_draws = np.empty((44, noise_levels.size, 2))

    clean_losses = np.array(
        [
            expected_brier(p_audit, regularized.predict_proba(x_audit)[:, 1]),
            expected_brier(p_audit, unpruned.predict_proba(x_audit)[:, 1]),
        ]
    )
    clean_reference = expected_brier(p_audit, np.full_like(p_audit, prior))

    for draw in range(skill_draws.shape[0]):
        base_noise = rng.standard_normal(x_audit.shape) @ chol.T
        for j, severity in enumerate(noise_levels):
            x_shift = x_audit + severity * base_noise
            # The perturbation is measurement error: it changes the observed
            # covariates, not the latent case or its outcome distribution.
            p_shift = p_audit
            reference = clean_reference
            for k, model in enumerate((regularized, unpruned)):
                loss = expected_brier(p_shift, model.predict_proba(x_shift)[:, 1])
                denominator = clean_reference - clean_losses[k]
                skill_draws[draw, j, k] = (reference - loss) / denominator

    skill_mean = skill_draws.mean(axis=0)
    skill_half = 1.96 * skill_draws.std(axis=0, ddof=1) / math.sqrt(skill_draws.shape[0])

    return GeneralizationResult(
        depth_labels=labels,
        train_mean=train_stats[:, 0],
        train_lo=train_stats[:, 1],
        train_hi=train_stats[:, 2],
        test_mean=test_stats[:, 0],
        test_lo=test_stats[:, 1],
        test_hi=test_stats[:, 2],
        noise_levels=noise_levels,
        regularized_skill=skill_mean[:, 0],
        regularized_lo=skill_mean[:, 0] - skill_half[:, 0],
        regularized_hi=skill_mean[:, 0] + skill_half[:, 0],
        unpruned_skill=skill_mean[:, 1],
        unpruned_lo=skill_mean[:, 1] - skill_half[:, 1],
        unpruned_hi=skill_mean[:, 1] + skill_half[:, 1],
        fixed_models=(regularized, unpruned),
        paired_models=(regularized, depth_four),
        fixed_audit_x=x_audit,
        fixed_audit_p=p_audit,
        covariance=covariance,
    )


def plot_estimands(result: GeneralizationResult, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), constrained_layout=True)
    x = np.arange(len(result.depth_labels))

    ax = axes[0]
    ax.plot(x, result.train_mean, marker="o", color=GOLD, label="Training loss")
    ax.fill_between(x, result.train_lo, result.train_hi, color=GOLD_LIGHT, alpha=0.55)
    ax.plot(x, result.test_mean, marker="s", color=BLUE, label="Population loss")
    ax.fill_between(x, result.test_lo, result.test_hi, color=BLUE_LIGHT, alpha=0.45)
    ax.set_xticks(x, result.depth_labels)
    ax.set_xlabel("Tree depth")
    ax.set_ylabel("Brier loss")
    ax.set_title("Generalization across model complexity", loc="left", pad=20)
    ax.text(
        0.0,
        1.015,
        "Mean and 95% Monte Carlo interval; 54 training samples",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=7.0,
        va="bottom",
    )
    ax.legend(loc="upper left")
    clean_axes(ax)
    panel_label(ax, "A")

    ax = axes[1]
    ax.axhline(0.0, color=INK, linewidth=0.8, linestyle="--", label="Constant predictor")
    ax.plot(
        result.noise_levels,
        result.regularized_skill,
        marker="o",
        color=BLUE,
        label="Depth 3",
    )
    ax.fill_between(
        result.noise_levels,
        result.regularized_lo,
        result.regularized_hi,
        color=BLUE_LIGHT,
        alpha=0.45,
    )
    ax.plot(
        result.noise_levels,
        result.unpruned_skill,
        marker="s",
        color=GOLD,
        label="Unpruned",
    )
    ax.fill_between(
        result.noise_levels,
        result.unpruned_lo,
        result.unpruned_hi,
        color=GOLD_LIGHT,
        alpha=0.55,
    )
    ax.set_xlabel(r"Correlated-noise severity $\lambda$")
    ax.set_ylabel("Retained skill")
    ax.set_title("Perturbation-response profile", loc="left", pad=20)
    ax.text(
        0.0,
        1.015,
        "Paired measurement-noise draws; latent labels remain fixed",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=7.0,
        va="bottom",
    )
    ax.legend(loc="lower left")
    clean_axes(ax)
    panel_label(ax, "B")
    save_figure(fig, output / "estimand_separation")


@dataclass(frozen=True)
class MonteCarloResult:
    r_values: np.ndarray
    paired_sd: np.ndarray
    unpaired_sd: np.ndarray
    paired_samples_r8: np.ndarray
    unpaired_samples_r8: np.ndarray
    naive_false_positive_rate: float
    full_false_positive_rate: float
    paired_ratio_r16: float


def run_monte_carlo_experiment(
    rng: np.random.Generator, generalization: GeneralizationResult
) -> MonteCarloResult:
    model_a, model_b = generalization.paired_models
    x = generalization.fixed_audit_x[:700]
    latent_p = generalization.fixed_audit_p[:700]
    covariance = generalization.covariance
    chol = np.linalg.cholesky(covariance)
    severity = 0.42
    pool_size = 1200
    loss_a = np.empty(pool_size)
    loss_b = np.empty(pool_size)

    for i in range(pool_size):
        noise = rng.standard_normal(x.shape) @ chol.T
        shifted = x + severity * noise
        loss_a[i] = expected_brier(latent_p, model_a.predict_proba(shifted)[:, 1])
        loss_b[i] = expected_brier(latent_p, model_b.predict_proba(shifted)[:, 1])

    r_values = np.array([2, 4, 8, 16, 32, 64])
    paired_sd = np.empty(r_values.size)
    unpaired_sd = np.empty(r_values.size)
    paired_r8 = np.empty(0)
    unpaired_r8 = np.empty(0)

    for j, r_count in enumerate(r_values):
        paired_samples = np.empty(900)
        unpaired_samples = np.empty(900)
        for k in range(paired_samples.size):
            paired_idx = rng.integers(0, pool_size, size=r_count)
            idx_a = rng.integers(0, pool_size, size=r_count)
            idx_b = rng.integers(0, pool_size, size=r_count)
            paired_samples[k] = np.mean(loss_b[paired_idx] - loss_a[paired_idx])
            unpaired_samples[k] = np.mean(loss_b[idx_b]) - np.mean(loss_a[idx_a])
        paired_sd[j] = paired_samples.std(ddof=1)
        unpaired_sd[j] = unpaired_samples.std(ddof=1)
        if r_count == 8:
            paired_r8 = paired_samples.copy()
            unpaired_r8 = unpaired_samples.copy()

    naive_p, full_p = selection_aware_permutation_pvalues(rng)
    naive_fpr = float(np.mean(naive_p <= 0.05))
    full_fpr = float(np.mean(full_p <= 0.05))
    r16_idx = int(np.flatnonzero(r_values == 16)[0])
    ratio = float(unpaired_sd[r16_idx] / paired_sd[r16_idx])

    return MonteCarloResult(
        r_values=r_values,
        paired_sd=paired_sd,
        unpaired_sd=unpaired_sd,
        paired_samples_r8=paired_r8,
        unpaired_samples_r8=unpaired_r8,
        naive_false_positive_rate=naive_fpr,
        full_false_positive_rate=full_fpr,
        paired_ratio_r16=ratio,
    )


def selection_aware_permutation_pvalues(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Null experiment: select the best of 40 random predictors on audit data."""
    repetitions = 140
    permutations = 199
    n = 160
    candidates = 40
    naive = np.empty(repetitions)
    full = np.empty(repetitions)

    for rep in range(repetitions):
        y = rng.integers(0, 2, size=n)
        predictions = rng.integers(0, 2, size=(candidates, n))
        observed_scores = np.mean(predictions == y[None, :], axis=1)
        selected = int(np.argmax(observed_scores))
        observed = float(observed_scores[selected])
        naive_null = np.empty(permutations)
        full_null = np.empty(permutations)
        for b in range(permutations):
            permuted = rng.permutation(y)
            scores = np.mean(predictions == permuted[None, :], axis=1)
            naive_null[b] = scores[selected]
            full_null[b] = float(scores.max())
        naive[rep] = (1.0 + np.sum(naive_null >= observed)) / (permutations + 1.0)
        full[rep] = (1.0 + np.sum(full_null >= observed)) / (permutations + 1.0)
    return naive, full


def plot_monte_carlo(result: MonteCarloResult, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35), constrained_layout=True)

    ax = axes[0]
    lo = min(result.paired_samples_r8.min(), result.unpaired_samples_r8.min())
    hi = max(result.paired_samples_r8.max(), result.unpaired_samples_r8.max())
    bins = np.linspace(lo, hi, 28)
    ax.hist(
        result.unpaired_samples_r8,
        bins=bins,
        density=True,
        histtype="stepfilled",
        color=GOLD_LIGHT,
        edgecolor=GOLD,
        linewidth=1.1,
        alpha=0.65,
        label="Independent draws",
    )
    ax.hist(
        result.paired_samples_r8,
        bins=bins,
        density=True,
        histtype="step",
        color=BLUE,
        linewidth=1.5,
        label="Paired draws",
    )
    ax.set_xlabel("Estimated loss difference")
    ax.set_ylabel("Density")
    ax.set_title("Monte Carlo estimator, $R=8$", loc="left")
    ax.legend(loc="upper left")
    clean_axes(ax)
    panel_label(ax, "A")

    ax = axes[1]
    ax.loglog(
        result.r_values,
        result.unpaired_sd,
        marker="s",
        color=GOLD,
        label="Independent draws",
    )
    ax.loglog(
        result.r_values,
        result.paired_sd,
        marker="o",
        color=BLUE,
        label="Paired draws",
    )
    ax.set_xticks(result.r_values, [str(v) for v in result.r_values])
    ax.set_xlabel("Perturbation draws $R$")
    ax.set_ylabel("Estimator SD")
    ax.set_title("Precision from common draws", loc="left")
    ax.legend(loc="upper right")
    clean_axes(ax, grid_axis="both")
    panel_label(ax, "B")

    ax = axes[2]
    rates = [result.naive_false_positive_rate, result.full_false_positive_rate]
    bars = ax.bar(
        [0, 1],
        rates,
        width=0.58,
        color=[GOLD_LIGHT, BLUE_LIGHT],
        edgecolor=[GOLD, BLUE],
        linewidth=1.0,
    )
    ax.axhline(0.05, color=INK, linestyle="--", linewidth=0.9, label="Nominal 5%")
    ax.set_xticks([0, 1], ["Hold choice\nfixed", "Repeat full\nselection"])
    ax.set_ylim(0, max(0.20, max(rates) * 1.20))
    ax.set_ylabel("False-positive rate")
    ax.set_title("Permutation falsification", loc="left")
    for bar, rate in zip(bars, rates, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.012,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color=INK,
        )
    ax.legend(loc="upper right")
    clean_axes(ax)
    panel_label(ax, "C")

    save_figure(fig, output / "paired_monte_carlo")


def sample_xor_dgp(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray]:
    y = rng.integers(0, 2, size=n)
    sign = rng.choice(np.array([-1.0, 1.0]), size=n)
    x1 = sign + rng.normal(0.0, 0.30, size=n)
    x2 = (2.0 * y - 1.0) * sign + rng.normal(0.0, 0.30, size=n)
    return np.column_stack([x1, x2]), y.astype(int)


def empirical_bootstrap(
    rng: np.random.Generator, x: np.ndarray, y: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.integers(0, x.shape[0], size=n)
    return x[idx].copy(), y[idx].copy()


def smoothed_bootstrap(
    rng: np.random.Generator, x: np.ndarray, y: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    sx, sy = empirical_bootstrap(rng, x, y, n)
    return sx + rng.normal(0.0, 0.12, size=sx.shape), sy


def independent_class_marginals(
    rng: np.random.Generator, x: np.ndarray, y: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    sy = rng.choice(y, size=n, replace=True)
    sx = np.empty((n, x.shape[1]))
    for i, label in enumerate(sy):
        pool = np.flatnonzero(y == label)
        for j in range(x.shape[1]):
            sx[i, j] = x[rng.choice(pool), j]
    return sx, sy.astype(int)


def c2st_auc(
    rng: np.random.Generator,
    real_x: np.ndarray,
    real_y: np.ndarray,
    synthetic_x: np.ndarray,
    synthetic_y: np.ndarray,
) -> float:
    n = min(real_x.shape[0], synthetic_x.shape[0], 900)
    real_idx = rng.choice(real_x.shape[0], size=n, replace=False)
    synthetic_idx = rng.choice(synthetic_x.shape[0], size=n, replace=False)
    real_joint = np.column_stack([real_x[real_idx], real_y[real_idx]])
    synthetic_joint = np.column_stack(
        [synthetic_x[synthetic_idx], synthetic_y[synthetic_idx]]
    )
    x = np.vstack([real_joint, synthetic_joint])
    labels = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    train_x, test_x, train_y, test_y = train_test_split(
        x, labels, test_size=0.40, stratify=labels, random_state=7_331
    )
    discriminator = RandomForestClassifier(
        n_estimators=180,
        max_depth=7,
        min_samples_leaf=5,
        n_jobs=1,
        random_state=9_901,
    ).fit(train_x, train_y)
    auc = float(roc_auc_score(test_y, discriminator.predict_proba(test_x)[:, 1]))
    return max(auc, 1.0 - auc)


def conditional_ks(real_x: np.ndarray, real_y: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> float:
    distances: list[float] = []
    for label in (0, 1):
        for j in range(real_x.shape[1]):
            distances.append(
                float(ks_2samp(real_x[real_y == label, j], sx[sy == label, j]).statistic)
            )
    return float(np.mean(distances))


def conditional_correlation_error(
    real_x: np.ndarray, real_y: np.ndarray, sx: np.ndarray, sy: np.ndarray
) -> float:
    errors: list[float] = []
    for label in (0, 1):
        real_corr = float(np.corrcoef(real_x[real_y == label].T)[0, 1])
        synthetic_corr = float(np.corrcoef(sx[sy == label].T)[0, 1])
        errors.append(abs(real_corr - synthetic_corr))
    return float(np.mean(errors))


def exact_copy_rate(
    train_x: np.ndarray, train_y: np.ndarray, sx: np.ndarray, sy: np.ndarray
) -> float:
    keys = {(float(row[0]), float(row[1]), int(label)) for row, label in zip(train_x, train_y, strict=True)}
    copied = sum(
        (float(row[0]), float(row[1]), int(label)) in keys
        for row, label in zip(sx, sy, strict=True)
    )
    return copied / sx.shape[0]


@dataclass(frozen=True)
class SyntheticResult:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    generated: dict[str, tuple[np.ndarray, np.ndarray]]
    diagnostics: list[dict[str, float | str]]


def run_synthetic_experiment(rng: np.random.Generator) -> SyntheticResult:
    train_x, train_y = sample_xor_dgp(rng, 850)
    test_x, test_y = sample_xor_dgp(rng, 3200)
    generated = {
        "Empirical bootstrap": empirical_bootstrap(rng, train_x, train_y, 850),
        "Smoothed bootstrap": smoothed_bootstrap(rng, train_x, train_y, 850),
        "Independent marginals": independent_class_marginals(rng, train_x, train_y, 850),
    }

    diagnostics: list[dict[str, float | str]] = []
    baseline_model = RandomForestClassifier(
        n_estimators=220,
        max_depth=7,
        min_samples_leaf=3,
        n_jobs=1,
        random_state=4_242,
    ).fit(train_x, train_y)
    baseline_accuracy = float(baseline_model.score(test_x, test_y))
    diagnostics.append(
        {
            "generator": "Real training data",
            "conditional_ks": 0.0,
            "correlation_error": 0.0,
            "c2st_auc": 0.5,
            "tstr_accuracy": baseline_accuracy,
            "copy_rate": 0.0,
        }
    )

    for j, (name, (sx, sy)) in enumerate(generated.items()):
        utility_model = RandomForestClassifier(
            n_estimators=220,
            max_depth=7,
            min_samples_leaf=3,
            n_jobs=1,
            random_state=5_000 + j,
        ).fit(sx, sy)
        diagnostics.append(
            {
                "generator": name,
                "conditional_ks": conditional_ks(test_x, test_y, sx, sy),
                "correlation_error": conditional_correlation_error(test_x, test_y, sx, sy),
                "c2st_auc": c2st_auc(rng, test_x, test_y, sx, sy),
                "tstr_accuracy": float(utility_model.score(test_x, test_y)),
                "copy_rate": exact_copy_rate(train_x, train_y, sx, sy),
            }
        )

    return SyntheticResult(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        generated=generated,
        diagnostics=diagnostics,
    )


def plot_synthetic(result: SyntheticResult, output: Path) -> None:
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.15, 2.15),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.18]},
    )

    panels = [
        ("Real holdout", result.test_x, result.test_y),
        (
            "Smoothed bootstrap",
            result.generated["Smoothed bootstrap"][0],
            result.generated["Smoothed bootstrap"][1],
        ),
        (
            "Independent marginals",
            result.generated["Independent marginals"][0],
            result.generated["Independent marginals"][1],
        ),
    ]
    plot_rng = np.random.default_rng(88_021)
    for j, (title, x, y) in enumerate(panels):
        ax = axes[j]
        take = plot_rng.choice(x.shape[0], size=min(480, x.shape[0]), replace=False)
        ax.scatter(
            x[take, 0],
            x[take, 1],
            c=np.where(y[take] == 1, BLUE, GOLD),
            s=7,
            alpha=0.60,
            linewidths=0,
        )
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("$x_1$")
        if j == 0:
            ax.set_ylabel("$x_2$")
        else:
            ax.set_yticklabels([])
        ax.set_title(title, loc="left", fontsize=8.4)
        clean_axes(ax, grid_axis="both")
        panel_label(ax, chr(ord("A") + j))

    ax = axes[3]
    names = [str(row["generator"]) for row in result.diagnostics]
    values = [float(row["tstr_accuracy"]) for row in result.diagnostics]
    short_names = ["Real", "Empirical", "Smoothed", "Marginal"]
    colors = [INK, BLUE_LIGHT, BLUE, GOLD_LIGHT]
    edges = [INK, BLUE, BLUE, GOLD]
    bars = ax.bar(np.arange(len(values)), values, color=colors, edgecolor=edges, linewidth=0.9)
    ax.axhline(0.5, color=INK, linestyle="--", linewidth=0.8, label="Chance")
    ax.set_ylim(0.4, 1.02)
    ax.set_xticks(np.arange(len(values)), short_names, rotation=28, ha="right")
    ax.set_ylabel("Real-holdout accuracy")
    ax.set_title("Train synthetic, test real", loc="left", fontsize=8.4)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.8,
        )
    clean_axes(ax)
    panel_label(ax, "D")
    _ = names  # Retained in diagnostics; short labels keep the panel legible.
    save_figure(fig, output / "synthetic_replica_audit")


def latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def write_synthetic_table(result: SyntheticResult, tables: Path) -> None:
    tables.mkdir(parents=True, exist_ok=True)
    csv_path = tables / "synthetic_diagnostics.csv"
    fieldnames = [
        "generator",
        "conditional_ks",
        "correlation_error",
        "c2st_auc",
        "tstr_accuracy",
        "copy_rate",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.diagnostics)

    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Replica & KS $\downarrow$ & Corr. err. $\downarrow$ & C2ST AUC $\downarrow$ & TSTR $\uparrow$ & Copies $\downarrow$ \\",
        r"\midrule",
    ]
    for row in result.diagnostics:
        lines.append(
            "{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.1f}\\% \\\\".format(
                latex_escape(str(row["generator"])),
                float(row["conditional_ks"]),
                float(row["correlation_error"]),
                float(row["c2st_auc"]),
                float(row["tstr_accuracy"]),
                100.0 * float(row["copy_rate"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (tables / "synthetic_diagnostics.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_table(
    generalization: GeneralizationResult,
    monte_carlo: MonteCarloResult,
    synthetic: SyntheticResult,
    tables: Path,
) -> None:
    best_idx = int(np.argmin(generalization.test_mean))
    unpruned_idx = len(generalization.depth_labels) - 1
    marginal = next(
        row for row in synthetic.diagnostics if row["generator"] == "Independent marginals"
    )
    lines = [
        r"\begin{tabular}{lp{0.54\linewidth}}",
        r"\toprule",
        r"Check & Deterministic result \\",
        r"\midrule",
        "Generalization & Population Brier is minimized at depth {} ({:.3f}); the unpruned tree reaches {:.3f}. \\\\".format(
            generalization.depth_labels[best_idx],
            generalization.test_mean[best_idx],
            generalization.test_mean[unpruned_idx],
        ),
        "Paired Monte Carlo & At $R=16$, independent perturbations have {:.2f}$\\times$ the estimator SD of common draws. \\\\".format(
            monte_carlo.paired_ratio_r16
        ),
        "Falsification & Null false-positive rates are {:.1f}\\% when model choice is held fixed and {:.1f}\\% when the full selection is repeated. \\\\".format(
            100.0 * monte_carlo.naive_false_positive_rate,
            100.0 * monte_carlo.full_false_positive_rate,
        ),
        "Synthetic audit & Independent class marginals achieve TSTR accuracy {:.3f} despite low marginal KS discrepancy {:.3f}. \\\\".format(
            float(marginal["tstr_accuracy"]), float(marginal["conditional_ks"])
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (tables / "experiment_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=repo_root / "paper",
        help="Destination containing figures/ and tables/ (default: paper/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paper_dir = args.paper_dir.resolve()
    figures = paper_dir / "figures"
    tables = paper_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    seed_sequence = np.random.SeedSequence(SEED)
    general_rng, monte_carlo_rng, synthetic_rng = [
        np.random.default_rng(seed) for seed in seed_sequence.spawn(3)
    ]

    generalization = run_generalization_experiment(general_rng)
    plot_estimands(generalization, figures)

    monte_carlo = run_monte_carlo_experiment(monte_carlo_rng, generalization)
    plot_monte_carlo(monte_carlo, figures)

    synthetic = run_synthetic_experiment(synthetic_rng)
    plot_synthetic(synthetic, figures)

    write_synthetic_table(synthetic, tables)
    write_summary_table(generalization, monte_carlo, synthetic, tables)

    print(f"StressFold paper artifacts written to {paper_dir}")
    print(f"  figures: {len(list(figures.glob('*.pdf')))} PDF + {len(list(figures.glob('*.png')))} PNG")
    print(f"  tables:  {len(list(tables.glob('*')))} files")
    print(f"  fixed seed: {SEED}")


if __name__ == "__main__":
    main()
