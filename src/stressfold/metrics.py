"""Metric evaluation with explicit directionality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    name: str
    higher_is_better: bool
    label: str


METRICS: dict[str, MetricDefinition] = {
    "roc_auc": MetricDefinition("roc_auc", True, "ROC AUC"),
    "accuracy": MetricDefinition("accuracy", True, "Accuracy"),
    "balanced_accuracy": MetricDefinition(
        "balanced_accuracy", True, "Balanced accuracy"
    ),
    "log_loss": MetricDefinition("log_loss", False, "Log loss"),
    "brier": MetricDefinition("brier", False, "Brier score"),
    "rmse": MetricDefinition("rmse", False, "RMSE"),
    "mae": MetricDefinition("mae", False, "MAE"),
    "r2": MetricDefinition("r2", True, "R²"),
}

_ALLOWED = {
    "binary_classification": {
        "roc_auc",
        "accuracy",
        "balanced_accuracy",
        "log_loss",
        "brier",
    },
    "regression": {"rmse", "mae", "r2"},
}


def resolve_metrics(task: str, requested: Iterable[str] | None) -> tuple[str, ...]:
    defaults = {
        "binary_classification": ("roc_auc", "log_loss", "balanced_accuracy"),
        "regression": ("rmse", "mae", "r2"),
    }
    metrics = tuple(requested) if requested is not None else defaults[task]
    unsupported = [metric for metric in metrics if metric not in _ALLOWED[task]]
    if unsupported:
        raise ValueError(f"Metrics {unsupported!r} are not valid for task {task!r}")
    return metrics


def metric_degradation(metric: str, value: float, baseline: float) -> float:
    """Return a signed effect where positive always means worse."""

    if not np.isfinite(value) or not np.isfinite(baseline):
        return float("nan")
    return float(
        baseline - value if METRICS[metric].higher_is_better else value - baseline
    )


def _positive_label(model: Any, y: np.ndarray, configured: Any | None) -> Any:
    observed = list(dict.fromkeys(np.asarray(y).tolist()))
    if configured is not None:
        if configured not in observed:
            raise ValueError(f"positive_label {configured!r} does not occur in y")
        return configured
    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) == 2:
        return classes[1]
    try:
        return sorted(observed)[-1]
    except TypeError:
        return observed[-1]


def _classification_outputs(
    model: Any, X: Any, y: np.ndarray, positive_label: Any | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(model.predict(X))
    positive = _positive_label(model, y, positive_label)
    y_binary = (np.asarray(y) == positive).astype(int)
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        classes = list(getattr(model, "classes_", []))
        column = (
            classes.index(positive)
            if positive in classes
            else probabilities.shape[1] - 1
        )
        score = probabilities[:, column]
    elif hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        if raw.ndim == 2:
            classes = list(getattr(model, "classes_", []))
            column = (
                classes.index(positive) if positive in classes else raw.shape[1] - 1
            )
            raw = raw[:, column]
        score = expit(raw)
    else:
        score = (predicted == positive).astype(float)
    return (
        predicted,
        np.clip(np.asarray(score, dtype=float), 1e-15, 1.0 - 1e-15),
        y_binary,
    )


def evaluate(
    model: Any,
    X: Any,
    y: Any,
    *,
    task: str,
    metrics: Iterable[str],
    positive_label: Any | None = None,
) -> dict[str, float]:
    """Evaluate natural-scale metrics, leaving losses un-negated."""

    y_array = np.asarray(y)
    values: dict[str, float] = {}
    if task == "binary_classification":
        predicted, score, y_binary = _classification_outputs(
            model, X, y_array, positive_label
        )
        for metric in metrics:
            if metric == "roc_auc":
                values[metric] = (
                    float(roc_auc_score(y_binary, score))
                    if len(np.unique(y_binary)) == 2
                    else float("nan")
                )
            elif metric == "accuracy":
                values[metric] = float(accuracy_score(y_array, predicted))
            elif metric == "balanced_accuracy":
                values[metric] = float(balanced_accuracy_score(y_array, predicted))
            elif metric == "log_loss":
                values[metric] = float(log_loss(y_binary, score, labels=[0, 1]))
            elif metric == "brier":
                values[metric] = float(brier_score_loss(y_binary, score))
    else:
        predicted = np.asarray(model.predict(X), dtype=float)
        truth = np.asarray(y, dtype=float)
        for metric in metrics:
            if metric == "rmse":
                values[metric] = float(np.sqrt(mean_squared_error(truth, predicted)))
            elif metric == "mae":
                values[metric] = float(mean_absolute_error(truth, predicted))
            elif metric == "r2":
                values[metric] = float(r2_score(truth, predicted))
    return values
