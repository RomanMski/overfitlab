"""The fold-aware StressFold audit engine."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone

from .config import AuditConfig, StressSuite
from .metrics import evaluate, metric_degradation, resolve_metrics
from .protocols import repeated_holdout
from .random import derive_seed
from .results import AuditResult, Variant
from .stressors import (
    inject_feature_noise,
    inject_label_noise,
    inject_missingness,
    subsample_training,
)


def _coerce_data(X: Any, y: Any, task: str) -> tuple[pd.DataFrame, pd.Series]:
    if isinstance(X, pd.DataFrame):
        features = X.copy(deep=True)
    else:
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be a two-dimensional table")
        features = pd.DataFrame(
            array, columns=[f"x{column}" for column in range(array.shape[1])]
        )
    if features.columns.has_duplicates:
        raise ValueError("X column names must be unique")
    if isinstance(y, pd.Series):
        if isinstance(X, pd.DataFrame) and not X.index.equals(y.index):
            raise ValueError(
                "X and y pandas indices differ; align them explicitly before audit()"
            )
        target = y.copy(deep=True)
    else:
        target_array = np.asarray(y)
        if target_array.ndim == 2 and target_array.shape[1] == 1:
            target_array = target_array.reshape(-1)
        elif target_array.ndim != 1:
            raise ValueError("y must be one-dimensional or a single-column array")
        target = pd.Series(target_array, name="target")
    if len(features) != len(target):
        raise ValueError(
            f"X and y have different row counts: {len(features)} and {len(target)}"
        )
    if len(features) < 4:
        raise ValueError("StressFold requires at least four rows")
    if features.shape[1] < 1:
        raise ValueError("X must contain at least one feature")
    if target.isna().any():
        raise ValueError("y contains missing values")
    if task == "binary_classification":
        observed = pd.unique(target)
        if len(observed) != 2:
            raise ValueError(
                f"binary classification requires exactly two target classes; found {len(observed)}"
            )
    else:
        try:
            target = pd.to_numeric(target, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("regression targets must be numeric") from exc
    features = features.reset_index(drop=True)
    target = target.reset_index(drop=True)
    if target.name is None:
        target.name = "target"
    return features, target


def _fingerprint(X: pd.DataFrame, y: pd.Series) -> str:
    digest = hashlib.blake2b(digest_size=16, person=b"stressfold-data")
    digest.update(
        repr(
            [
                (str(column), str(dtype))
                for column, dtype in zip(X.columns, X.dtypes, strict=True)
            ]
        ).encode()
    )
    digest.update(
        pd.util.hash_pandas_object(X, index=True, categorize=True).to_numpy().tobytes()
    )
    digest.update(
        pd.util.hash_pandas_object(y, index=True, categorize=True).to_numpy().tobytes()
    )
    return digest.hexdigest()


def _clone_with_seed(estimator: Any, seed: int) -> Any:
    try:
        instance = clone(estimator)
    except TypeError:
        instance = deepcopy(estimator)
    if hasattr(instance, "get_params") and hasattr(instance, "set_params"):
        parameters = instance.get_params(deep=True)
        updates = {
            name: seed
            for name, value in parameters.items()
            if (name == "random_state" or name.endswith("__random_state"))
            and value is None
        }
        if updates:
            instance.set_params(**updates)
    return instance


def _fit(estimator: Any, X: pd.DataFrame, y: pd.Series, seed: int) -> Any:
    model = _clone_with_seed(estimator, seed)
    model.fit(X, y)
    return model


def _append_records(
    records: list[dict[str, Any]],
    values: dict[str, float],
    baseline: dict[str, float],
    *,
    repeat: int,
    scenario_repeat: int,
    experiment: str,
    evidence: str,
    evaluation_name: str,
    level: float,
    split_seed: int,
    operation_seed: int,
    model_seed: int,
    n_train: int,
    n_test: int,
) -> None:
    for metric, value in values.items():
        baseline_value = float(baseline[metric])
        numeric_value = float(value)
        records.append(
            {
                "repeat": repeat,
                "scenario_repeat": scenario_repeat,
                "experiment": experiment,
                "evidence": evidence,
                "evaluation": evaluation_name,
                "level": float(level),
                "metric": metric,
                "value": numeric_value,
                "baseline_value": baseline_value,
                "raw_delta": numeric_value - baseline_value,
                "degradation": metric_degradation(
                    metric, numeric_value, baseline_value
                ),
                "split_seed": split_seed,
                "operation_seed": operation_seed,
                "model_seed": model_seed,
                "n_train": n_train,
                "n_test": n_test,
            }
        )


def _variant(
    variants: list[Variant],
    enabled: bool,
    *,
    repeat: int,
    experiment: str,
    level: float,
    partition: str,
    seed: int,
    X: pd.DataFrame,
    y: pd.Series,
    row_indices: Iterable[int],
    metadata: dict[str, Any],
) -> None:
    if not enabled:
        return
    variants.append(
        Variant(
            repeat=repeat,
            experiment=experiment,
            level=float(level),
            partition=partition,
            seed=seed,
            X=X.reset_index(drop=True).copy(deep=True),
            y=y.reset_index(drop=True).copy(deep=True),
            row_indices=np.asarray(list(row_indices), dtype=int),
            metadata=metadata,
        )
    )


def audit(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    config: AuditConfig,
    suite: StressSuite | None = None,
) -> AuditResult:
    """Run a paired, repeated stress audit of a tabular estimator.

    The unit under test is the complete training procedure, not one saved
    model object. ``estimator`` is cloned and refitted on the training rows of
    every split, so an already-fitted instance has its learned state discarded
    and relearned. That is what keeps preprocessing and selection inside the
    audit rather than leaking across the split. To evaluate one frozen model
    instead, score it yourself on data this function never sees.

    Within a split the fitted model is then held fixed: feature noise and
    missingness are evaluation-set robustness probes against it. Label noise
    and train-fraction paths refit and therefore measure training stability.
    Permutations are a null control. They are kept separate because none is,
    by itself, proof of overfitting.
    """

    selected_suite = suite or StressSuite.standard()
    features, target = _coerce_data(X, y, config.task)
    metric_names = resolve_metrics(config.task, config.metrics)
    records: list[dict[str, Any]] = []
    seeds: list[dict[str, Any]] = []
    variants: list[Variant] = []
    errors: list[dict[str, Any]] = []

    for split in repeated_holdout(
        len(features),
        target.to_numpy(),
        task=config.task,
        repeats=config.repeats,
        test_size=config.test_size,
        random_state=config.random_state,
    ):
        repeat = split.repeat
        X_train = features.iloc[split.train_indices].copy()
        y_train = target.iloc[split.train_indices].copy()
        X_test = features.iloc[split.test_indices].copy()
        y_test = target.iloc[split.test_indices].copy()
        model_seed = derive_seed(config.random_state, "model", repeat)
        seeds.extend(
            [
                {
                    "repeat": repeat,
                    "operation": "split",
                    "experiment": "baseline",
                    "level": 0.0,
                    "seed": split.seed,
                    "stratified": split.stratified,
                },
                {
                    "repeat": repeat,
                    "operation": "fit",
                    "experiment": "baseline",
                    "level": 0.0,
                    "seed": model_seed,
                },
            ]
        )
        baseline_model = _fit(estimator, X_train, y_train, model_seed)
        baseline_train = evaluate(
            baseline_model,
            X_train,
            y_train,
            task=config.task,
            metrics=metric_names,
            positive_label=config.positive_label,
        )
        baseline_test = evaluate(
            baseline_model,
            X_test,
            y_test,
            task=config.task,
            metrics=metric_names,
            positive_label=config.positive_label,
        )
        common = {
            "repeat": repeat,
            "scenario_repeat": 0,
            "experiment": "baseline",
            "evidence": "generalization",
            "level": 0.0,
            "split_seed": split.seed,
            "operation_seed": model_seed,
            "model_seed": model_seed,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        _append_records(
            records, baseline_train, baseline_train, evaluation_name="train", **common
        )
        _append_records(
            records, baseline_test, baseline_test, evaluation_name="test", **common
        )

        for experiment, levels, function in (
            ("feature_noise", selected_suite.feature_noise, inject_feature_noise),
            ("missingness", selected_suite.missingness, inject_missingness),
        ):
            for level in levels:
                operation_seed = derive_seed(
                    config.random_state, experiment, repeat, level
                )
                seeds.append(
                    {
                        "repeat": repeat,
                        "operation": "perturb_test",
                        "experiment": experiment,
                        "level": level,
                        "seed": operation_seed,
                    }
                )
                try:
                    if experiment == "feature_noise":
                        stressed = function(
                            X_test,
                            level,
                            X_train=X_train,
                            random_state=operation_seed,
                        )
                    else:
                        stressed = function(X_test, level, random_state=operation_seed)
                    X_stressed = stressed.data
                    values = evaluate(
                        baseline_model,
                        X_stressed,
                        y_test,
                        task=config.task,
                        metrics=metric_names,
                        positive_label=config.positive_label,
                    )
                    _append_records(
                        records,
                        values,
                        baseline_test,
                        repeat=repeat,
                        scenario_repeat=0,
                        experiment=experiment,
                        evidence="prediction_robustness",
                        evaluation_name="test",
                        level=level,
                        split_seed=split.seed,
                        operation_seed=operation_seed,
                        model_seed=model_seed,
                        n_train=len(X_train),
                        n_test=len(X_test),
                    )
                    _variant(
                        variants,
                        config.store_variants,
                        repeat=repeat,
                        experiment=experiment,
                        level=level,
                        partition="test",
                        seed=operation_seed,
                        X=X_stressed,
                        y=y_test,
                        row_indices=split.test_indices,
                        metadata={"split_seed": split.seed, **stressed.metadata},
                    )
                except (
                    Exception
                ) as exc:  # one unsupported stressor must not erase the audit
                    errors.append(
                        {
                            "repeat": repeat,
                            "experiment": experiment,
                            "level": level,
                            "seed": operation_seed,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )

        for level in selected_suite.label_noise:
            operation_seed = derive_seed(
                config.random_state, "label_noise", repeat, level
            )
            seeds.append(
                {
                    "repeat": repeat,
                    "operation": "perturb_train_labels",
                    "experiment": "label_noise",
                    "level": level,
                    "seed": operation_seed,
                }
            )
            try:
                stressed = inject_label_noise(
                    y_train,
                    level,
                    task=config.task,
                    random_state=operation_seed,
                )
                y_stressed = stressed.data
                model = _fit(estimator, X_train, y_stressed, model_seed)
                train_values = evaluate(
                    model,
                    X_train,
                    y_stressed,
                    task=config.task,
                    metrics=metric_names,
                    positive_label=config.positive_label,
                )
                test_values = evaluate(
                    model,
                    X_test,
                    y_test,
                    task=config.task,
                    metrics=metric_names,
                    positive_label=config.positive_label,
                )
                for evaluation_name, values, baseline in (
                    ("train_fit", train_values, baseline_train),
                    ("test", test_values, baseline_test),
                ):
                    _append_records(
                        records,
                        values,
                        baseline,
                        repeat=repeat,
                        scenario_repeat=0,
                        experiment="label_noise",
                        evidence="training_stability",
                        evaluation_name=evaluation_name,
                        level=level,
                        split_seed=split.seed,
                        operation_seed=operation_seed,
                        model_seed=model_seed,
                        n_train=len(X_train),
                        n_test=len(X_test),
                    )
                _variant(
                    variants,
                    config.store_variants,
                    repeat=repeat,
                    experiment="label_noise",
                    level=level,
                    partition="train",
                    seed=operation_seed,
                    X=X_train,
                    y=y_stressed,
                    row_indices=split.train_indices,
                    metadata={"split_seed": split.seed, **stressed.metadata},
                )
            except Exception as exc:
                errors.append(
                    {
                        "repeat": repeat,
                        "experiment": "label_noise",
                        "level": level,
                        "seed": operation_seed,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

        for fraction in selected_suite.train_fraction:
            operation_seed = derive_seed(
                config.random_state, "train_fraction", repeat, fraction
            )
            seeds.append(
                {
                    "repeat": repeat,
                    "operation": "subsample_train",
                    "experiment": "train_fraction",
                    "level": fraction,
                    "seed": operation_seed,
                }
            )
            try:
                stressed = subsample_training(
                    X_train,
                    y_train,
                    fraction,
                    task=config.task,
                    random_state=operation_seed,
                )
                X_subsample, y_subsample = stressed.data
                model = _fit(estimator, X_subsample, y_subsample, model_seed)
                values = evaluate(
                    model,
                    X_test,
                    y_test,
                    task=config.task,
                    metrics=metric_names,
                    positive_label=config.positive_label,
                )
                _append_records(
                    records,
                    values,
                    baseline_test,
                    repeat=repeat,
                    scenario_repeat=0,
                    experiment="train_fraction",
                    evidence="training_stability",
                    evaluation_name="test",
                    level=fraction,
                    split_seed=split.seed,
                    operation_seed=operation_seed,
                    model_seed=model_seed,
                    n_train=len(X_subsample),
                    n_test=len(X_test),
                )
                selected_indices = np.asarray(X_subsample.index, dtype=int)
                _variant(
                    variants,
                    config.store_variants,
                    repeat=repeat,
                    experiment="train_fraction",
                    level=fraction,
                    partition="train",
                    seed=operation_seed,
                    X=X_subsample,
                    y=y_subsample,
                    row_indices=selected_indices,
                    metadata={"split_seed": split.seed, **stressed.metadata},
                )
            except Exception as exc:
                errors.append(
                    {
                        "repeat": repeat,
                        "experiment": "train_fraction",
                        "level": fraction,
                        "seed": operation_seed,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

        for permutation in range(selected_suite.permutation_repeats):
            operation_seed = derive_seed(
                config.random_state, "permutation_null", repeat, permutation
            )
            seeds.append(
                {
                    "repeat": repeat,
                    "operation": "permute_train_labels",
                    "experiment": "permutation_null",
                    "level": 1.0,
                    "scenario_repeat": permutation,
                    "seed": operation_seed,
                }
            )
            rng = np.random.default_rng(operation_seed)
            y_permuted = pd.Series(
                rng.permutation(y_train.to_numpy()),
                index=y_train.index,
                name=y_train.name,
            )
            try:
                model = _fit(estimator, X_train, y_permuted, model_seed)
                values = evaluate(
                    model,
                    X_test,
                    y_test,
                    task=config.task,
                    metrics=metric_names,
                    positive_label=config.positive_label,
                )
                _append_records(
                    records,
                    values,
                    baseline_test,
                    repeat=repeat,
                    scenario_repeat=permutation,
                    experiment="permutation_null",
                    evidence="null_control",
                    evaluation_name="test",
                    level=1.0,
                    split_seed=split.seed,
                    operation_seed=operation_seed,
                    model_seed=model_seed,
                    n_train=len(X_train),
                    n_test=len(X_test),
                )
                _variant(
                    variants,
                    config.store_variants,
                    repeat=repeat,
                    experiment="permutation_null",
                    level=1.0,
                    partition="train",
                    seed=operation_seed,
                    X=X_train,
                    y=y_permuted,
                    row_indices=split.train_indices,
                    metadata={"split_seed": split.seed, "permutation": permutation},
                )
            except Exception as exc:
                errors.append(
                    {
                        "repeat": repeat,
                        "scenario_repeat": permutation,
                        "experiment": "permutation_null",
                        "level": 1.0,
                        "seed": operation_seed,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

    return AuditResult(
        config=config,
        suite=selected_suite,
        records=records,
        seeds=seeds,
        variants=variants,
        errors=errors,
        data_fingerprint=_fingerprint(features, target),
        estimator=repr(estimator),
        n_samples=len(features),
        n_features=features.shape[1],
    )
