"""Selection-aware auditing of a hyperparameter search.

``audit()`` audits one already-fitted estimator.  This module audits the
*search* that produced it, which is where most reported optimism comes from.
A search reports the best score it found across many candidate configurations,
and that best score is biased upward by the act of selecting it.

Three measurements are produced, and none of them is combined with the others.

Selection optimism
    The search is wrapped in an outer holdout it never sees.  The difference
    between the score the search reported to itself and the score its chosen
    configuration achieves outside the search is the optimism.

Selection-aware permutation null
    The complete search is rerun against permuted targets.  The resulting null
    distribution describes what a search of this size reaches when no signal
    exists at all.  Holding the winning configuration fixed and permuting
    around it instead would understate the null badly.

Winner stability
    The complete search is rerun on perturbed copies of the table.  When the
    winning configuration changes under mild jitter, the specific settings the
    search picked are not uniquely justified and should not be reported as
    optimal.  This is a claim about the configuration, not about the score.  A
    dataset with real signal can still hand back a different winner every run,
    because many configurations are close to equivalent.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import check_scoring

from .config import normalize_task
from .protocols import repeated_holdout
from .random import derive_seed
from .stressors import inject_feature_noise

__all__ = [
    "SearchAuditConfig",
    "SearchAuditResult",
    "audit_search",
]


def _levels(values: Iterable[float]) -> tuple[float, ...]:
    parsed = {float(value) for value in values}
    parsed.add(0.0)
    for value in parsed:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"noise_levels must be finite and non-negative; got {value}")
    return tuple(sorted(parsed))


@dataclass(frozen=True, slots=True)
class SearchAuditConfig:
    """Controls a selection-aware search audit.

    The audit reruns the supplied search many times, so the cost is roughly
    ``outer_repeats + permutation_repeats + noise_repeats * (len(noise_levels) - 1)``
    multiplied by the cost of one search.  Start from :meth:`quick` on a real
    grid before running the defaults.
    """

    task: str
    outer_repeats: int = 5
    test_size: float = 0.25
    random_state: int = 0
    interval: float = 0.90
    permutation_repeats: int = 20
    noise_levels: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)
    noise_repeats: int = 3
    verbose: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", normalize_task(self.task))
        object.__setattr__(self, "noise_levels", _levels(self.noise_levels))
        if self.outer_repeats < 2:
            raise ValueError("outer_repeats must be at least 2")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must lie strictly between 0 and 1")
        if not 0.0 < self.interval < 1.0:
            raise ValueError("interval must lie strictly between 0 and 1")
        if self.permutation_repeats < 0:
            raise ValueError("permutation_repeats cannot be negative")
        if self.noise_repeats < 1:
            raise ValueError("noise_repeats must be at least 1")
        if (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, int)
            or self.random_state < 0
        ):
            raise ValueError("random_state must be a non-negative integer")

    @classmethod
    def quick(cls, task: str, **overrides: Any) -> "SearchAuditConfig":
        values: dict[str, Any] = {
            "outer_repeats": 3,
            "permutation_repeats": 5,
            "noise_levels": (0.0, 0.25),
            "noise_repeats": 2,
        }
        values.update(overrides)
        return cls(task=task, **values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchAuditResult:
    """The outcome of a selection-aware search audit."""

    task: str
    metric: str
    n_candidates: int
    n_samples: int
    n_features: int
    reported_score: float
    reported_params: dict[str, Any]
    optimism_records: tuple[dict[str, Any], ...]
    permutation_records: tuple[dict[str, Any], ...]
    stability_records: tuple[dict[str, Any], ...]
    config: dict[str, Any]
    seeds: dict[str, int]
    errors: tuple[str, ...] = field(default=())

    # -- frames ---------------------------------------------------------

    def optimism_frame(self) -> pd.DataFrame:
        """One row per outer split, with the inner and outer scores."""

        return pd.DataFrame(list(self.optimism_records))

    def permutation_frame(self) -> pd.DataFrame:
        """One row per permuted rerun of the complete search."""

        return pd.DataFrame(list(self.permutation_records))

    def stability_frame(self) -> pd.DataFrame:
        """One row per perturbed rerun of the complete search."""

        return pd.DataFrame(list(self.stability_records))

    # -- summaries ------------------------------------------------------

    def optimism_summary(self) -> dict[str, float]:
        """Report how much the search flattered itself, on the metric scale."""

        frame = self.optimism_frame()
        if frame.empty:
            return {}
        optimism = frame["optimism"].to_numpy(dtype=float)
        inner = frame["inner_best_score"].to_numpy(dtype=float)
        outer = frame["outer_score"].to_numpy(dtype=float)
        low = (1.0 - float(self.config["interval"])) / 2.0
        return {
            "n_outer_splits": int(len(optimism)),
            "mean_inner_best_score": float(np.mean(inner)),
            "mean_outer_score": float(np.mean(outer)),
            "mean_optimism": float(np.mean(optimism)),
            "median_optimism": float(np.median(optimism)),
            "optimism_low": float(np.quantile(optimism, low)),
            "optimism_high": float(np.quantile(optimism, 1.0 - low)),
        }

    def permutation_summary(self) -> dict[str, float]:
        """Report the selection-aware permutation null and its p-value.

        Every permutation reruns the complete search, so the null describes a
        search of this size rather than a single fixed configuration.
        """

        frame = self.permutation_frame()
        if frame.empty:
            return {}
        null_scores = frame["null_best_score"].to_numpy(dtype=float)
        finite = null_scores[np.isfinite(null_scores)]
        if finite.size == 0:
            return {}
        exceedances = int(np.sum(finite >= self.reported_score))
        return {
            "n_permutations": int(finite.size),
            "reported_score": float(self.reported_score),
            "null_mean_best_score": float(np.mean(finite)),
            "null_max_best_score": float(np.max(finite)),
            "exceedances": exceedances,
            "p_value": float((1.0 + exceedances) / (finite.size + 1.0)),
        }

    def stability_summary(self) -> pd.DataFrame:
        """Report how often the reported configuration keeps winning."""

        frame = self.stability_frame()
        if frame.empty:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for level, group in frame.groupby("level", sort=True):
            scores = group["best_score"].to_numpy(dtype=float)
            finite = scores[np.isfinite(scores)]
            rows.append(
                {
                    "level": float(level),
                    "n_runs": int(len(group)),
                    "kept_reported_winner": int(group["matches_reported"].sum()),
                    "winner_retention_rate": float(group["matches_reported"].mean()),
                    "distinct_winners": int(group["params_key"].nunique()),
                    "mean_best_score": float(np.mean(finite)) if finite.size else float("nan"),
                }
            )
        return pd.DataFrame(rows)

    def summary_text(self) -> str:
        """Return a short readable account of the three measurements."""

        lines = [
            f"StressFold search audit  metric={self.metric}  task={self.task}",
            f"  data              {self.n_samples} rows, {self.n_features} features",
            f"  candidates tried  {self.n_candidates}",
            f"  reported score    {self.reported_score:.4f}",
        ]
        optimism = self.optimism_summary()
        if optimism:
            lines += [
                "",
                "Selection optimism",
                f"  the search reported   {optimism['mean_inner_best_score']:.4f}",
                f"  it actually scored    {optimism['mean_outer_score']:.4f} outside the search",
                f"  optimism              {optimism['mean_optimism']:+.4f}"
                f" (median {optimism['median_optimism']:+.4f})",
            ]
        null = self.permutation_summary()
        if null:
            lines += [
                "",
                "Selection-aware permutation null",
                f"  a search this size reaches {null['null_mean_best_score']:.4f}"
                " on shuffled targets",
                f"  best null run             {null['null_max_best_score']:.4f}",
                f"  p-value                   {null['p_value']:.4f}"
                f" ({null['exceedances']} of {null['n_permutations']} matched or beat it)",
            ]
        stability = self.stability_summary()
        if not stability.empty:
            lines += ["", "Winner stability under feature noise"]
            for row in stability.itertuples(index=False):
                lines.append(
                    f"  level {row.level:<5} kept the reported winner"
                    f" {row.kept_reported_winner}/{row.n_runs} times,"
                    f" {row.distinct_winners} distinct winners"
                )
        if self.errors:
            lines += ["", f"Errors recorded: {len(self.errors)}"]
        return "\n".join(lines)

    # -- artifacts ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "stressfold_search_audit",
            "task": self.task,
            "metric": self.metric,
            "data": {"n_samples": self.n_samples, "n_features": self.n_features},
            "search": {
                "n_candidates": self.n_candidates,
                "reported_score": self.reported_score,
                "reported_params": _json_safe(self.reported_params),
            },
            "config": self.config,
            "seeds": self.seeds,
            "selection_optimism": self.optimism_summary(),
            "permutation_null": self.permutation_summary(),
            "winner_stability": self.stability_summary().to_dict(orient="records"),
            "records": {
                "optimism": [_json_safe(record) for record in self.optimism_records],
                "permutation": [
                    _json_safe(record) for record in self.permutation_records
                ],
                "stability": [_json_safe(record) for record in self.stability_records],
            },
            "errors": list(self.errors),
        }

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False, default=str),
            encoding="utf-8",
        )
        return destination


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _params_key(params: dict[str, Any]) -> str:
    return json.dumps(_json_safe(params), sort_keys=True, default=str)


_SEARCH_HINT = (
    "audit_search expects an unfitted scikit-learn search object such as "
    "GridSearchCV or RandomizedSearchCV, which exposes an inner estimator and "
    "sets best_score_ and best_params_ once fitted. A plain estimator performs "
    "no selection, so there is nothing to audit. Use audit() for that. Raw "
    "Optuna studies are not supported either, so wrap them in OptunaSearchCV "
    "or an equivalent estimator."
)


def _require_search(search: Any) -> None:
    for attribute in ("fit", "get_params", "estimator"):
        if not hasattr(search, attribute):
            raise TypeError(_SEARCH_HINT)


def _fit_search(search: Any, X: pd.DataFrame, y: pd.Series, seed: int) -> Any:
    try:
        instance = clone(search)
    except TypeError:
        instance = deepcopy(search)
    parameters = instance.get_params(deep=True)
    updates = {
        name: seed
        for name, value in parameters.items()
        if (name == "random_state" or name.endswith("__random_state")) and value is None
    }
    if updates:
        instance.set_params(**updates)
    instance.fit(X, y)
    for attribute in ("best_score_", "best_params_"):
        if not hasattr(instance, attribute):
            raise TypeError(
                f"the fitted search does not expose {attribute}, so its selection "
                "cannot be audited"
            )
    return instance


def _candidate_count(fitted: Any) -> int:
    results = getattr(fitted, "cv_results_", None)
    if isinstance(results, dict) and "params" in results:
        return int(len(results["params"]))
    return 0


def _log(config: SearchAuditConfig, message: str) -> None:
    if config.verbose:
        print(f"[stressfold] {message}", file=sys.stderr, flush=True)


def audit_search(
    search: Any,
    X: Any,
    y: Any,
    *,
    config: SearchAuditConfig,
    scoring: str | Callable[..., float] | None = None,
) -> SearchAuditResult:
    """Audit the selection performed by a hyperparameter search.

    ``search`` is an unfitted scikit-learn search object.  It is refitted many
    times, so pass the search itself rather than an already-fitted one.  The
    scorer is taken from the search when it declares one, and every reported
    score is on a higher-is-better scale.
    """

    _require_search(search)
    frame, target = _coerce(X, y, config.task)
    scorer = check_scoring(
        getattr(search, "estimator", None),
        scoring=scoring if scoring is not None else getattr(search, "scoring", None),
    )
    metric_name = _scorer_name(scoring, search)
    errors: list[str] = []

    seeds = {
        "root": int(config.random_state),
        "reported": derive_seed(config.random_state, "search", "reported"),
    }

    _log(config, "fitting the search once on all rows to record what it reports")
    reported = _fit_search(search, frame, target, seeds["reported"])
    reported_score = float(reported.best_score_)
    reported_params = dict(reported.best_params_)
    reported_key = _params_key(reported_params)
    n_candidates = _candidate_count(reported)

    optimism_records = _measure_optimism(
        search, frame, target, config, scorer, errors
    )
    permutation_records = _measure_permutation_null(
        search, frame, target, config, errors
    )
    stability_records = _measure_stability(
        search, frame, target, config, reported_key, errors
    )

    return SearchAuditResult(
        task=config.task,
        metric=metric_name,
        n_candidates=n_candidates,
        n_samples=int(len(frame)),
        n_features=int(frame.shape[1]),
        reported_score=reported_score,
        reported_params=reported_params,
        optimism_records=tuple(optimism_records),
        permutation_records=tuple(permutation_records),
        stability_records=tuple(stability_records),
        config=config.to_dict(),
        seeds=seeds,
        errors=tuple(errors),
    )


def _scorer_name(scoring: Any, search: Any) -> str:
    declared = scoring if scoring is not None else getattr(search, "scoring", None)
    if isinstance(declared, str):
        return declared
    if declared is None:
        return "estimator_default_score"
    return getattr(declared, "__name__", "custom_scorer")


def _coerce(X: Any, y: Any, task: str) -> tuple[pd.DataFrame, pd.Series]:
    from .engine import _coerce_data

    return _coerce_data(X, y, task)


def _measure_optimism(
    search: Any,
    X: pd.DataFrame,
    y: pd.Series,
    config: SearchAuditConfig,
    scorer: Callable[..., float],
    errors: list[str],
) -> list[dict[str, Any]]:
    """Wrap the whole search in an outer split it never sees."""

    records: list[dict[str, Any]] = []
    splits = repeated_holdout(
        len(X),
        y.to_numpy(),
        task=config.task,
        repeats=config.outer_repeats,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    for split in splits:
        _log(config, f"selection optimism, outer split {split.repeat + 1}")
        seed = derive_seed(config.random_state, "optimism", split.repeat)
        X_train = X.iloc[split.train_indices].reset_index(drop=True)
        y_train = y.iloc[split.train_indices].reset_index(drop=True)
        X_audit = X.iloc[split.test_indices].reset_index(drop=True)
        y_audit = y.iloc[split.test_indices].reset_index(drop=True)
        try:
            fitted = _fit_search(search, X_train, y_train, seed)
            inner = float(fitted.best_score_)
            outer = float(scorer(fitted.best_estimator_, X_audit, y_audit))
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            errors.append(f"optimism split {split.repeat}: {type(exc).__name__}: {exc}")
            continue
        records.append(
            {
                "repeat": int(split.repeat),
                "seed": int(seed),
                "n_train": int(len(X_train)),
                "n_audit": int(len(X_audit)),
                "inner_best_score": inner,
                "outer_score": outer,
                "optimism": inner - outer,
                "best_params": dict(fitted.best_params_),
            }
        )
    return records


def _measure_permutation_null(
    search: Any,
    X: pd.DataFrame,
    y: pd.Series,
    config: SearchAuditConfig,
    errors: list[str],
) -> list[dict[str, Any]]:
    """Rerun the complete search against permuted targets."""

    records: list[dict[str, Any]] = []
    for index in range(config.permutation_repeats):
        _log(
            config,
            f"permutation null, rerunning the whole search {index + 1}"
            f"/{config.permutation_repeats}",
        )
        seed = derive_seed(config.random_state, "permutation", index)
        rng = np.random.default_rng(seed)
        permuted = pd.Series(
            rng.permutation(y.to_numpy()), name=y.name, index=y.index
        )
        try:
            fitted = _fit_search(search, X, permuted, seed)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            errors.append(f"permutation {index}: {type(exc).__name__}: {exc}")
            continue
        records.append(
            {
                "permutation": int(index),
                "seed": int(seed),
                "null_best_score": float(fitted.best_score_),
                "best_params": dict(fitted.best_params_),
            }
        )
    return records


def _measure_stability(
    search: Any,
    X: pd.DataFrame,
    y: pd.Series,
    config: SearchAuditConfig,
    reported_key: str,
    errors: list[str],
) -> list[dict[str, Any]]:
    """Rerun the complete search on jittered copies of the table.

    Noise scales are calibrated on the supplied table rather than on a training
    fold, because the question here is whether the search settles on the same
    configuration when the data moves, not how well that configuration
    generalizes.
    """

    records: list[dict[str, Any]] = []
    for level in config.noise_levels:
        repeats = 1 if level == 0.0 else config.noise_repeats
        for index in range(repeats):
            _log(config, f"winner stability, level {level}, run {index + 1}/{repeats}")
            seed = derive_seed(config.random_state, "stability", level, index)
            try:
                if level == 0.0:
                    jittered = X
                else:
                    jittered = inject_feature_noise(
                        X, level, X_train=X, random_state=seed
                    ).data
                fitted = _fit_search(search, jittered, y, seed)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                errors.append(
                    f"stability level {level} run {index}: {type(exc).__name__}: {exc}"
                )
                continue
            params = dict(fitted.best_params_)
            key = _params_key(params)
            records.append(
                {
                    "level": float(level),
                    "run": int(index),
                    "seed": int(seed),
                    "best_score": float(fitted.best_score_),
                    "best_params": params,
                    "params_key": key,
                    "matches_reported": bool(key == reported_key),
                }
            )
    return records
