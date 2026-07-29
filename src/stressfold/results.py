"""Audit result containers, summaries, and artifact export."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AuditConfig, StressSuite
from .metrics import METRICS, metric_degradation


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(slots=True)
class Variant:
    """A generated train or test dataset with full perturbation provenance."""

    repeat: int
    experiment: str
    level: float
    partition: str
    seed: int
    X: pd.DataFrame
    y: pd.Series
    row_indices: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.X) != len(self.y) or len(self.X) != len(self.row_indices):
            raise ValueError(
                "Variant X, y, and row_indices must have identical lengths"
            )

    def manifest_entry(self, filename: str | None = None) -> dict[str, Any]:
        entry = {
            "repeat": self.repeat,
            "experiment": self.experiment,
            "level": self.level,
            "partition": self.partition,
            "seed": self.seed,
            "rows": len(self.X),
            "columns": [str(column) for column in self.X.columns],
            "row_indices": self.row_indices,
            "metadata": self.metadata,
        }
        if filename is not None:
            entry["file"] = filename
        return _jsonable(entry)


@dataclass(slots=True)
class AuditResult:
    """Structured, inspectable output from :func:`stressfold.audit`."""

    config: AuditConfig
    suite: StressSuite
    records: list[dict[str, Any]]
    seeds: list[dict[str, Any]]
    variants: list[Variant] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    data_fingerprint: str = ""
    estimator: str = ""
    n_samples: int = 0
    n_features: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def records_frame(self) -> pd.DataFrame:
        """Return one row per metric evaluation and Monte Carlo replicate."""

        return pd.DataFrame.from_records(self.records).copy()

    def seeds_frame(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.seeds).copy()

    def summary_frame(self) -> pd.DataFrame:
        """Aggregate paired runs without pretending split scores are independent.

        ``mc_low`` and ``mc_high`` are empirical Monte Carlo quantiles. They are
        deliberately not labelled as confidence intervals.
        """

        frame = self.records_frame()
        columns = [
            "experiment",
            "evidence",
            "evaluation",
            "level",
            "metric",
            "n",
            "mean",
            "median",
            "std",
            "mc_low",
            "mc_high",
            "baseline_mean",
            "mean_raw_delta",
            "mean_degradation",
            "median_degradation",
        ]
        if frame.empty:
            return pd.DataFrame(columns=columns)
        alpha = (1.0 - self.config.interval) / 2.0
        grouped_rows: list[dict[str, Any]] = []
        keys = ["experiment", "evidence", "evaluation", "level", "metric"]
        for key, group in frame.groupby(keys, dropna=False, sort=True):
            values = pd.to_numeric(group["value"], errors="coerce").dropna()
            degradation = pd.to_numeric(group["degradation"], errors="coerce").dropna()
            raw_delta = pd.to_numeric(group["raw_delta"], errors="coerce").dropna()
            baseline = pd.to_numeric(group["baseline_value"], errors="coerce").dropna()
            row = dict(zip(keys, key, strict=True))
            row.update(
                {
                    "n": int(len(values)),
                    "mean": float(values.mean()) if len(values) else float("nan"),
                    "median": float(values.median()) if len(values) else float("nan"),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "mc_low": float(values.quantile(alpha))
                    if len(values)
                    else float("nan"),
                    "mc_high": float(values.quantile(1.0 - alpha))
                    if len(values)
                    else float("nan"),
                    "baseline_mean": float(baseline.mean())
                    if len(baseline)
                    else float("nan"),
                    "mean_raw_delta": float(raw_delta.mean())
                    if len(raw_delta)
                    else float("nan"),
                    "mean_degradation": float(degradation.mean())
                    if len(degradation)
                    else float("nan"),
                    "median_degradation": float(degradation.median())
                    if len(degradation)
                    else float("nan"),
                }
            )
            grouped_rows.append(row)
        return pd.DataFrame(grouped_rows, columns=columns)

    def generalization_frame(self) -> pd.DataFrame:
        """Return paired clean train-to-test gaps; positive always means worse."""

        frame = self.records_frame()
        if frame.empty:
            return pd.DataFrame(columns=["repeat", "metric", "train", "test", "gap"])
        clean = frame[
            (frame["experiment"] == "baseline")
            & (frame["evaluation"].isin(["train", "test"]))
        ]
        pivot = clean.pivot_table(
            index=["repeat", "metric"],
            columns="evaluation",
            values="value",
            aggfunc="first",
        ).reset_index()
        if "train" not in pivot or "test" not in pivot:
            return pd.DataFrame(columns=["repeat", "metric", "train", "test", "gap"])
        pivot["gap"] = [
            metric_degradation(metric, test, train)
            for metric, test, train in zip(
                pivot["metric"], pivot["test"], pivot["train"], strict=True
            )
        ]
        return pivot[["repeat", "metric", "train", "test", "gap"]]

    def generalization_summary(self) -> pd.DataFrame:
        frame = self.generalization_frame()
        if frame.empty:
            return pd.DataFrame(
                columns=["metric", "n", "mean_gap", "median_gap", "mc_low", "mc_high"]
            )
        alpha = (1.0 - self.config.interval) / 2.0
        rows = []
        for metric, group in frame.groupby("metric", sort=True):
            gaps = group["gap"].dropna()
            rows.append(
                {
                    "metric": metric,
                    "n": len(gaps),
                    "mean_gap": gaps.mean(),
                    "median_gap": gaps.median(),
                    "mc_low": gaps.quantile(alpha),
                    "mc_high": gaps.quantile(1.0 - alpha),
                }
            )
        return pd.DataFrame(rows)

    def permutation_summary(self) -> pd.DataFrame:
        """Summarize paired null fits without treating overlapping splits as independent.

        ``null_mean`` and ``observed_mean`` weight each valid holdout once. The
        explicitly named pooled exceedance rate remains fit-weighted and reports
        its per-holdout fit-count range alongside it.
        """

        frame = self.records_frame()
        null = frame[
            (frame["experiment"] == "permutation_null")
            & (frame["evaluation"] == "test")
        ]
        rows = []
        for metric, group in null.groupby("metric", sort=True):
            valid = group[
                ["repeat", "scenario_repeat", "value", "baseline_value"]
            ].dropna()
            definition = METRICS[metric]
            if definition.higher_is_better:
                exceed = valid["value"] >= valid["baseline_value"]
            else:
                exceed = valid["value"] <= valid["baseline_value"]
            count = int(exceed.sum())
            n = int(len(valid))
            fits_per_repeat = valid.groupby("repeat").size()
            null_by_repeat = valid.groupby("repeat")["value"].mean()
            observed_by_repeat = valid.groupby("repeat")["baseline_value"].first()
            rows.append(
                {
                    "metric": metric,
                    "n": n,
                    "repeat_count": int(valid["repeat"].nunique()),
                    "null_fits_per_repeat_min": int(fits_per_repeat.min())
                    if len(fits_per_repeat)
                    else 0,
                    "null_fits_per_repeat_max": int(fits_per_repeat.max())
                    if len(fits_per_repeat)
                    else 0,
                    "null_mean": float(null_by_repeat.mean())
                    if len(null_by_repeat)
                    else float("nan"),
                    "observed_mean": float(observed_by_repeat.mean())
                    if len(observed_by_repeat)
                    else float("nan"),
                    "paired_exceedances": count,
                    "pooled_paired_exceedance_rate": count / n
                    if n
                    else float("nan"),
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "metric",
                "n",
                "repeat_count",
                "null_fits_per_repeat_min",
                "null_fits_per_repeat_max",
                "null_mean",
                "observed_mean",
                "paired_exceedances",
                "pooled_paired_exceedance_rate",
            ],
        )

    def to_dict(self, *, include_records: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "1.1",
            "created_at": self.created_at,
            "package_version": "0.1.0",
            "scope": "i.i.d. tabular binary classification and regression",
            "interpretation": (
                "StressFold estimates generalization gaps, refit stability, and sensitivity under the stated "
                "protocol. Passing these tests does not prove absence of overfitting or robustness to arbitrary shift."
            ),
            "config": self.config.to_dict(),
            "suite": self.suite.to_dict(),
            "data": {
                "fingerprint": self.data_fingerprint,
                "n_samples": self.n_samples,
                "n_features": self.n_features,
            },
            "estimator": self.estimator,
            "seeds": self.seeds,
            "errors": self.errors,
            "summary": self.summary_frame().to_dict(orient="records"),
            "generalization_summary": self.generalization_summary().to_dict(
                orient="records"
            ),
            "permutation_summary": self.permutation_summary().to_dict(orient="records"),
            "variant_manifest": [variant.manifest_entry() for variant in self.variants],
        }
        if include_records:
            payload["records"] = self.records
        return _jsonable(payload)

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        return destination

    def write_html(self, path: str | Path) -> Path:
        from .reporting import render_html

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(render_html(self), encoding="utf-8")
        return destination

    def export_variants(self, directory: str | Path) -> Path:
        """Write captured scenario datasets as CSV plus a provenance manifest."""

        if not self.variants:
            raise ValueError(
                "No variants were retained. Set AuditConfig(store_variants=True) before running audit()."
            )
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        manifest_entries: list[dict[str, Any]] = []
        for variant in self.variants:
            level = f"{variant.level:.8g}".replace("-", "m").replace(".", "p")
            safe_experiment = re.sub(r"[^a-zA-Z0-9_-]+", "_", variant.experiment)
            filename = f"r{variant.repeat:03d}_{safe_experiment}_l{level}_{variant.partition}_s{variant.seed}.csv"
            frame = variant.X.copy()
            row_column = "__row_index__"
            while row_column in frame.columns:
                row_column = "_" + row_column
            target_name = str(variant.y.name or "target")
            if target_name in frame.columns:
                target_name = "__stressfold_target__"
            while target_name in frame.columns or target_name == row_column:
                target_name = "_" + target_name
            frame.insert(0, row_column, variant.row_indices)
            frame[target_name] = np.asarray(variant.y)
            frame.to_csv(root / filename, index=False)
            entry = variant.manifest_entry(filename)
            entry["row_index_column"] = row_column
            entry["target_column"] = target_name
            manifest_entries.append(entry)
        manifest = {
            "schema_version": "1.0",
            "source_fingerprint": self.data_fingerprint,
            "variants": manifest_entries,
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8"
        )
        return manifest_path
