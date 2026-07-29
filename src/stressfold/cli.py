"""Command-line interface for local CSV audits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .config import AuditConfig, StressSuite
from .engine import audit


def _estimator(task: str, model: str, seed: int) -> Pipeline:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, make_column_selector(dtype_include="number")),
            ("categorical", categorical, make_column_selector(dtype_exclude="number")),
        ],
        remainder="drop",
    )
    if task == "binary_classification":
        final = (
            DecisionTreeClassifier(random_state=seed, min_samples_leaf=2)
            if model == "tree"
            else LogisticRegression(max_iter=2_000, random_state=seed)
        )
    else:
        final = (
            DecisionTreeRegressor(random_state=seed, min_samples_leaf=2)
            if model == "tree"
            else Ridge()
        )
    return Pipeline([("preprocess", preprocessing), ("model", final)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stressfold",
        description="Run reproducible generalization stress tests on a local CSV file.",
    )
    parser.add_argument("csv", type=Path, help="Input CSV file")
    parser.add_argument("--target", required=True, help="Target column")
    parser.add_argument("--task", required=True, choices=("binary", "regression"))
    parser.add_argument("--model", choices=("linear", "tree"), default="linear")
    parser.add_argument("--output", type=Path, default=Path("stressfold-audit"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics", nargs="+", default=None)
    parser.add_argument(
        "--quick", action="store_true", help="Use fewer stress levels and null fits"
    )
    parser.add_argument(
        "--export-variants",
        action="store_true",
        help="Retain and export generated datasets",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = pd.read_csv(args.csv)
    if args.target not in frame.columns:
        raise SystemExit(f"Target column {args.target!r} was not found in {args.csv}")
    y = frame.pop(args.target)
    config = AuditConfig(
        task=args.task,
        metrics=tuple(args.metrics) if args.metrics else None,
        repeats=args.repeats,
        test_size=args.test_size,
        random_state=args.seed,
        store_variants=args.export_variants,
    )
    suite = StressSuite.quick() if args.quick else StressSuite.standard()
    result = audit(
        _estimator(config.task, args.model, args.seed),
        frame,
        y,
        config=config,
        suite=suite,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report = result.write_html(args.output / "report.html")
    payload = result.write_json(args.output / "results.json")
    if args.export_variants:
        result.export_variants(args.output / "variants")
    print(f"StressFold audit complete: {report}")
    print(f"Machine-readable results: {payload}")
    if result.errors:
        print(
            f"Warning: {len(result.errors)} scenario runs failed; see results.json for details."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
