from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stressfold import AuditConfig, StressSuite, audit


def _minimal_suite(**overrides) -> StressSuite:
    values = {
        "feature_noise": (0.0,),
        "label_noise": (0.0,),
        "missingness": (0.0,),
        "train_fraction": (1.0,),
        "permutation_repeats": 0,
    }
    values.update(overrides)
    return StressSuite(**values)


def test_half_randomized_training_labels_degrade_clear_signal() -> None:
    X, y = make_classification(
        n_samples=360,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        class_sep=2.5,
        flip_y=0,
        random_state=8,
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1_000))
    result = audit(
        model,
        pd.DataFrame(X),
        pd.Series(y),
        config=AuditConfig(
            task="binary", metrics=("roc_auc",), repeats=4, random_state=12
        ),
        suite=_minimal_suite(label_noise=(0.0, 0.5)),
    )
    summary = result.summary_frame()
    stressed = summary[
        (summary["experiment"] == "label_noise")
        & (summary["evaluation"] == "test")
        & (summary["level"] == 0.5)
        & (summary["metric"] == "roc_auc")
    ].iloc[0]

    assert stressed["mean_degradation"] > 0.15


def test_permutation_null_is_near_chance_and_below_observed_signal() -> None:
    X, y = make_classification(
        n_samples=400,
        n_features=7,
        n_informative=6,
        n_redundant=0,
        class_sep=2.8,
        flip_y=0,
        random_state=23,
    )
    result = audit(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1_000)),
        pd.DataFrame(X),
        pd.Series(y),
        config=AuditConfig(
            task="binary", metrics=("roc_auc",), repeats=3, random_state=5
        ),
        suite=_minimal_suite(permutation_repeats=8),
    )
    row = result.permutation_summary().iloc[0]

    assert 0.35 < row["null_mean"] < 0.65
    assert row["observed_mean"] > 0.9
    assert row["pooled_paired_exceedance_rate"] <= 0.08
    assert row["repeat_count"] == 3
    assert row["null_fits_per_repeat_min"] == 8
    assert row["null_fits_per_repeat_max"] == 8


def test_large_heldout_feature_noise_increases_regression_error() -> None:
    X, y = make_regression(n_samples=320, n_features=6, noise=4.0, random_state=31)
    model = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=1.0))
    result = audit(
        model,
        pd.DataFrame(X),
        pd.Series(y),
        config=AuditConfig(
            task="regression", metrics=("rmse",), repeats=4, random_state=17
        ),
        suite=_minimal_suite(feature_noise=(0.0, 2.0)),
    )
    summary = result.summary_frame()
    stressed = summary[
        (summary["experiment"] == "feature_noise")
        & (summary["level"] == 2.0)
        & (summary["metric"] == "rmse")
    ].iloc[0]

    assert stressed["median_degradation"] > 50.0
