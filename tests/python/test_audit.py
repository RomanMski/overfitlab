from __future__ import annotations

import json

import pandas as pd
import pandas.testing as pdt
import pytest
from sklearn.linear_model import LogisticRegression

from stressfold import AuditConfig, StressSuite, audit


def _small_suite() -> StressSuite:
    return StressSuite(
        feature_noise=(0.0, 0.4),
        label_noise=(0.0, 0.2),
        missingness=(0.0, 0.1),
        train_fraction=(0.5, 1.0),
        permutation_repeats=2,
    )


def test_audit_is_deterministic_and_records_named_seeds(
    classification_data, classification_model
) -> None:
    X, y = classification_data
    config = AuditConfig(task="binary", repeats=2, random_state=93)

    first = audit(classification_model, X, y, config=config, suite=_small_suite())
    second = audit(classification_model, X, y, config=config, suite=_small_suite())

    pdt.assert_frame_equal(first.records_frame(), second.records_frame())
    pdt.assert_frame_equal(first.seeds_frame(), second.seeds_frame())
    assert first.data_fingerprint == second.data_fingerprint
    assert {
        "split",
        "fit",
        "perturb_test",
        "perturb_train_labels",
        "subsample_train",
        "permute_train_labels",
    } <= set(first.seeds_frame()["operation"])
    assert first.errors == []


def test_zero_severity_and_full_fraction_match_paired_baseline(
    classification_data, classification_model
) -> None:
    X, y = classification_data
    result = audit(
        classification_model,
        X,
        y,
        config=AuditConfig(task="binary", repeats=2, random_state=4),
        suite=_small_suite(),
    )
    frame = result.records_frame()
    baseline = frame[
        (frame["experiment"] == "baseline") & (frame["evaluation"] == "test")
    ][["repeat", "metric", "value"]].rename(columns={"value": "baseline"})

    for experiment, level in (
        ("feature_noise", 0.0),
        ("missingness", 0.0),
        ("label_noise", 0.0),
        ("train_fraction", 1.0),
    ):
        scenario = frame[
            (frame["experiment"] == experiment)
            & (frame["evaluation"] == "test")
            & (frame["level"] == level)
        ][["repeat", "metric", "value"]]
        paired = scenario.merge(
            baseline, on=["repeat", "metric"], validate="one_to_one"
        )
        assert len(paired) == len(baseline)
        assert (paired["value"] == paired["baseline"]).all(), experiment


def test_result_exposes_paired_summaries_and_generalization_gap(
    classification_data, classification_model
) -> None:
    X, y = classification_data
    result = audit(
        classification_model,
        X,
        y,
        config=AuditConfig(
            task="binary", metrics=("roc_auc", "log_loss"), repeats=3, interval=0.8
        ),
        suite=_small_suite(),
    )

    summary = result.summary_frame()
    assert {"mc_low", "mc_high", "mean_degradation", "baseline_mean"} <= set(
        summary.columns
    )
    assert (summary["mc_low"] <= summary["mc_high"]).all()
    gaps = result.generalization_frame()
    assert len(gaps) == 3 * 2
    assert gaps["gap"].notna().all()
    permutation = result.permutation_summary()
    assert set(permutation["metric"]) == {"roc_auc", "log_loss"}
    assert permutation["plus_one_p"].between(0, 1).all()


def test_variant_export_contains_source_rows_and_manifest(
    tmp_path, classification_data, classification_model
) -> None:
    X, y = classification_data
    result = audit(
        classification_model,
        X,
        y,
        config=AuditConfig(
            task="binary", metrics=("accuracy",), repeats=1, store_variants=True
        ),
        suite=StressSuite(
            feature_noise=(0.2,),
            label_noise=(0.1,),
            missingness=(0.1,),
            train_fraction=(0.5,),
            permutation_repeats=1,
        ),
    )
    manifest_path = result.export_variants(tmp_path / "variants")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["source_fingerprint"] == result.data_fingerprint
    assert len(payload["variants"]) == len(result.variants)
    first = payload["variants"][0]
    exported = pd.read_csv(manifest_path.parent / first["file"])
    assert "__row_index__" in exported
    assert len(exported) == first["rows"]


def test_export_requires_explicit_variant_retention(
    classification_data, classification_model, tmp_path
) -> None:
    X, y = classification_data
    result = audit(
        classification_model,
        X,
        y,
        config=AuditConfig(task="binary", metrics=("accuracy",), repeats=1),
        suite=StressSuite(
            feature_noise=(0,),
            label_noise=(0,),
            missingness=(0,),
            train_fraction=(1,),
            permutation_repeats=0,
        ),
    )
    with pytest.raises(ValueError, match="store_variants=True"):
        result.export_variants(tmp_path)


def test_unsupported_missingness_is_reported_without_losing_other_results(
    classification_data,
) -> None:
    X, y = classification_data
    result = audit(
        LogisticRegression(max_iter=1_000),
        X,
        y,
        config=AuditConfig(task="binary", metrics=("accuracy",), repeats=1),
        suite=StressSuite(
            feature_noise=(0,),
            label_noise=(0,),
            missingness=(0, 0.2),
            train_fraction=(1,),
            permutation_repeats=0,
        ),
    )

    assert any(
        error["experiment"] == "missingness" and error["level"] == 0.2
        for error in result.errors
    )
    assert "baseline" in set(result.records_frame()["experiment"])


def test_misaligned_pandas_indices_fail_instead_of_silent_reordering(
    classification_data, classification_model
) -> None:
    X, y = classification_data
    y.index = y.index[::-1]
    with pytest.raises(ValueError, match="indices differ"):
        audit(
            classification_model,
            X,
            y,
            config=AuditConfig(task="binary", repeats=1),
            suite=StressSuite.quick(permutation_repeats=0),
        )


def test_multicolumn_target_array_is_rejected(
    classification_data, classification_model
) -> None:
    X, y = classification_data
    invalid = pd.concat([y, y], axis=1).to_numpy()
    with pytest.raises(ValueError, match="single-column"):
        audit(
            classification_model,
            X,
            invalid,
            config=AuditConfig(task="binary", repeats=1),
            suite=StressSuite.quick(permutation_repeats=0),
        )
