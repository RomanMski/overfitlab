from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from stressfold.stressors import (
    inject_feature_noise,
    inject_label_noise,
    inject_missingness,
    robust_scales,
    subsample_training,
)


def test_feature_noise_uses_training_scale_and_preserves_inputs() -> None:
    train = pd.DataFrame({"value": [-1.0, 0.0, 1.0, 2.0], "kind": ["a", "a", "b", "b"]})
    heldout = pd.DataFrame({"value": [1_000.0, 2_000.0], "kind": ["a", "b"]})
    untouched = heldout.copy(deep=True)

    first = inject_feature_noise(heldout, 0.5, X_train=train, random_state=9)
    second = inject_feature_noise(heldout, 0.5, X_train=train, random_state=9)

    pdt.assert_frame_equal(heldout, untouched)
    pdt.assert_frame_equal(first.data, second.data)
    assert first.data["kind"].equals(heldout["kind"])
    expected_scale = robust_scales(train)["value"]
    assert first.metadata["column_details"][0]["scale"] == expected_scale
    assert first.metadata["scale_source"] == "training_fold"


def test_zero_level_operations_are_value_identity_copies() -> None:
    X = pd.DataFrame({"n": [1, 2, 3], "c": ["x", "y", "z"]})
    y = pd.Series([0, 1, 0], name="label")

    noisy_X = inject_feature_noise(X, 0.0, X_train=X, random_state=2).data
    noisy_y = inject_label_noise(y, 0.0, task="binary", random_state=2).data
    missing_X = inject_missingness(X, 0.0, random_state=2).data

    pdt.assert_frame_equal(noisy_X, X)
    pdt.assert_series_equal(noisy_y, y)
    pdt.assert_frame_equal(missing_X, X)
    assert noisy_X is not X
    assert noisy_y is not y


def test_binary_label_noise_flips_exact_rounded_count() -> None:
    y = pd.Series(["no", "yes"] * 10)
    result = inject_label_noise(y, 0.25, task="binary", random_state=4)

    assert int((result.data != y).sum()) == 5
    assert result.metadata["changed_rows"] == 5
    assert len(result.metadata["selected_positions"]) == 5


def test_missingness_counts_only_observed_cells_and_skips_null_column() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "empty": [np.nan] * 4})
    result = inject_missingness(X, 0.5, random_state=11)

    assert result.metadata["eligible_cells"] == 3
    assert result.metadata["injected_missing_cells"] == 2
    assert result.metadata["skipped_all_null_columns"] == ["empty"]
    assert int(result.data["a"].isna().sum()) == 3


def test_binary_subsample_is_paired_and_keeps_both_classes() -> None:
    X = pd.DataFrame({"row": np.arange(20)}, index=np.arange(100, 120))
    y = pd.Series([0] * 15 + [1] * 5, index=X.index, name="target")
    result = subsample_training(X, y, 0.25, task="binary", random_state=7)
    X_sub, y_sub = result.data

    assert len(X_sub) == 5
    assert X_sub.index.equals(y_sub.index)
    assert set(y_sub) == {0, 1}
    assert X_sub["row"].tolist() == [index - 100 for index in X_sub.index]
    assert result.metadata["stratified"] is True
