"""Behaviour tests for the selection-aware search audit."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from stressfold import SearchAuditConfig, audit_search


def _tree_search(**kwargs):
    pipeline = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=0))
    grid = {
        "decisiontreeclassifier__max_depth": [1, 3, None],
        "decisiontreeclassifier__min_samples_leaf": [1, 5],
    }
    defaults = {"cv": 3, "scoring": "roc_auc"}
    defaults.update(kwargs)
    return GridSearchCV(pipeline, grid, **defaults)


def _noise_table(n=140, features=5, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, features)), columns=[f"f{i}" for i in range(features)]
    )
    y = pd.Series(rng.integers(0, 2, size=n), name="target")
    return X, y


def _signal_table(n=140, features=5, seed=3):
    values, target = make_classification(
        n_samples=n,
        n_features=features,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        class_sep=2.0,
        random_state=seed,
    )
    X = pd.DataFrame(values, columns=[f"f{i}" for i in range(features)])
    return X, pd.Series(target, name="target")


def _quick(**overrides):
    values = {
        "random_state": 5,
        "outer_repeats": 3,
        "permutation_repeats": 6,
        "noise_levels": (0.0, 0.3),
        "noise_repeats": 2,
    }
    values.update(overrides)
    return SearchAuditConfig(task="binary", **values)


def test_pure_noise_search_is_not_significant_and_looks_optimistic():
    X, y = _noise_table()
    result = audit_search(_tree_search(), X, y, config=_quick())

    null = result.permutation_summary()
    optimism = result.optimism_summary()

    # A grid search on unrelated targets still reports a score above chance.
    assert result.reported_score > 0.5
    # The permutation null must refuse to call it signal.
    assert null["p_value"] > 0.1
    # A search of this size reaches a similar score with no signal at all.
    assert null["null_mean_best_score"] > 0.5
    # The reported score should not survive stepping outside the search.
    assert optimism["mean_optimism"] > 0.0
    assert not result.errors


def test_real_signal_reaches_the_permutation_floor():
    X, y = _signal_table()
    config = _quick()
    result = audit_search(_tree_search(), X, y, config=config)

    null = result.permutation_summary()
    floor = 1.0 / (null["n_permutations"] + 1.0)

    assert result.reported_score > 0.8
    assert null["exceedances"] == 0
    # With M permutations the smallest attainable p-value is 1 / (M + 1).
    assert null["p_value"] == pytest.approx(floor)
    assert null["null_mean_best_score"] < result.reported_score


def test_permutation_null_reruns_the_whole_search_not_one_configuration():
    """Each permutation must be free to pick a different winning config."""

    X, y = _noise_table(seed=11)
    result = audit_search(
        _tree_search(), X, y, config=_quick(permutation_repeats=10)
    )
    frame = result.permutation_frame()

    assert len(frame) == 10
    assert "best_params" in frame.columns
    # If selection were held fixed every permutation would report one config.
    distinct = {json.dumps(params, sort_keys=True) for params in frame["best_params"]}
    assert len(distinct) > 1


def test_audit_is_deterministic_for_a_fixed_seed():
    X, y = _noise_table(seed=2)
    first = audit_search(_tree_search(), X, y, config=_quick())
    second = audit_search(_tree_search(), X, y, config=_quick())

    assert first.reported_score == second.reported_score
    assert first.optimism_summary() == second.optimism_summary()
    assert first.permutation_summary() == second.permutation_summary()
    pd.testing.assert_frame_equal(first.stability_summary(), second.stability_summary())


def test_stability_reports_retention_and_counts_candidates():
    X, y = _signal_table(seed=4)
    result = audit_search(_tree_search(), X, y, config=_quick())
    stability = result.stability_summary()

    assert result.n_candidates == 6
    # The clean level is run once and must agree with the reported winner.
    clean = stability.loc[stability["level"] == 0.0].iloc[0]
    assert clean["n_runs"] == 1
    assert clean["winner_retention_rate"] == 1.0
    # The perturbed level is run noise_repeats times.
    jittered = stability.loc[stability["level"] == 0.3].iloc[0]
    assert jittered["n_runs"] == 2
    assert 0.0 <= jittered["winner_retention_rate"] <= 1.0


def _unseeded_forest_search():
    """A search whose estimator carries random_state=None.

    The tree search used elsewhere hardcodes random_state=0, which hides any
    confusion between data sensitivity and the estimator's own randomness.
    """

    return GridSearchCV(
        RandomForestClassifier(n_estimators=25),
        {"max_depth": [2, 3, None], "min_samples_leaf": [1, 4]},
        cv=3,
        scoring="roc_auc",
    )


def test_zero_noise_stability_is_not_disturbed_by_the_selection_seed():
    """Regression test for a real defect.

    The clean stability run used to be refitted under a freshly derived seed,
    so an estimator with random_state=None could report a different winner at
    zero noise even though the data was byte-identical. Every stability run now
    reuses one selection seed, leaving the data as the only thing that varies.
    """

    X, y = _noise_table(seed=21)
    result = audit_search(
        _unseeded_forest_search(),
        X,
        y,
        config=_quick(permutation_repeats=0, noise_levels=(0.0, 0.3)),
    )
    stability = result.stability_summary()
    clean = stability.loc[stability["level"] == 0.0].iloc[0]

    assert clean["winner_retention_rate"] == 1.0
    assert clean["distinct_winners"] == 1
    # Every perturbed run must share the clean run's selection seed.
    frame = result.stability_frame()
    assert frame["seed"].nunique() == 1


def test_selection_randomness_is_reported_separately_from_noise():
    """Seed-driven movement must not be charged to the noise operator."""

    X, y = _noise_table(seed=22)
    result = audit_search(
        _unseeded_forest_search(),
        X,
        y,
        config=_quick(
            permutation_repeats=0, noise_levels=(0.0,), selection_seed_repeats=4
        ),
    )
    reseed = result.selection_noise_summary()

    assert reseed["n_reseeds"] == 4
    # An unseeded forest search moves on identical data, and that belongs here
    # rather than in the stability curve.
    assert reseed["distinct_winners"] > 1
    assert reseed["winner_retention_rate"] < 1.0
    assert result.stability_summary().iloc[0]["winner_retention_rate"] == 1.0


def test_selection_randomness_can_be_switched_off():
    X, y = _signal_table(seed=23)
    result = audit_search(
        _tree_search(),
        X,
        y,
        config=_quick(permutation_repeats=0, selection_seed_repeats=0),
    )
    assert result.selection_noise_summary() == {}
    assert result.selection_noise_frame().empty


def test_regression_task_is_supported():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    y = pd.Series(X["a"] * 2.0 + rng.normal(scale=0.3, size=120), name="target")
    search = GridSearchCV(
        make_pipeline(StandardScaler(), Ridge()),
        {"ridge__alpha": [0.01, 1.0, 100.0]},
        cv=3,
        scoring="r2",
    )
    config = SearchAuditConfig(
        task="regression",
        random_state=0,
        outer_repeats=2,
        permutation_repeats=4,
        noise_levels=(0.0,),
    )
    result = audit_search(search, X, y, config=config)

    assert result.metric == "r2"
    assert result.reported_score > 0.8
    assert result.permutation_summary()["null_mean_best_score"] < 0.5
    assert not result.errors


def test_json_artifact_is_serialisable_and_complete(tmp_path):
    X, y = _signal_table(seed=6)
    result = audit_search(_tree_search(), X, y, config=_quick())
    destination = result.write_json(tmp_path / "search.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["kind"] == "stressfold_search_audit"
    assert payload["search"]["n_candidates"] == 6
    assert payload["selection_optimism"]["n_outer_splits"] == 3
    assert payload["permutation_null"]["n_permutations"] == 6
    assert payload["winner_stability"]
    assert payload["selection_randomness"]["n_reseeds"] >= 1
    assert payload["config"]["task"] == "binary_classification"
    assert payload["seeds"]["root"] == 5


def test_summary_text_names_all_three_measurements():
    X, y = _signal_table(seed=8)
    text = audit_search(_tree_search(), X, y, config=_quick()).summary_text()

    assert "Selection optimism" in text
    assert "permutation null" in text.lower()
    assert "Winner stability" in text


def test_rejects_objects_that_are_not_searches():
    X, y = _noise_table()
    with pytest.raises(TypeError, match="GridSearchCV"):
        audit_search(object(), X, y, config=_quick())


def test_rejects_a_plain_estimator_that_performs_no_selection():
    """A bare estimator selects nothing, so it points the caller at audit()."""

    X, y = _noise_table()
    estimator = DecisionTreeClassifier(random_state=0)
    with pytest.raises(TypeError, match=r"no selection"):
        audit_search(estimator, X, y, config=_quick())


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"outer_repeats": 1}, "outer_repeats"),
        ({"test_size": 0.0}, "test_size"),
        ({"permutation_repeats": -1}, "permutation_repeats"),
        ({"noise_repeats": 0}, "noise_repeats"),
        ({"noise_levels": (-0.5,)}, "noise_levels"),
        ({"random_state": -1}, "random_state"),
    ],
)
def test_config_validation(overrides, message):
    values = {"task": "binary"}
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        SearchAuditConfig(**values)


def test_noise_levels_always_include_the_clean_level():
    config = SearchAuditConfig(task="binary", noise_levels=(0.4, 0.2))
    assert config.noise_levels == (0.0, 0.2, 0.4)
