from __future__ import annotations

import pytest

from stressfold import AuditConfig, StressSuite
from stressfold.random import derive_seed


def test_config_normalizes_task_and_metrics() -> None:
    config = AuditConfig(task="binary", metrics=("ROC_AUC", "log_loss", "roc_auc"))

    assert config.task == "binary_classification"
    assert config.metrics == ("roc_auc", "log_loss")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"task": "multiclass"}, "Unsupported task"),
        ({"task": "regression", "repeats": 0}, "repeats"),
        ({"task": "regression", "test_size": 1.0}, "test_size"),
        ({"task": "regression", "interval": 0.0}, "interval"),
    ],
)
def test_invalid_config_fails_early(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AuditConfig(**kwargs)


def test_suite_includes_comparison_anchors_and_validates_ranges() -> None:
    suite = StressSuite(
        feature_noise=(0.3,),
        label_noise=(0.2,),
        missingness=(0.1,),
        train_fraction=(0.5,),
        permutation_repeats=0,
    )

    assert suite.feature_noise == (0.0, 0.3)
    assert suite.label_noise == (0.0, 0.2)
    assert suite.missingness == (0.0, 0.1)
    assert suite.train_fraction == (0.5, 1.0)
    with pytest.raises(ValueError, match="missingness"):
        StressSuite(missingness=(1.1,))
    with pytest.raises(ValueError, match="finite"):
        StressSuite(feature_noise=(float("inf"),))


def test_named_seeds_are_stable_and_path_specific() -> None:
    first = derive_seed(17, "feature_noise", 2, 0.25)

    assert first == derive_seed(17, "feature_noise", 2, 0.25)
    assert first != derive_seed(17, "feature_noise", 2, 0.5)
    assert first != derive_seed(18, "feature_noise", 2, 0.25)
