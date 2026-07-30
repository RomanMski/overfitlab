"""Audit a hyperparameter search instead of a single fitted model.

The point of this example is that both searches below look successful. One is
searching over targets that carry real signal and the other over targets that
are pure noise, and ``GridSearchCV`` reports a score above chance for both.
Only the audit separates them.

Run it with::

    python examples/python/search_optimism.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from stressfold import SearchAuditConfig, audit_search


def build_search() -> GridSearchCV:
    """A small but genuinely multi-parameter search, 24 candidates."""

    pipeline = make_pipeline(
        StandardScaler(), DecisionTreeClassifier(random_state=0)
    )
    grid = {
        "decisiontreeclassifier__max_depth": [1, 2, 3, 5, 8, None],
        "decisiontreeclassifier__min_samples_leaf": [1, 2, 5, 10],
    }
    return GridSearchCV(pipeline, grid, cv=4, scoring="roc_auc")


def signal_table(n: int = 300) -> tuple[pd.DataFrame, pd.Series]:
    values, target = make_classification(
        n_samples=n,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        class_sep=1.5,
        random_state=3,
    )
    columns = [f"feature_{index}" for index in range(values.shape[1])]
    return pd.DataFrame(values, columns=columns), pd.Series(target, name="outcome")


def noise_table(n: int = 300) -> tuple[pd.DataFrame, pd.Series]:
    """The same shape, but the target is unrelated to every feature."""

    rng = np.random.default_rng(0)
    columns = [f"feature_{index}" for index in range(8)]
    features = pd.DataFrame(rng.normal(size=(n, 8)), columns=columns)
    return features, pd.Series(rng.integers(0, 2, size=n), name="outcome")


def report(name: str, X: pd.DataFrame, y: pd.Series, output: Path) -> None:
    config = SearchAuditConfig(
        task="binary",
        random_state=7,
        outer_repeats=5,
        permutation_repeats=30,
        noise_levels=(0.0, 0.1, 0.25),
        noise_repeats=3,
        verbose=True,
    )
    result = audit_search(build_search(), X, y, config=config)

    print()
    print("#" * 68)
    print(f"# {name}")
    print("#" * 68)
    print(result.summary_text())
    print()

    null = result.permutation_summary()
    if null["null_max_best_score"] >= result.reported_score:
        print(
            "READ THIS: at least one search over shuffled targets matched or beat"
            f" the reported score of {result.reported_score:.4f}. A search of this"
            " size can reach that number with no signal present."
        )
    result.write_json(output)
    print(f"wrote {output}")


if __name__ == "__main__":
    output = Path("stressfold-output")
    report("Real signal", *signal_table(), output / "search-signal.json")
    report("Pure noise", *noise_table(), output / "search-noise.json")
