from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SOURCE = Path(__file__).resolve().parents[2] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(
        n_samples=180,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        class_sep=1.5,
        random_state=41,
    )
    return pd.DataFrame(
        X, columns=[f"x{index}" for index in range(X.shape[1])]
    ), pd.Series(y, name="outcome")


@pytest.fixture
def classification_model() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1_000)),
        ]
    )
