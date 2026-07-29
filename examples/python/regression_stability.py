"""Trace regression error as measurement noise and sample size change."""

import pandas as pd
from sklearn.datasets import make_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stressfold import AuditConfig, StressSuite, audit


X_array, y_array = make_regression(
    n_samples=700,
    n_features=12,
    n_informative=8,
    noise=18.0,
    random_state=19,
)
X = pd.DataFrame(X_array, columns=[f"x{index}" for index in range(X_array.shape[1])])
y = pd.Series(y_array, name="response")
model = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=3.0))

result = audit(
    model,
    X,
    y,
    config=AuditConfig(
        task="regression", metrics=("rmse", "mae", "r2"), repeats=15, random_state=19
    ),
    suite=StressSuite(
        feature_noise=(0.0, 0.1, 0.25, 0.5, 1.0),
        label_noise=(0.0, 0.1, 0.25),
        missingness=(0.0, 0.05, 0.1),
        train_fraction=(0.2, 0.4, 0.7, 1.0),
        permutation_repeats=20,
    ),
)
print(result.summary_frame().to_string(index=False))
