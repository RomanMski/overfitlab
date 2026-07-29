"""A complete binary-classification audit with a self-contained report."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stressfold import AuditConfig, StressSuite, audit


X_array, y_array = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=6,
    class_sep=1.4,
    random_state=7,
)
X = pd.DataFrame(X_array, columns=[f"feature_{index}" for index in range(10)])
y = pd.Series(y_array, name="outcome")

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=2_000, random_state=7),
)
result = audit(
    model,
    X,
    y,
    config=AuditConfig(task="binary", repeats=20, random_state=7, store_variants=True),
    suite=StressSuite.standard(),
)

output = Path("stressfold-output")
result.write_html(output / "report.html")
result.write_json(output / "results.json")
result.export_variants(output / "variants")
print(result.generalization_summary().to_string(index=False))
