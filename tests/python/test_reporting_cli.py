from __future__ import annotations

import json

from stressfold import AuditConfig, StressSuite, audit
from stressfold.cli import main


def test_json_and_html_reports_are_self_contained(
    tmp_path, classification_data, classification_model
) -> None:
    X, y = classification_data
    result = audit(
        classification_model,
        X,
        y,
        config=AuditConfig(task="binary", metrics=("roc_auc",), repeats=1),
        suite=StressSuite(
            feature_noise=(0, 0.3),
            label_noise=(0,),
            missingness=(0,),
            train_fraction=(1,),
            permutation_repeats=1,
        ),
    )
    html_path = result.write_html(tmp_path / "report.html")
    json_path = result.write_json(tmp_path / "results.json")
    html = html_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert "StressFold" in html
    assert "Monte Carlo variability intervals" in html
    assert "<svg" in html
    assert "https://" not in html
    assert payload["schema_version"] == "1.0"
    assert payload["data"]["fingerprint"] == result.data_fingerprint
    assert payload["interpretation"].startswith("StressFold estimates")


def test_cli_runs_local_csv_audit(tmp_path, classification_data) -> None:
    X, y = classification_data
    frame = X.copy()
    frame["segment"] = frame["x0"].gt(0).map({True: "north", False: "south"})
    frame["outcome"] = y
    source = tmp_path / "sample.csv"
    output = tmp_path / "audit"
    frame.to_csv(source, index=False)

    status = main(
        [
            str(source),
            "--target",
            "outcome",
            "--task",
            "binary",
            "--metrics",
            "accuracy",
            "--quick",
            "--repeats",
            "1",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert (output / "report.html").is_file()
    assert (output / "results.json").is_file()
