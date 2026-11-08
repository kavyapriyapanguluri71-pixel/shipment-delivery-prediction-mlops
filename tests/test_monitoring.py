import json
from pathlib import Path

import pandas as pd
import pytest

from mlops_pipeline.monitoring import (
    _distribution_index,
    _drift_status,
    _numeric_index,
    compare_batches,
)
from mlops_pipeline.schema import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_psi_is_zero_for_identical_distributions() -> None:
    values = pd.Series([1, 2, 2, 3, 4, 5])
    categories = pd.Series(["A", "B", "A", "C"])

    assert _numeric_index(values, values.copy()) == pytest.approx(0.0)
    assert _distribution_index(categories, categories.copy()) == pytest.approx(0.0)


def test_psi_detects_shift_and_status_thresholds() -> None:
    reference = pd.Series(range(100))
    shifted = pd.Series([1000] * 100)

    assert _numeric_index(reference, shifted) >= 0.2
    assert [_drift_status(value) for value in (0.099, 0.1, 0.1999, 0.2)] == [
        "low",
        "moderate",
        "moderate",
        "high",
    ]


def test_compare_batches_reports_and_saves_feature_drift(tmp_path) -> None:
    source = pd.read_csv(PROJECT_ROOT / "Train.csv")
    reference = source.sample(n=300, random_state=4).loc[:, list(FEATURE_COLUMNS)]
    current = reference.copy()
    current["Weight_in_gms"] = current["Weight_in_gms"].max() + 5000
    reference_path = tmp_path / "reference.csv"
    current_path = tmp_path / "current.csv"
    report_path = tmp_path / "reports" / "drift.json"
    reference.to_csv(reference_path, index=False)
    current.to_csv(current_path, index=False)

    report = compare_batches(reference_path, current_path, report_path)

    by_feature = {item["feature"]: item for item in report["features"]}
    assert report["reference_rows"] == report["current_rows"] == 300
    assert by_feature["Weight_in_gms"]["drift_status"] == "high"
    assert "Weight_in_gms" in report["high_drift_features"]
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_drift_monitor_rejects_missing_columns_and_nonfinite_values(tmp_path) -> None:
    source = pd.read_csv(PROJECT_ROOT / "Train.csv").head(5).loc[:, list(FEATURE_COLUMNS)]
    missing_path = tmp_path / "missing.csv"
    source.drop(columns=["Gender"]).to_csv(missing_path, index=False)
    valid_path = tmp_path / "valid.csv"
    source.to_csv(valid_path, index=False)
    invalid = source.copy()
    invalid["Weight_in_gms"] = invalid["Weight_in_gms"].astype(float)
    invalid.loc[0, "Weight_in_gms"] = float("inf")
    invalid_path = tmp_path / "invalid.csv"
    invalid.to_csv(invalid_path, index=False)

    with pytest.raises(ValueError, match="missing feature columns"):
        compare_batches(missing_path, valid_path)
    with pytest.raises(ValueError, match="infinite values"):
        compare_batches(valid_path, invalid_path)
