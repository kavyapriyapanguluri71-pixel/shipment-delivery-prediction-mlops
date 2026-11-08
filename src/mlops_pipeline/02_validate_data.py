"""Lab 2: Validate shipment dataset quality and generate a report."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mlops_pipeline.schema import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    ID_COLUMN,
)
from mlops_pipeline.validation import load_training_data


def main():
    data_path = Path("Train.csv")
    report_path = Path("reports/lab02_validation.json")

    print("=" * 60)
    print("LAB 2: SHIPMENT DATA VALIDATION")
    print("=" * 60)

    df = load_training_data(data_path)

    numeric_checks = {}
    for column in NUMERIC_FEATURES:
        values = df[column].to_numpy(dtype=float)
        numeric_checks[column] = bool(np.isfinite(values).all())

    report = {
        "lab": 2,
        "stage": "data_validation",
        "status": "PASSED",
        "dataset": data_path.name,
        "rows": int(len(df)),
        "required_columns_present": True,
        "missing_values": int(
            df[list(FEATURE_COLUMNS) + [TARGET_COLUMN]].isna().sum().sum()
        ),
        "duplicate_ids": int(df[ID_COLUMN].duplicated().sum()),
        "numeric_columns_finite": numeric_checks,
        "target_column": TARGET_COLUMN,
        "target_classes": sorted(
            int(value) for value in df[TARGET_COLUMN].unique()
        ),
        "target_distribution": {
            str(key): int(value)
            for key, value in df[TARGET_COLUMN].value_counts().items()
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    if not all(numeric_checks.values()):
        raise ValueError("Non-finite numeric values found.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Rows validated: {len(df)}")
    print(f"Required columns: {len(FEATURE_COLUMNS) + 2}")
    print(f"Duplicate IDs: {report['duplicate_ids']}")
    print(f"Missing values: {report['missing_values']}")
    print(f"Report saved: {report_path}")
    print("[SUCCESS] LAB 2 PASSED")


if __name__ == "__main__":
    main()
