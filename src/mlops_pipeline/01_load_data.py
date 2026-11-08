"""Lab 1: Load shipment dataset and generate a data loading report."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from mlops_pipeline.schema import ID_COLUMN, TARGET_COLUMN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Train.csv")
    parser.add_argument("--report", default="reports/lab01_data_loading.json")
    args = parser.parse_args()

    data_path = Path(args.data)
    report_path = Path(args.report)

    print("=" * 60)
    print("LAB 1: SHIPMENT DATA LOADING")
    print("=" * 60)

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    if df.empty:
        raise ValueError("Shipment dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")

    file_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    target_distribution = {
        str(key): int(value)
        for key, value in df[TARGET_COLUMN].value_counts().items()
    }

    report = {
        "lab": 1,
        "stage": "data_loading",
        "status": "PASSED",
        "dataset": data_path.name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "target_column": TARGET_COLUMN,
        "target_distribution": target_distribution,
        "duplicate_ids": (
            int(df[ID_COLUMN].duplicated().sum())
            if ID_COLUMN in df.columns else None
        ),
        "missing_values_total": int(df.isna().sum().sum()),
        "file_size_bytes": data_path.stat().st_size,
        "sha256": file_hash,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Dataset loaded: {data_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution: {target_distribution}")
    print(f"SHA-256: {file_hash}")
    print(f"Report saved: {report_path}")
    print("[SUCCESS] LAB 1 PASSED")


if __name__ == "__main__":
    main()
