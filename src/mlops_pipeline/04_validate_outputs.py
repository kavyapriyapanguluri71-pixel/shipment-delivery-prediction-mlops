"""Lab 4: Validate preprocessed shipment outputs."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main():
    print("=" * 60)
    print("LAB 4: PREPROCESSING OUTPUT VALIDATION")
    print("=" * 60)

    output_dir = Path("outputs/lab03")

    files = {
        "X_train": output_dir / "X_train_processed.csv",
        "X_test": output_dir / "X_test_processed.csv",
        "y_train": output_dir / "y_train.csv",
        "y_test": output_dir / "y_test.csv",
    }

    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing output: {path}")

    X_train = pd.read_csv(files["X_train"])
    X_test = pd.read_csv(files["X_test"])
    y_train = pd.read_csv(files["y_train"])
    y_test = pd.read_csv(files["y_test"])

    if X_train.empty or X_test.empty:
        raise ValueError("Processed feature data is empty.")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Training and testing feature counts differ.")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and testing feature names differ.")

    for name, frame in {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }.items():
        if frame.isna().any().any():
            raise ValueError(f"Missing values found in {name}.")

    if not np.isfinite(X_train.to_numpy(dtype=float)).all():
        raise ValueError("Invalid numeric values in training features.")

    if not np.isfinite(X_test.to_numpy(dtype=float)).all():
        raise ValueError("Invalid numeric values in testing features.")

    preprocessor_path = Path("models/shipment_preprocessor.joblib")
    if not preprocessor_path.exists():
        raise FileNotFoundError("Preprocessor artifact not found.")

    joblib.load(preprocessor_path)

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and target row counts do not match.")

    report = {
        "lab": 4,
        "stage": "preprocessing_output_validation",
        "status": "PASSED",
        "training_shape": list(X_train.shape),
        "testing_shape": list(X_test.shape),
        "training_target_rows": len(y_train),
        "testing_target_rows": len(y_test),
        "missing_values": 0,
        "numeric_values_finite": True,
        "preprocessor_artifact_valid": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path = Path("reports/lab04_output_validation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Training features: {X_train.shape}")
    print(f"Testing features: {X_test.shape}")
    print(f"Training target rows: {len(y_train)}")
    print(f"Testing target rows: {len(y_test)}")
    print("Missing values: 0")
    print("Numeric values: finite")
    print("Preprocessor artifact: valid")
    print(f"Report saved: {report_path}")
    print("[SUCCESS] LAB 4 PASSED")


if __name__ == "__main__":
    main()
