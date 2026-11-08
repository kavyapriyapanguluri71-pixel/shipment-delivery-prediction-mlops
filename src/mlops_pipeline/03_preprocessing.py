"""Lab 3: Preprocess shipment features and save reproducible outputs."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_pipeline.schema import FEATURE_COLUMNS, TARGET_COLUMN
from mlops_pipeline.train import build_preprocessor
from mlops_pipeline.validation import load_training_data


def main():
    print("=" * 60)
    print("LAB 3: SHIPMENT DATA PREPROCESSING")
    print("=" * 60)

    frame = load_training_data("Train.csv")
    X = frame.loc[:, list(FEATURE_COLUMNS)]
    y = frame[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    train_output = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
    )
    test_output = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
    )

    output_dir = Path("outputs/lab03")
    model_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_output.to_csv(output_dir / "X_train_processed.csv", index=False)
    test_output.to_csv(output_dir / "X_test_processed.csv", index=False)

    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    joblib.dump(preprocessor, model_dir / "shipment_preprocessor.joblib")

    if not np.isfinite(train_output.to_numpy()).all():
        raise ValueError("Training preprocessing generated invalid values.")

    if not np.isfinite(test_output.to_numpy()).all():
        raise ValueError("Testing preprocessing generated invalid values.")

    report = {
        "lab": 3,
        "stage": "data_preprocessing",
        "status": "PASSED",
        "original_rows": int(len(frame)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "original_features": int(X.shape[1]),
        "processed_features": int(len(feature_names)),
        "train_shape": list(train_output.shape),
        "test_shape": list(test_output.shape),
        "scaling": "StandardScaler",
        "categorical_encoding": "OneHotEncoder",
        "random_state": 42,
        "preprocessor": "models/shipment_preprocessor.joblib",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path = Path("reports/lab03_preprocessing.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Processed features: {len(feature_names)}")
    print(f"Training output: {train_output.shape}")
    print(f"Testing output: {test_output.shape}")
    print(f"Report saved: {report_path}")
    print("[SUCCESS] LAB 3 PASSED")


if __name__ == "__main__":
    main()
