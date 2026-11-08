"""Lab 9: Run inference using the production shipment model."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from mlops_pipeline.schema import FEATURE_COLUMNS, TARGET_COLUMN
from mlops_pipeline.validation import load_training_data


def main():
    print("=" * 60)
    print("LAB 9: SHIPMENT MODEL INFERENCE")
    print("=" * 60)

    registry_path = Path("models/registry.json")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    production_version = registry.get("production_version")
    if not production_version:
        raise ValueError("No production model version found.")

    entry = next(
        (
            item for item in registry["versions"]
            if item["version"] == production_version
            and item["status"] == "PRODUCTION"
        ),
        None,
    )

    if entry is None:
        raise ValueError("Production model registry entry is invalid.")

    model_path = Path(entry["artifact"])
    if not model_path.exists():
        raise FileNotFoundError(f"Production model missing: {model_path}")

    model = joblib.load(model_path)

    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise TypeError("Loaded artifact is not a prediction pipeline.")

    df = load_training_data("Train.csv")

    # Use one feature row from the dataset for a demonstration prediction.
    sample = df.loc[:, list(FEATURE_COLUMNS)].iloc[[0]]
    actual = int(df[TARGET_COLUMN].iloc[0])

    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]

    classes = list(model.classes_)
    late_index = classes.index(1)
    late_probability = float(probabilities[late_index])

    result = {
        "lab": 9,
        "stage": "model_inference",
        "status": "PASSED",
        "model_version": production_version,
        "prediction": prediction,
        "prediction_label": (
            "Late shipment" if prediction == 1 else "On-time shipment"
        ),
        "late_probability": late_probability,
        "actual_label": actual,
        "actual_label_meaning": (
            "Late shipment" if actual == 1 else "On-time shipment"
        ),
        "prediction_matches_actual": prediction == actual,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path = Path("reports/lab09_inference.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Production version: {production_version}")
    print(f"Prediction: {result['prediction_label']}")
    print(f"Late probability: {late_probability:.4f}")
    print(f"Actual label: {result['actual_label_meaning']}")
    print(f"Report saved: {report_path}")
    print("[SUCCESS] LAB 9 PASSED")


if __name__ == "__main__":
    main()
