"""Lab 6: Evaluate the trained shipment prediction model."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main():
    print("=" * 60)
    print("LAB 6: SHIPMENT MODEL EVALUATION")
    print("=" * 60)

    metadata_path = Path("models/model_metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError("Model metadata not found.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    metrics = metadata.get("test_metrics", {})
    required_metrics = [
        "accuracy", "precision", "recall", "f1", "roc_auc"
    ]

    for metric in required_metrics:
        if metric not in metrics:
            raise ValueError(f"Missing evaluation metric: {metric}")

        value = float(metrics[metric])
        if not 0 <= value <= 1:
            raise ValueError(f"Invalid {metric} value: {value}")

    artifact_paths = [
        Path("models/shipments_pipeline.joblib"),
        Path("outputs/final_metrics.json"),
        Path("outputs/model_comparison.csv"),
        Path("outputs/confusion_matrix.png"),
        Path("outputs/roc_curve.png"),
        Path("outputs/feature_importance.csv"),
        Path("outputs/feature_importance.png"),
    ]

    missing = [str(path) for path in artifact_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing evaluation artifacts: " + ", ".join(missing)
        )

    comparison = pd.read_csv("outputs/model_comparison.csv")
    if comparison.empty:
        raise ValueError("Model comparison report is empty.")

    report = {
        "lab": 6,
        "stage": "model_evaluation",
        "status": "PASSED",
        "selected_model": metadata["model"],
        "test_metrics": {
            key: float(metrics[key]) for key in required_metrics
        },
        "model_comparison_rows": int(len(comparison)),
        "evaluation_artifacts_verified": [
            str(path).replace("\\", "/") for path in artifact_paths
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path = Path("reports/lab06_model_evaluation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Selected model: {metadata['model']}")
    for metric in required_metrics:
        print(f"{metric.upper()}: {float(metrics[metric]):.4f}")

    print(f"Compared models: {len(comparison)}")
    print("Evaluation plots and reports: verified")
    print(f"Report saved: {report_path}")
    print("[SUCCESS] LAB 6 PASSED")


if __name__ == "__main__":
    main()
