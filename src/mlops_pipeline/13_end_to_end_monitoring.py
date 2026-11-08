"""Lab 13: End-to-end shipment MLOps monitoring."""

import json
from datetime import datetime, timezone
from pathlib import Path

from mlops_pipeline.monitoring import compare_batches


def main():
    print("=" * 60)
    print("LAB 13: END-TO-END MONITORING")
    print("=" * 60)

    checks = {}

    required_paths = {
        "training_dataset": Path("Train.csv"),
        "production_model": Path("models/shipments_pipeline.joblib"),
        "model_registry": Path("models/registry.json"),
        "inference_report": Path("reports/lab09_inference.json"),
        "performance_report": Path("reports/lab12_api_performance.json"),
    }

    for name, path in required_paths.items():
        checks[name] = path.exists()
        print(f"{name}: {'FOUND' if checks[name] else 'MISSING'}")

    if not all(checks.values()):
        raise FileNotFoundError("An end-to-end monitoring input is missing.")

    registry = json.loads(
        Path("models/registry.json").read_text(encoding="utf-8")
    )

    production_version = registry.get("production_version")

    if not production_version:
        raise RuntimeError("Production model version is not configured.")

    # Compare the training dataset with a separate sample from that dataset.
    # This verifies the drift-monitoring workflow, not real production drift.
    import pandas as pd

    frame = pd.read_csv("Train.csv")
    reference_path = Path("outputs/lab13_reference.csv")
    current_path = Path("outputs/lab13_current.csv")

    reference_path.parent.mkdir(parents=True, exist_ok=True)
    frame.iloc[:500].to_csv(reference_path, index=False)
    frame.iloc[500:1000].to_csv(current_path, index=False)

    drift_path = Path("reports/lab13_drift_report.json")

    drift_result = compare_batches(
        reference_path,
        current_path,
        output_path=drift_path,
    )

    drift_report = (
        json.loads(drift_path.read_text(encoding="utf-8"))
        if drift_path.exists()
        else {"result": str(drift_result)}
    )

    previous_labs = [
        Path(f"reports/lab{i:02d}_{name}.json")
        for i, name in [
            (2, "validation"),
            (3, "preprocessing"),
            (4, "output_validation"),
            (5, "model_registry"),
            (6, "model_evaluation"),
            (7, "quality_gate"),
            (8, "lifecycle"),
            (9, "inference"),
            (10, "failure_handling"),
            (11, "deployment_verification"),
            (12, "api_performance"),
        ]
    ]

    missing_reports = [
        str(path) for path in previous_labs if not path.exists()
    ]

    if missing_reports:
        raise FileNotFoundError(
            "Missing lab reports: " + ", ".join(missing_reports)
        )

    report = {
        "lab": 13,
        "stage": "end_to_end_monitoring",
        "status": "PASSED",
        "production_version": production_version,
        "checks": checks,
        "previous_lab_reports_verified": len(previous_labs),
        "drift_monitoring": "EXECUTED",
        "drift_comparison": "Two samples from the training dataset",
        "drift_report": str(drift_path).replace("\\", "/"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path = Path("reports/lab13_end_to_end_monitoring.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Production version: {production_version}")
    print(f"Previous lab reports verified: {len(previous_labs)}")
    print("Drift monitoring: executed")
    print(f"Final report saved: {report_path}")
    print("[SUCCESS] LAB 13 PASSED")


if __name__ == "__main__":
    main()
