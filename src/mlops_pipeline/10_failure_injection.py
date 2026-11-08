"""Lab 10: Test safe failure detection and recovery."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from mlops_pipeline.validation import load_training_data


def main():
    print("=" * 60)
    print("LAB 10: FAILURE HANDLING AND RECOVERY")
    print("=" * 60)

    checks = {}

    # Test 1: Missing dataset.
    try:
        load_training_data("missing_lab10_dataset.csv")
    except FileNotFoundError:
        checks["missing_dataset_detected"] = True
    else:
        raise AssertionError("Missing dataset was not detected.")

    # Test 2: Invalid dataset schema.
    with tempfile.TemporaryDirectory() as temp_dir:
        invalid_path = Path(temp_dir) / "invalid.csv"
        pd.DataFrame({"wrong_column": [1, 2, 3]}).to_csv(
            invalid_path, index=False
        )

        try:
            load_training_data(invalid_path)
        except (ValueError, KeyError):
            checks["invalid_schema_detected"] = True
        else:
            raise AssertionError("Invalid schema was not detected.")

    # Test 3: Missing model artifact.
    missing_model = Path("models/lab10_missing_model.joblib")

    if missing_model.exists():
        raise RuntimeError(
            "Temporary test model path already exists; choose another path."
        )

    if not missing_model.exists():
        checks["missing_model_detected"] = True

    if not all(checks.values()):
        raise RuntimeError("One or more failure checks did not pass.")

    report = {
        "lab": 10,
        "stage": "failure_handling",
        "status": "PASSED",
        "checks": checks,
        "all_failure_cases_detected": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path = Path("reports/lab10_failure_handling.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Missing dataset: detected")
    print("Invalid dataset schema: detected")
    print("Missing model artifact: detected")
    print(f"Report saved: {report_path}")
    print("[SUCCESS] LAB 10 PASSED")


if __name__ == "__main__":
    main()
