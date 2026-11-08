"""Lab 7: Apply model quality gates using actual test metrics."""

import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    print("=" * 60)
    print("LAB 7: MODEL QUALITY GATE")
    print("=" * 60)

    metadata_path = Path("models/model_metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError("Model metadata not found.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = metadata.get("test_metrics", {})

    thresholds = {
        "accuracy": 0.60,
        "precision": 0.70,
        "recall": 0.40,
        "f1": 0.50,
        "roc_auc": 0.65,
    }

    results = {}
    for metric, minimum in thresholds.items():
        if metric not in metrics:
            raise ValueError(f"Missing metric: {metric}")

        actual = float(metrics[metric])
        passed = actual >= minimum

        results[metric] = {
            "actual": actual,
            "minimum_required": minimum,
            "passed": passed,
        }

        print(
            f"{metric.upper()}: {actual:.4f} "
            f"(minimum {minimum:.2f}) - "
            f"{'PASS' if passed else 'FAIL'}"
        )

    overall_pass = all(item["passed"] for item in results.values())

    report = {
        "lab": 7,
        "stage": "model_quality_gate",
        "model": metadata.get("model"),
        "status": "PASSED" if overall_pass else "FAILED",
        "overall_quality_gate": overall_pass,
        "checks": results,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path = Path("reports/lab07_quality_gate.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Report saved: {report_path}")

    if not overall_pass:
        raise RuntimeError(
            "Model failed the quality gate. "
            "Review the evaluation metrics before promotion."
        )

    print("[SUCCESS] LAB 7 PASSED")


if __name__ == "__main__":
    main()
