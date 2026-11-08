"""Lab 12: Measure production model inference performance."""

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from mlops_pipeline.schema import FEATURE_COLUMNS
from mlops_pipeline.validation import load_training_data


def main():
    print("=" * 60)
    print("LAB 12: API PERFORMANCE MONITORING")
    print("=" * 60)

    registry = json.loads(
        Path("models/registry.json").read_text(encoding="utf-8")
    )

    production_version = registry.get("production_version")
    entry = next(
        (
            item for item in registry["versions"]
            if item["version"] == production_version
            and item["status"] == "PRODUCTION"
        ),
        None,
    )

    if entry is None:
        raise RuntimeError("Production model not found.")

    model_path = Path(entry["artifact"])
    if not model_path.exists():
        raise FileNotFoundError("Production model artifact is missing.")

    model = joblib.load(model_path)

    data = load_training_data("Train.csv")
    samples = data.loc[:, list(FEATURE_COLUMNS)].head(100)

    if samples.empty:
        raise ValueError("No samples available for performance testing.")

    latencies = []
    predictions = []

    for _, row in samples.iterrows():
        sample = pd.DataFrame([row])

        start = time.perf_counter()
        prediction = int(model.predict(sample)[0])
        elapsed_ms = (time.perf_counter() - start) * 1000

        predictions.append(prediction)
        latencies.append(elapsed_ms)

    sorted_latencies = sorted(latencies)

    average_latency = statistics.mean(latencies)
    p95_latency = sorted_latencies[
        min(len(sorted_latencies) - 1, int(0.95 * len(sorted_latencies)))
    ]

    report = {
        "lab": 12,
        "stage": "api_performance_monitoring",
        "status": "PASSED",
        "model_version": production_version,
        "requests_tested": len(predictions),
        "successful_predictions": len(predictions),
        "average_latency_ms": average_latency,
        "p95_latency_ms": p95_latency,
        "maximum_latency_ms": max(latencies),
        "late_predictions": sum(prediction == 1 for prediction in predictions),
        "on_time_predictions": sum(prediction == 0 for prediction in predictions),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path = Path("reports/lab12_api_performance.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Model version: {production_version}")
    print(f"Requests tested: {len(predictions)}")
    print(f"Successful predictions: {len(predictions)}")
    print(f"Average latency: {average_latency:.3f} ms")
    print(f"P95 latency: {p95_latency:.3f} ms")
    print(f"Maximum latency: {max(latencies):.3f} ms")
    print(f"Report saved: {report_path}")
    print("[SUCCESS] LAB 12 PASSED")


if __name__ == "__main__":
    main()
