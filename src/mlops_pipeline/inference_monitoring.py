"""Summarize privacy-safe prediction logs for operational monitoring."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize_log(
    log_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Read JSONL prediction records and report volume, score, and latency."""
    source = Path(log_path)
    if not source.is_file():
        raise FileNotFoundError(f"Inference log does not exist: {source}")

    records: list[dict[str, object]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {source} at line {line_number}.") from error
        if not isinstance(record, dict):
            raise ValueError(f"Expected a JSON object at line {line_number} in {source}.")
        prediction = record.get("prediction")
        probability = record.get("late_probability")
        if prediction not in (0, 1) or not isinstance(probability, (int, float)):
            raise ValueError(f"Invalid prediction record at line {line_number} in {source}.")
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"Invalid late_probability at line {line_number} in {source}.")
        latency = record.get("latency_ms")
        if latency is not None and (
            not isinstance(latency, (int, float))
            or not np.isfinite(latency)
            or latency < 0
        ):
            raise ValueError(f"Invalid latency_ms at line {line_number} in {source}.")
        records.append(record)

    if not records:
        raise ValueError(f"Inference log contains no prediction records: {source}")

    probabilities = np.asarray([record["late_probability"] for record in records], dtype=float)
    predictions = np.asarray([record["prediction"] for record in records], dtype=int)
    latencies = np.asarray(
        [record["latency_ms"] for record in records if record.get("latency_ms") is not None],
        dtype=float,
    )
    report: dict[str, object] = {
        "source_file": source.name,
        "request_count": len(records),
        "late_prediction_count": int(predictions.sum()),
        "late_prediction_rate": float(predictions.mean()),
        "mean_late_probability": float(probabilities.mean()),
        "latency_record_count": int(len(latencies)),
        "latency_ms": {
            "p50": float(np.percentile(latencies, 50)) if len(latencies) else None,
            "p95": float(np.percentile(latencies, 95)) if len(latencies) else None,
            "max": float(latencies.max()) if len(latencies) else None,
        },
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        default="logs/inference/inference_log.jsonl",
        help="JSONL inference metadata log.",
    )
    parser.add_argument("--output", default="outputs/inference_monitoring.json", help="Report path.")
    args = parser.parse_args()
    print(json.dumps(summarize_log(args.log, args.output), indent=2))


if __name__ == "__main__":
    main()
