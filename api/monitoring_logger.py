"""Append privacy-conscious prediction metadata for operational monitoring."""

import json
from datetime import datetime, timezone
from pathlib import Path


def log_prediction(
    path: str | Path,
    request_id: str,
    prediction: int,
    late_probability: float,
    latency_ms: float,
) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "prediction": prediction,
        "late_probability": late_probability,
        "latency_ms": latency_ms,
    }
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, allow_nan=False) + "\n")
