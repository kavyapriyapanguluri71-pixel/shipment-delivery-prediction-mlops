import json

import pytest

from mlops_pipeline.inference_monitoring import summarize_log


def _write_records(path, records) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_summary_reports_request_prediction_probability_and_latency_metrics(tmp_path) -> None:
    log_path = tmp_path / "inference.jsonl"
    output_path = tmp_path / "reports" / "summary.json"
    _write_records(
        log_path,
        [
            {"prediction": 1, "late_probability": 0.8, "latency_ms": 10},
            {"prediction": 0, "late_probability": 0.2, "latency_ms": 20},
            {"prediction": 1, "late_probability": 0.6, "latency_ms": 30},
        ],
    )

    report = summarize_log(log_path, output_path)

    assert report["request_count"] == 3
    assert report["late_prediction_count"] == 2
    assert report["late_prediction_rate"] == pytest.approx(2 / 3)
    assert report["mean_late_probability"] == pytest.approx(0.5333333333)
    assert report["latency_record_count"] == 3
    assert report["latency_ms"] == {"p50": 20.0, "p95": 29.0, "max": 30.0}
    assert json.loads(output_path.read_text(encoding="utf-8")) == report


def test_summary_accepts_legacy_records_without_latency(tmp_path) -> None:
    log_path = tmp_path / "legacy.jsonl"
    _write_records(log_path, [{"prediction": 0, "late_probability": 0.1}])

    report = summarize_log(log_path)

    assert report["latency_record_count"] == 0
    assert report["latency_ms"] == {"p50": None, "p95": None, "max": None}


@pytest.mark.parametrize(
    "contents, message",
    [
        ('{"prediction": 1, "late_probability": 1.2}', "Invalid late_probability"),
        ('{"prediction": 1, "late_probability": 0.5, "latency_ms": -1}', "Invalid latency_ms"),
        ("not-json", "Invalid JSON"),
    ],
)
def test_summary_rejects_invalid_log_records(tmp_path, contents, message) -> None:
    log_path = tmp_path / "bad.jsonl"
    log_path.write_text(contents + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        summarize_log(log_path)


def test_summary_rejects_missing_and_empty_logs(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        summarize_log(tmp_path / "absent.jsonl")
    empty_log = tmp_path / "empty.jsonl"
    empty_log.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no prediction records"):
        summarize_log(empty_log)
