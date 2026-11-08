"""Build-independent smoke verification for the running shipment API image."""

import argparse
import json
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4


PAYLOAD = {
    "Warehouse_block": "D",
    "Mode_of_Shipment": "Flight",
    "Customer_care_calls": 4,
    "Customer_rating": 2,
    "Cost_of_the_Product": 177,
    "Prior_purchases": 3,
    "Product_importance": "low",
    "Gender": "F",
    "Discount_offered": 44,
    "Weight_in_gms": 1233,
}


def _run_docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _request_json(url: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def verify(image: str) -> dict[str, object]:
    container_name = f"shipment-api-verify-{uuid4().hex[:10]}"
    created = _run_docker(
        "run",
        "--detach",
        "--name",
        container_name,
        "--publish",
        "127.0.0.1::8000",
        "--env",
        "INFERENCE_LOG_PATH=/tmp/docker-verification.jsonl",
        image,
    )
    container_id = created.stdout.strip()
    results: dict[str, object] = {"image": image, "container_started": True}
    try:
        dependency_check = _run_docker("exec", container_id, "python", "-m", "pip", "check")
        results["dependencies"] = dependency_check.stdout.strip() or "all requirements satisfied"

        port_output = _run_docker("port", container_id, "8000/tcp").stdout.strip()
        port = port_output.rsplit(":", maxsplit=1)[-1]
        base_url = f"http://127.0.0.1:{port}"

        last_error: Exception | None = None
        health: dict[str, object] | None = None
        for _ in range(60):
            try:
                health = _request_json(f"{base_url}/health")
                break
            except (URLError, TimeoutError, OSError) as error:
                last_error = error
                time.sleep(1)
        if health is None:
            raise RuntimeError(f"API health endpoint did not become available: {last_error}")
        if health != {"status": "ok", "model_loaded": True}:
            raise RuntimeError(f"Unexpected health response: {health}")
        results["health"] = health

        prediction = _request_json(f"{base_url}/predict", PAYLOAD)
        if (
            prediction.get("prediction") not in (0, 1)
            or prediction.get("delivery_status") not in ("late", "on_time")
            or not 0 <= float(prediction.get("late_probability", -1)) <= 1
            or not prediction.get("request_id")
        ):
            raise RuntimeError(f"Unexpected prediction response: {prediction}")
        results["prediction"] = {
            "prediction": prediction["prediction"],
            "delivery_status": prediction["delivery_status"],
            "late_probability": prediction["late_probability"],
        }

        artifact_check = _run_docker(
            "exec",
            container_id,
            "python",
            "-c",
            "from api.model_loader import load_model; model = load_model(); "
            "assert hasattr(model, 'predict_proba'); print(type(model).__name__)",
        )
        results["model_artifact"] = {
            "loaded_inside_container": True,
            "type": artifact_check.stdout.strip(),
        }

        log_result = _run_docker("logs", container_id)
        logs = log_result.stdout + log_result.stderr
        error_markers = [marker for marker in ("Traceback", "ERROR") if marker.lower() in logs.lower()]
        if error_markers:
            raise RuntimeError(f"Container logs contain error markers: {error_markers}\n{logs}")
        results["container_logs"] = "no Traceback or ERROR entries"
        return results
    except Exception:
        logs = _run_docker("logs", container_id, check=False)
        print(logs.stdout + logs.stderr, file=sys.stderr)
        raise
    finally:
        _run_docker("rm", "--force", container_id, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="shipment-prediction-api:verify")
    args = parser.parse_args()
    print(json.dumps(verify(args.image), indent=2))


if __name__ == "__main__":
    main()
