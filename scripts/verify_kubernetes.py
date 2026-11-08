"""Deploy and verify the shipment prediction API on the current Kubernetes context."""

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT = ROOT / "k8s" / "deployment.yaml"
SERVICE = ROOT / "k8s" / "service.yaml"
APP = "shipment-prediction-api"
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


def _kubectl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kubectl", *args],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _kubectl_json(*args: str) -> dict[str, object]:
    return json.loads(_kubectl(*args, "-o", "json").stdout)


def _free_local_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _http_json(url: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _assert_probes(pod: dict[str, object]) -> dict[str, bool | int]:
    containers = pod["spec"]["containers"]
    api = next(container for container in containers if container["name"] == "api")
    expected = {"startupProbe", "readinessProbe", "livenessProbe"}
    if not expected.issubset(api):
        raise RuntimeError(f"Pod is missing health probes: {sorted(expected - set(api))}")
    for name in expected:
        probe = api[name]
        if probe.get("httpGet", {}).get("path") != "/health":
            raise RuntimeError(f"{name} is not configured to call /health.")
        if probe.get("httpGet", {}).get("port") != "http":
            raise RuntimeError(f"{name} does not target the named HTTP port.")
    statuses = pod.get("status", {}).get("containerStatuses", [])
    api_status = next((item for item in statuses if item["name"] == "api"), None)
    ready_condition = any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in pod.get("status", {}).get("conditions", [])
    )
    if not api_status or not api_status.get("ready") or not ready_condition:
        raise RuntimeError("The API pod has not passed Kubernetes readiness.")
    if not api_status.get("started"):
        raise RuntimeError("The API container has not passed its startup probe.")
    return {
        "startup_probe_passed": True,
        "readiness_probe_passed": True,
        "restart_count": int(api_status.get("restartCount", 0)),
        "liveness_period_seconds": int(api["livenessProbe"].get("periodSeconds", 10)),
    }


def verify(cleanup: bool = False) -> dict[str, object]:
    result: dict[str, object] = {}
    _kubectl("apply", "--dry-run=server", "-f", str(DEPLOYMENT))
    _kubectl("apply", "--dry-run=server", "-f", str(SERVICE))
    result["manifest_validation"] = "server-side dry-run passed for both manifests"

    _kubectl("apply", "-f", str(DEPLOYMENT))
    _kubectl("apply", "-f", str(SERVICE))
    rollout = _kubectl("rollout", "status", f"deployment/{APP}", "--timeout=180s")
    result["rollout"] = rollout.stdout.strip()

    deployment = _kubectl_json("get", "deployment", APP)
    desired = deployment.get("spec", {}).get("replicas", 1)
    ready = deployment.get("status", {}).get("readyReplicas", 0)
    if ready < desired:
        raise RuntimeError(f"Deployment has {ready}/{desired} ready replicas.")

    pods = _kubectl_json("get", "pods", "-l", f"app={APP}")
    ready_pods = [
        pod
        for pod in pods.get("items", [])
        if any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in pod.get("status", {}).get("conditions", [])
        )
    ]
    if not ready_pods:
        raise RuntimeError("No Ready shipment API pod was found.")
    pod = ready_pods[0]
    pod_name = pod["metadata"]["name"]
    result["pod"] = {"name": pod_name, "phase": pod["status"].get("phase"), "ready": True}
    probe_result = _assert_probes(pod)
    result["probes"] = probe_result

    endpoints = _kubectl_json("get", "endpoints", APP)
    addresses = [
        address
        for subset in endpoints.get("subsets", [])
        for address in subset.get("addresses", [])
    ]
    if not addresses:
        raise RuntimeError("The service has no ready pod endpoints.")
    result["service"] = {"name": APP, "ready_endpoints": len(addresses)}

    # Observe one full liveness interval and ensure Kubernetes does not restart the API.
    time.sleep(probe_result["liveness_period_seconds"] + 2)
    observed_pod = _kubectl_json("get", "pod", pod_name)
    observed_status = next(
        item
        for item in observed_pod.get("status", {}).get("containerStatuses", [])
        if item["name"] == "api"
    )
    observed_ready = any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in observed_pod.get("status", {}).get("conditions", [])
    )
    if (
        int(observed_status.get("restartCount", 0)) != probe_result["restart_count"]
        or not observed_status.get("ready")
        or not observed_ready
    ):
        raise RuntimeError("The API pod failed to remain live and ready during probe observation.")
    result["probes"]["liveness_stable_for_seconds"] = probe_result["liveness_period_seconds"] + 2

    model_load = _kubectl(
        "exec",
        pod_name,
        "--",
        "python",
        "-c",
        "from api.model_loader import load_model; model = load_model(); "
        "assert hasattr(model, 'predict_proba'); print(type(model).__name__)",
    )
    result["model_artifact"] = {"loaded_in_pod": True, "type": model_load.stdout.strip()}

    port = _free_local_port()
    port_forward = subprocess.Popen(
        ["kubectl", "port-forward", "--address", "127.0.0.1", f"service/{APP}", f"{port}:8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        health = None
        last_error: Exception | None = None
        for _ in range(60):
            if port_forward.poll() is not None:
                raise RuntimeError(f"kubectl port-forward exited with {port_forward.returncode}.")
            try:
                health = _http_json(f"http://127.0.0.1:{port}/health")
                break
            except (URLError, TimeoutError, OSError) as error:
                last_error = error
                time.sleep(1)
        if health != {"status": "ok", "model_loaded": True}:
            raise RuntimeError(f"Health endpoint failed: {health or last_error}")
        result["health"] = health

        prediction = _http_json(f"http://127.0.0.1:{port}/predict", PAYLOAD)
        if (
            prediction.get("prediction") not in (0, 1)
            or prediction.get("delivery_status") not in ("late", "on_time")
            or not 0 <= float(prediction.get("late_probability", -1)) <= 1
            or not prediction.get("request_id")
        ):
            raise RuntimeError(f"Unexpected prediction response: {prediction}")
        result["prediction"] = {
            "prediction": prediction["prediction"],
            "delivery_status": prediction["delivery_status"],
            "late_probability": prediction["late_probability"],
        }
    finally:
        port_forward.terminate()
        try:
            port_forward.wait(timeout=10)
        except subprocess.TimeoutExpired:
            port_forward.kill()
            port_forward.wait(timeout=5)

    logs = _kubectl("logs", pod_name).stdout
    error_markers = [marker for marker in ("Traceback", "ERROR") if marker.lower() in logs.lower()]
    if error_markers:
        raise RuntimeError(f"Pod logs contain error markers {error_markers}:\n{logs}")
    result["pod_logs"] = "no Traceback or ERROR entries"

    if cleanup:
        _kubectl("delete", "-f", str(SERVICE), "--ignore-not-found=true")
        _kubectl("delete", "-f", str(DEPLOYMENT), "--ignore-not-found=true")
        result["cleanup"] = "deployment and service deleted"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete the deployment and service after successful verification.",
    )
    args = parser.parse_args()
    try:
        print(json.dumps(verify(cleanup=args.cleanup), indent=2))
    except Exception as error:
        print(f"Kubernetes verification failed: {error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
