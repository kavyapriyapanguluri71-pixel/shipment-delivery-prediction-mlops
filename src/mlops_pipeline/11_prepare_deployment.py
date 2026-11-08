"""Lab 11: Verify shipment deployment prerequisites and configurations."""

import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    print("=" * 60)
    print("LAB 11: DEPLOYMENT VERIFICATION")
    print("=" * 60)

    checks = {
        "dockerfile": Path("Dockerfile"),
        "docker_verification_script": Path("scripts/verify_docker.py"),
        "kubernetes_verification_script": Path("scripts/verify_kubernetes.py"),
        "production_model": Path("models/shipments_pipeline.joblib"),
        "model_registry": Path("models/registry.json"),
        "api_application": Path("api/main.py"),
    }

    results = {}

    for name, path in checks.items():
        results[name] = path.exists()
        print(f"{name}: {'FOUND' if results[name] else 'MISSING'}")

    if not all(results.values()):
        raise FileNotFoundError(
            "One or more deployment prerequisites are missing."
        )

    registry = json.loads(
        Path("models/registry.json").read_text(encoding="utf-8")
    )

    production_version = registry.get("production_version")
    if not production_version:
        raise RuntimeError("No production model registered.")

    production_entry = next(
        (
            item for item in registry["versions"]
            if item["version"] == production_version
            and item["status"] == "PRODUCTION"
        ),
        None,
    )

    if production_entry is None:
        raise RuntimeError("Production registry entry is invalid.")

    if not Path(production_entry["artifact"]).exists():
        raise FileNotFoundError("Production model artifact is missing.")

    report = {
        "lab": 11,
        "stage": "deployment_verification",
        "status": "PASSED",
        "checks": results,
        "production_version": production_version,
        "docker_runtime_test": "NOT_RUN",
        "kubernetes_runtime_test": "NOT_RUN",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path = Path("reports/lab11_deployment_verification.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Production version: {production_version}")
    print("Deployment prerequisites: verified")
    print("Docker/Kubernetes runtime tests: not run by this lab")
    print(f"Report saved: {report_path}")
    print("[SUCCESS] LAB 11 PASSED")


if __name__ == "__main__":
    main()
