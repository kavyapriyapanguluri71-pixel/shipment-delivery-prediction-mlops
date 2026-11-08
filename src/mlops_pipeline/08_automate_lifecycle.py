"""Lab 8: Automate shipment model lifecycle promotion."""

import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    print("=" * 60)
    print("LAB 8: MODEL LIFECYCLE AUTOMATION")
    print("=" * 60)

    registry_path = Path("models/registry.json")
    quality_path = Path("reports/lab07_quality_gate.json")

    if not registry_path.exists():
        raise FileNotFoundError("Model registry not found.")

    if not quality_path.exists():
        raise FileNotFoundError("Lab 7 quality report not found.")

    quality = json.loads(quality_path.read_text(encoding="utf-8"))

    if quality.get("status") != "PASSED":
        raise RuntimeError("Quality gate did not pass.")

    if quality.get("overall_quality_gate") is not True:
        raise RuntimeError("Model is not eligible for promotion.")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    versions = registry.get("versions", [])

    if not versions:
        raise ValueError("No registered model versions found.")

    latest = versions[-1]
    artifact = Path(latest["artifact"])

    if not artifact.exists():
        raise FileNotFoundError(
            f"Registered model artifact not found: {artifact}"
        )

    # Promote only the latest successfully registered version.
    for version in versions:
        version["status"] = (
            "PRODUCTION"
            if version["version"] == latest["version"]
            else "ARCHIVED"
        )

    registry["production_version"] = latest["version"]
    registry["updated_at"] = datetime.now(timezone.utc).isoformat()

    registry_path.write_text(
        json.dumps(registry, indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "lab": 8,
        "stage": "model_lifecycle_automation",
        "status": "PASSED",
        "quality_gate_verified": True,
        "promoted_version": latest["version"],
        "production_artifact": str(artifact).replace("\\", "/"),
        "production_status": "PRODUCTION",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path = Path("reports/lab08_lifecycle.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Promoted model: {latest['version']}")
    print("Production status: PRODUCTION")
    print(f"Registry updated: {registry_path}")
    print(f"Report saved: {report_path}")
    print("[SUCCESS] LAB 8 PASSED")


if __name__ == "__main__":
    main()
