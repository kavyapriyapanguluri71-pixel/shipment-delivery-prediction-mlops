"""Lab 5: Train shipment model and register a versioned artifact."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from mlops_pipeline.train import train


def main():
    print("=" * 60)
    print("LAB 5: MODEL TRAINING AND REGISTRATION")
    print("=" * 60)

    # Run the existing shipment training workflow.
    train(
        data_path="Train.csv",
        output_dir="outputs",
        model_dir="models",
    )

    model_path = Path("models/shipments_pipeline.joblib")

    if not model_path.exists():
        raise FileNotFoundError(
            "Training completed, but the expected model artifact "
            "models/shipments_pipeline.joblib was not created."
        )

    # Create a versioned local registry entry.
    timestamp = datetime.now(timezone.utc)
    version = timestamp.strftime("v%Y%m%d%H%M%S")

    registry_dir = Path("models/registry") / version
    registry_dir.mkdir(parents=True, exist_ok=False)

    registered_model = registry_dir / "shipments_pipeline.joblib"
    shutil.copy2(model_path, registered_model)

    metadata_path = Path("models/model_metadata.json")
    metadata = {}

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    registry_path = Path("models/registry.json")

    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {"model_name": "shipment_prediction", "versions": []}

    entry = {
        "version": version,
        "status": "REGISTERED",
        "artifact": str(registered_model).replace("\\", "/"),
        "source_artifact": str(model_path).replace("\\", "/"),
        "dataset": "Train.csv",
        "metadata": metadata,
        "registered_at": timestamp.isoformat(),
    }

    registry["versions"].append(entry)

    registry_path.write_text(
        json.dumps(registry, indent=2) + "\n",
        encoding="utf-8",
    )

    # Verify the registered artifact exists and is not empty.
    if not registered_model.exists() or registered_model.stat().st_size == 0:
        raise ValueError("Registered model artifact is missing or empty.")

    report = {
        "lab": 5,
        "stage": "model_training_and_registration",
        "status": "PASSED",
        "registry_type": "local_file_registry",
        "model_version": version,
        "registered_artifact": str(registered_model).replace("\\", "/"),
        "registry_file": str(registry_path).replace("\\", "/"),
        "artifact_size_bytes": registered_model.stat().st_size,
        "generated_at": timestamp.isoformat(),
    }

    report_path = Path("reports/lab05_model_registry.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Model artifact: {model_path}")
    print(f"Registered version: {version}")
    print(f"Registered artifact: {registered_model}")
    print(f"Registry file: {registry_path}")
    print(f"Report saved: {report_path}")
    print("[SUCCESS] LAB 5 PASSED")


if __name__ == "__main__":
    main()
