import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LAB_DIR = PROJECT_ROOT / "src" / "mlops_pipeline"
REPORT_DIR = PROJECT_ROOT / "reports"

LABS = [
    "01_load_data.py",
    "02_validate_data.py",
    "03_preprocessing.py",
    "04_validate_outputs.py",
    "05_train_registry.py",
    "06_model_evaluation.py",
    "07_quality_gate.py",
    "08_automate_lifecycle.py",
    "09_inference.py",
    "10_failure_injection.py",
    "11_prepare_deployment.py",
    "12_api_performance_monitoring.py",
    "13_end_to_end_monitoring.py",
]

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    log_parts = []
    overall_status = "PASSED"

    print("=" * 65)
    print("SHIPMENT PREDICTION - ALL 13 MLOPS LABS")
    print("=" * 65)

    for number, filename in enumerate(LABS, start=1):
        script = LAB_DIR / filename

        print(f"\n[{number}/13] Running Lab {number}: {filename}")

        if not script.is_file():
            status = "FAILED"
            output = f"Lab file not found: {script}"
            duration = 0.0
        else:
            start = time.perf_counter()
            try:
                process = subprocess.run(
                    [sys.executable, str(script)],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                duration = round(time.perf_counter() - start, 2)
                output = process.stdout
                if process.stderr:
                    output += "\n--- STDERR ---\n" + process.stderr

                success_marker = f"[SUCCESS] LAB {number} PASSED"
                status = (
                    "PASSED"
                    if process.returncode == 0 and success_marker in output
                    else "FAILED"
                )
            except Exception as error:
                duration = round(time.perf_counter() - start, 2)
                output = repr(error)
                status = "FAILED"

        print(output)
        print(f"Lab {number} status: {status} ({duration} seconds)")

        results.append({
            "lab": number,
            "script": filename,
            "status": status,
            "duration_seconds": duration,
        })
        log_parts.append(
            f"\n{'=' * 65}\nLAB {number}: {filename}\n"
            f"STATUS: {status}\n{'=' * 65}\n{output}\n"
        )

        if status != "PASSED":
            overall_status = "FAILED"
            print("\nStopping because a lab failed.")
            break

    completed = len(results)
    passed = sum(item["status"] == "PASSED" for item in results)

    for number in range(completed + 1, len(LABS) + 1):
        results.append({
            "lab": number,
            "script": LABS[number - 1],
            "status": "NOT_RUN",
            "duration_seconds": 0,
        })

    report = {
        "project": "Shipment Prediction MLOps",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "overall_status": overall_status,
        "total_labs": len(LABS),
        "passed": passed,
        "failed": sum(item["status"] == "FAILED" for item in results),
        "not_run": sum(item["status"] == "NOT_RUN" for item in results),
        "labs": results,
    }

    json_path = REPORT_DIR / "all_labs_execution.json"
    log_path = REPORT_DIR / "all_labs_execution.log"

    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log_path.write_text("".join(log_parts), encoding="utf-8")

    print("\n" + "=" * 65)
    print(f"FINAL RESULT: {overall_status}")
    print(f"Passed: {passed}/13")
    print(f"Failed: {report['failed']}")
    print(f"Not run: {report['not_run']}")
    print(f"JSON report: {json_path}")
    print(f"Execution log: {log_path}")
    print("=" * 65)

    return 0 if overall_status == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
