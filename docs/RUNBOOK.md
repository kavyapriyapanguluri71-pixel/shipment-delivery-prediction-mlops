# Operations runbook

This runbook describes the supported training and serving path for shipment
delivery prediction. The supplied churn project tree is a structural reference;
the dataset and artifacts here are for shipment lateness.

## Train or refresh the model

From the repository root, activate the project environment and run:

```powershell
python -m pip install -r requirements.txt
python -m mlops_pipeline.train --data Train.csv
```

The training command checks the dataset schema, compares four classifiers with
stratified cross-validation, tunes the selected classifier, and evaluates it on
a held-out split. Review `outputs/final_metrics.json` and
`outputs/model_comparison.csv` before serving a refreshed artifact. The current
evaluation is a baseline; in particular, review recall alongside precision
because missed late shipments may matter operationally.

The serving artifact and its source-data hash are written to
`models/shipments_pipeline.joblib` and `models/model_metadata.json`. Evaluation
reports and plots are written under `outputs/`. Keep those files together when
recording a training run.

## Run the API

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Check `GET /health` for readiness, then send a shipment feature object to
`POST /predict`. The API rejects fields outside the documented schema. It logs
request ID, UTC timestamp, prediction, and late probability to
`logs/inference/inference_log.jsonl`; shipment feature values are not logged.
Set `MODEL_PATH` and `INFERENCE_LOG_PATH` to override the defaults.

Summarize the local inference log with:

```powershell
python -m mlops_pipeline.inference_monitoring --log logs/inference/inference_log.jsonl
```

This writes request volume, late-prediction rate, mean late probability, and
latency percentiles to `outputs/inference_monitoring.json`. Historical records
without a latency field remain readable. These are operational indicators;
they do not measure prediction accuracy because actual delivery outcomes are
not part of the inference log.

## Check for input data drift

Compare a new batch with the training data before using it for a retraining
decision:

```powershell
python -m mlops_pipeline.monitoring --reference Train.csv --current new_batch.csv
```

The report is written to `outputs/drift_report.json`. It uses PSI with bins
defined from the reference batch: below 0.1 is reported as low drift, 0.1 to
below 0.2 as moderate, and 0.2 or higher as high. Treat these as review signals,
not automatic retraining triggers. The input batch must contain all ten shipment
feature columns; extra columns such as `ID` and the label are ignored.

## Container deployment

Build after training so the model artifact is available to Docker:

```powershell
docker build -t shipment-prediction-api:verify .
python scripts/verify_docker.py --image shipment-prediction-api:verify
```

The verification script starts a temporary container on a free local port,
checks `GET /health`, posts a sample shipment to `POST /predict`, loads the model
inside the container, checks container logs for tracebacks or errors, and then
removes the temporary container. To run the container manually instead:

```powershell
docker run --rm -p 8000:8000 shipment-prediction-api:verify
```

The API listens on port 8000. If startup fails, verify that
`models/shipments_pipeline.joblib` exists and that the image was built after the
latest training run. The model artifact is Python/joblib serialized and should
only be loaded from a trusted source.

## Kubernetes

The local Docker Desktop deployment uses the existing `shipment-prediction-api:verify`
image. Confirm the cluster context before applying resources:

```powershell
kubectl config current-context
kubectl apply --dry-run=server -f k8s/deployment.yaml
kubectl apply --dry-run=server -f k8s/service.yaml
python scripts/verify_kubernetes.py
```

The verification script validates both manifests against the current API server,
applies them, waits for rollout, and checks the ready pod and Service endpoints.
It verifies the startup and readiness probe status, observes a full liveness
period without a restart, loads the model in the pod, checks the API health and
prediction endpoints through a temporary port forward, and scans pod logs for
tracebacks or errors. It leaves the deployment running. To access it manually:

```powershell
kubectl port-forward service/shipment-prediction-api 8000:8000
```

Then visit `http://127.0.0.1:8000/health` or send the shipment JSON from the API
example to `http://127.0.0.1:8000/predict`. The deployment uses the image's
bundled model, one replica, resource requests and limits, and HTTP probes. For a
remote cluster, push the image to an accessible registry and update the
Deployment's `image` field. Inference logs are written to pod-local `/tmp` and
are lost when a pod is replaced; use managed log retention in production.

Remove the local Kubernetes resources when finished:

```powershell
python scripts/verify_kubernetes.py --cleanup
```

Or remove them directly:

```powershell
kubectl delete -f k8s/service.yaml --ignore-not-found=true
kubectl delete -f k8s/deployment.yaml --ignore-not-found=true
```

## CI and local checks

Install the project with development dependencies, then run the report-producing
test command from the repository root:

```powershell
python -m pip install -r requirements.txt
python scripts/run_tests.py
```

The command runs the full pytest suite and creates `reports/test_report.md` (a
readable passed/failed test list) and `reports/test-results.xml` (JUnit XML).
GitHub Actions runs the same command on pushes and pull requests, uploads both
reports as the `shipment-project-test-report` workflow artifact, builds the API
container, and runs `scripts/verify_docker.py` against it. The workflow does not
retrain the model; training remains an explicit, reviewable operation.
