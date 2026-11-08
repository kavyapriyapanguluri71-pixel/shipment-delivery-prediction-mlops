# Shipment Delivery Prediction MLOps

An end-to-end project for training, evaluating, serving, and monitoring a model
that predicts late e-commerce shipments. The repository includes the shipment
dataset, exploratory notebook, research paper, trained model, API, automated
tests, Docker deployment, and Kubernetes manifests.

## Dataset

`Train.csv` contains 10,999 shipment records. The binary target,
`Reached.on.Time_Y.N`, uses **1 for late delivery** and **0 for on-time delivery**.
`ID` is an identifier and is not used as a model feature. The model uses these
ten input fields: warehouse block, shipment mode, customer-care calls, customer
rating, product cost, prior purchases, product importance, gender, discount,
and shipment weight.

The dataset, notebook (`ML-OPS.ipynb`), and research paper are retained as
project inputs and reference material. All are small enough for the repository;
the Docker build excludes the dataset and notebook because the serving image
only needs the packaged application and trained model.

## Architecture

1. `mlops_pipeline.validation` checks the training CSV schema and labels.
2. A stratified holdout is reserved. Four scikit-learn pipelines are compared
   with cross-validation on the training partition, and the best candidate is
   tuned before final evaluation on the holdout.
3. Preprocessing (numeric imputation and scaling, categorical imputation and
   one-hot encoding) is fit within each pipeline fold to avoid leakage.
4. The complete fitted pipeline and metadata are stored in `models/`; metrics,
   comparisons, feature importance, and plots are stored in `outputs/`.
5. FastAPI loads the model pipeline for `/health` and `/predict`. Inference
   logs store request metadata, prediction probability, and latency, not input
   feature values.
6. Drift and inference-monitoring commands summarize batch feature changes and
   operational prediction statistics. Docker and Kubernetes provide serving
   deployment paths.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for more detail.

## Setup and training

Python 3.12 is used in CI; Python 3.10 or newer is supported.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m mlops_pipeline.train --data Train.csv
```

Training compares Logistic Regression, Decision Tree, Random Forest, and
Gradient Boosting models using stratified cross-validation. It tunes the
highest ROC-AUC candidate and reports final results on an untouched stratified
test partition. Run `python -m mlops_pipeline.train --help` to change the data
path or artifact output directories.

## Model evaluation

The checked-in evaluation is a reproducibility baseline, not a performance
guarantee. It reports accuracy, precision, recall, F1, ROC-AUC, a confusion
matrix, and a classification report in `outputs/final_metrics.json`. The
current baseline has accuracy **0.678**, precision **0.952**, recall **0.485**,
F1 **0.643**, and ROC-AUC **0.759**. Review recall together with precision,
since missed late shipments may have operational cost. Training refreshes the
metrics and plots under `outputs/` and the canonical serving artifact at
`models/shipments_pipeline.joblib`.

## API

After training, start the service:

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

`GET /health` reports model readiness. `POST /predict` requires the ten model
features and rejects extra fields such as `ID`. Example PowerShell request:

```powershell
$shipment = @{
  Warehouse_block = "D"
  Mode_of_Shipment = "Flight"
  Customer_care_calls = 4
  Customer_rating = 2
  Cost_of_the_Product = 177
  Prior_purchases = 3
  Product_importance = "low"
  Gender = "F"
  Discount_offered = 44
  Weight_in_gms = 1233
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" -Body $shipment
```

Set `MODEL_PATH` to select another trusted joblib pipeline and
`INFERENCE_LOG_PATH` to move the JSONL metadata log.

## Data drift monitoring

Compare an incoming CSV batch to the training reference:

```powershell
python -m mlops_pipeline.monitoring --reference Train.csv --current new_batch.csv
```

The report goes to `outputs/drift_report.json`. It uses PSI thresholds: below
0.1 is low, 0.1 to below 0.2 is moderate, and 0.2 or higher is high drift.
These are review signals, not automatic retraining triggers.

## Inference monitoring

Summarize privacy-safe prediction logs:

```powershell
python -m mlops_pipeline.inference_monitoring `
  --log logs/inference/inference_log.jsonl
```

The report at `outputs/inference_monitoring.json` contains request count,
late-prediction rate, mean late probability, and latency percentiles. It does
not measure accuracy because actual delivery outcomes are not logged.

## Docker deployment

Build the image after a trained model is available, then verify the image and
its API:

```powershell
docker build -t shipment-prediction-api:verify .
python scripts/verify_docker.py --image shipment-prediction-api:verify
```

Run it manually with:

```powershell
docker run --rm -p 8000:8000 shipment-prediction-api:verify
```

The smoke verifier checks dependencies, health, a prediction request, model
loading inside the container, and logs.

## Kubernetes deployment

The local Docker Desktop manifests use the same `:verify` image tag:

```powershell
kubectl config current-context
python scripts/verify_kubernetes.py
```

The verifier validates the manifests against the active cluster, deploys the
API, waits for Ready status, checks the Service endpoint and HTTP probes, tests
health and prediction, and checks pod logs. It leaves the deployment running.
Use `python scripts/verify_kubernetes.py --cleanup` to remove its Service and
Deployment, or see [docs/RUNBOOK.md](docs/RUNBOOK.md) for manual commands and
remote-cluster image guidance.

## Automated testing

Install development dependencies through `requirements.txt`, then run:

```powershell
python scripts/run_tests.py
```

This runs pytest and writes `reports/test_report.md` and
`reports/test-results.xml`. The tests cover API behavior, data drift, inference
monitoring, model loading, input validation, and the training pipeline.

## GitHub Actions CI

`.github/workflows/ci.yml` runs on pushes and pull requests using Python 3.12.
It runs the test suite, uploads the Markdown and JUnit test reports, builds the
API Docker image, and runs the Docker smoke verifier. Workflow results are
available in the repository's **Actions** tab after the workflow runs.

## Project files

- `api/`: FastAPI application, schemas, model loader, and metadata logger.
- `src/mlops_pipeline/`: training, validation, drift, and inference monitoring.
- `models/`: canonical serving pipeline and model metadata.
- `outputs/`: model evaluation, EDA, and notebook compatibility artifacts.
- `tests/`: automated unit and API tests.
- `scripts/`: test reports and Docker/Kubernetes verification.
- `docs/`: architecture and operations runbook.
- `k8s/`: Deployment and ClusterIP Service manifests.
- `.github/workflows/`: CI workflow.
