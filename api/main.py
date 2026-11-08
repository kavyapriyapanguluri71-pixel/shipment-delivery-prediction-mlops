"""FastAPI application for late-shipment risk predictions."""

import os
from time import perf_counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, Request
from sklearn.pipeline import Pipeline

from api.model_loader import load_model
from api.monitoring_logger import log_prediction
from api.schemas import PredictionResponse, ShipmentFeatures
from mlops_pipeline.schema import FEATURE_COLUMNS


class ProbabilisticClassifier(Protocol):
    classes_: object

    def predict(self, features: pd.DataFrame) -> object: ...

    def predict_proba(self, features: pd.DataFrame) -> object: ...


def create_app(
    model: Pipeline | ProbabilisticClassifier | None = None,
    log_path: str | Path | None = None,
) -> FastAPI:
    inference_log = log_path or os.environ.get(
        "INFERENCE_LOG_PATH", "logs/inference/inference_log.jsonl"
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.model = model if model is not None else load_model()
        yield

    application = FastAPI(
        title="Shipment Delivery Prediction API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health(request: Request) -> dict[str, object]:
        loaded = hasattr(request.app.state, "model")
        return {"status": "ok" if loaded else "not_ready", "model_loaded": loaded}

    @application.post("/predict", response_model=PredictionResponse)
    def predict_shipment(
        shipment: ShipmentFeatures, request: Request
    ) -> PredictionResponse:
        started_at = perf_counter()
        features = pd.DataFrame([shipment.model_dump()]).loc[:, FEATURE_COLUMNS]
        classifier: ProbabilisticClassifier = request.app.state.model
        prediction = int(classifier.predict(features)[0])
        if prediction not in (0, 1):
            raise ValueError(f"Model returned non-binary prediction {prediction}.")
        classes = list(classifier.classes_)
        if 1 not in classes:
            raise ValueError(f"Model classes do not include the late class: {classes}.")
        probabilities = classifier.predict_proba(features)
        late_probability = float(probabilities[0][classes.index(1)])
        request_id = str(uuid4())
        latency_ms = (perf_counter() - started_at) * 1000

        log_prediction(
            inference_log,
            request_id=request_id,
            prediction=prediction,
            late_probability=late_probability,
            latency_ms=latency_ms,
        )
        return PredictionResponse(
            request_id=request_id,
            prediction=prediction,
            delivery_status="late" if prediction == 1 else "on_time",
            late_probability=late_probability,
        )

    return application


app = create_app()
