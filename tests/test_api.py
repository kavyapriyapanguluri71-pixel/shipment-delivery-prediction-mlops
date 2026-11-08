import json

import numpy as np
from fastapi.testclient import TestClient

from api.main import create_app


class FixedClassifier:
    classes_ = np.array([0, 1])

    def predict(self, features):
        assert len(features) == 1
        return np.array([1])

    def predict_proba(self, features):
        return np.array([[0.2, 0.8]])


def valid_request() -> dict[str, object]:
    return {
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


def test_health_and_prediction_log_only_metadata(tmp_path) -> None:
    log_path = tmp_path / "logs" / "inference.jsonl"
    app = create_app(model=FixedClassifier(), log_path=log_path)

    with TestClient(app) as client:
        health = client.get("/health")
        response = client.post("/predict", json=valid_request())

    assert health.json() == {"status": "ok", "model_loaded": True}
    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["delivery_status"] == "late"
    assert response.json()["late_probability"] == 0.8
    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["prediction"] == 1
    assert "Weight_in_gms" not in record
    assert record["latency_ms"] >= 0


def test_api_rejects_unknown_input_fields(tmp_path) -> None:
    app = create_app(model=FixedClassifier(), log_path=tmp_path / "log.jsonl")
    request = valid_request() | {"ID": 23}

    with TestClient(app) as client:
        response = client.post("/predict", json=request)

    assert response.status_code == 422


def test_api_rejects_missing_and_out_of_range_features(tmp_path) -> None:
    app = create_app(model=FixedClassifier(), log_path=tmp_path / "log.jsonl")
    missing_feature = valid_request()
    del missing_feature["Weight_in_gms"]
    invalid_rating = valid_request() | {"Customer_rating": 6}

    with TestClient(app) as client:
        missing_response = client.post("/predict", json=missing_feature)
        invalid_response = client.post("/predict", json=invalid_rating)

    assert missing_response.status_code == 422
    assert invalid_response.status_code == 422
