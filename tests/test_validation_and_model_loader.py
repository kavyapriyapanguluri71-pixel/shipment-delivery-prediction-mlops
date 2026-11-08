from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from api.model_loader import load_model
from mlops_pipeline.schema import FEATURE_COLUMNS, REQUIRED_TRAINING_COLUMNS
from mlops_pipeline.validation import load_training_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_model_reads_valid_pipeline_artifact(tmp_path) -> None:
    artifact_path = tmp_path / "model.joblib"
    model = Pipeline([("classifier", LogisticRegression())])
    model.fit([[0], [1]], [0, 1])
    joblib.dump(model, artifact_path)

    loaded = load_model(artifact_path)

    assert isinstance(loaded, Pipeline)
    assert loaded.predict_proba([[0]]).shape == (1, 2)


def test_load_model_rejects_missing_or_invalid_artifact(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Run the training command"):
        load_model(tmp_path / "missing.joblib")

    invalid_path = tmp_path / "not-a-pipeline.joblib"
    joblib.dump({"model": "invalid"}, invalid_path)
    with pytest.raises(TypeError, match="Expected a scikit-learn Pipeline"):
        load_model(invalid_path)


def test_load_training_data_rejects_missing_columns(tmp_path) -> None:
    source = pd.read_csv(PROJECT_ROOT / "Train.csv").head(5)
    missing_path = tmp_path / "missing-columns.csv"
    source.drop(columns=["Customer_rating"]).to_csv(missing_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_training_data(missing_path)


def test_load_training_data_rejects_non_numeric_feature_and_bad_labels(tmp_path) -> None:
    source = pd.read_csv(PROJECT_ROOT / "Train.csv").head(10)
    source["Cost_of_the_Product"] = "not-numeric"
    invalid_feature_path = tmp_path / "invalid-feature.csv"
    source.to_csv(invalid_feature_path, index=False)
    with pytest.raises(ValueError, match="must contain numeric values"):
        load_training_data(invalid_feature_path)

    source = pd.read_csv(PROJECT_ROOT / "Train.csv").head(10)
    source["Reached.on.Time_Y.N"] = 2
    invalid_label_path = tmp_path / "invalid-label.csv"
    source.to_csv(invalid_label_path, index=False)
    with pytest.raises(ValueError, match="both binary labels"):
        load_training_data(invalid_label_path)


def test_training_input_schema_contains_all_shipment_fields() -> None:
    frame = load_training_data(PROJECT_ROOT / "Train.csv")

    assert set(REQUIRED_TRAINING_COLUMNS).issubset(frame.columns)
    assert len(FEATURE_COLUMNS) == 10
