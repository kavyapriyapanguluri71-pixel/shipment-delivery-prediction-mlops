from pathlib import Path

import pandas as pd

from mlops_pipeline.schema import FEATURE_COLUMNS, TARGET_COLUMN
from mlops_pipeline.train import build_pipeline, candidate_models
from mlops_pipeline.validation import load_training_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_training_data_schema_and_labels() -> None:
    frame = load_training_data(PROJECT_ROOT / "Train.csv")

    assert len(frame) == 10_999
    assert set(frame[TARGET_COLUMN].unique()) == {0, 1}
    assert "ID" not in FEATURE_COLUMNS
    assert frame.loc[:, list(FEATURE_COLUMNS)].notna().all().all()


def test_pipeline_fits_and_predicts_without_id() -> None:
    frame = load_training_data(PROJECT_ROOT / "Train.csv").sample(
        n=250, random_state=7
    )
    model = build_pipeline(candidate_models()["LogisticRegression"])

    model_features = frame.loc[:, list(FEATURE_COLUMNS)]
    model.fit(model_features, frame[TARGET_COLUMN])
    predictions = model.predict(model_features)
    probabilities = model.predict_proba(model_features)

    assert len(predictions) == len(frame)
    assert probabilities.shape == (len(frame), 2)
