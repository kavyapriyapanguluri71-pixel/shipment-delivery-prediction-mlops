"""Loading and validating the serialized scikit-learn model."""

import os
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


def load_model(model_path: str | Path | None = None) -> Pipeline:
    configured_path = model_path or os.environ.get(
        "MODEL_PATH", "models/shipments_pipeline.joblib"
    )
    path = Path(configured_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run the training command first "
            "or set MODEL_PATH to a valid artifact."
        )
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected a scikit-learn Pipeline artifact at {path}.")
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Model artifact at {path} does not support probabilities.")
    return model
