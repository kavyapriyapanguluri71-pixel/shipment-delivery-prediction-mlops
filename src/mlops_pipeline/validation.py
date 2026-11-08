"""Input validation shared by the training command."""

from pathlib import Path

import pandas as pd

from mlops_pipeline.schema import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    REQUIRED_TRAINING_COLUMNS,
    TARGET_COLUMN,
)


def load_training_data(path: str | Path) -> pd.DataFrame:
    """Load a labeled CSV and fail with an actionable message if it is invalid."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Training data file does not exist: {data_path}")

    frame = pd.read_csv(data_path)
    missing_columns = sorted(set(REQUIRED_TRAINING_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Training data is missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Training data contains no rows.")

    model_data = frame.loc[:, [*FEATURE_COLUMNS, TARGET_COLUMN]]
    missing_values = model_data.isna().sum()
    columns_with_missing_values = missing_values[missing_values > 0].to_dict()
    if columns_with_missing_values:
        raise ValueError(
            f"Training data contains missing feature/target values: "
            f"{columns_with_missing_values}"
        )

    for column in NUMERIC_FEATURES:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"Feature {column!r} must contain numeric values.") from error

    target_values = set(frame[TARGET_COLUMN].unique())
    if target_values != {0, 1}:
        raise ValueError(
            f"Target {TARGET_COLUMN!r} must contain both binary labels 0 and 1; "
            f"found {sorted(target_values, key=str)}."
        )
    return frame
