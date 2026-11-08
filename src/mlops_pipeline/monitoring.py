"""Compare a new shipment batch with a reference batch for feature drift."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_pipeline.schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)

_EPSILON = 1e-6


def _drift_status(psi: float) -> str:
    if psi >= 0.2:
        return "high"
    if psi >= 0.1:
        return "moderate"
    return "low"


def _load_features(path: str | Path, label: str) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"{label} data file does not exist: {data_path}")
    frame = pd.read_csv(data_path)
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} data is missing feature columns: {missing}")
    frame = frame.loc[:, list(FEATURE_COLUMNS)].copy()
    if frame.empty:
        raise ValueError(f"{label} data contains no rows.")
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        observed = frame[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(observed).all():
            raise ValueError(f"Feature {column!r} contains infinite values.")
    return frame


def _distribution_index(reference: pd.Series, current: pd.Series) -> float:
    """Calculate PSI for categorical values with a small zero-count smoothing."""
    reference_values = reference.fillna("<MISSING>").astype(str)
    current_values = current.fillna("<MISSING>").astype(str)
    categories = sorted(set(reference_values) | set(current_values))
    reference_share = reference_values.value_counts(normalize=True).reindex(
        categories, fill_value=0.0
    )
    current_share = current_values.value_counts(normalize=True).reindex(
        categories, fill_value=0.0
    )
    ref = np.clip(reference_share.to_numpy(dtype=float), _EPSILON, None)
    cur = np.clip(current_share.to_numpy(dtype=float), _EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _numeric_index(reference: pd.Series, current: pd.Series) -> float:
    """Calculate PSI using quantile bins defined only by the reference data."""
    observed = reference.dropna().to_numpy(dtype=float)
    if observed.size == 0:
        # Keep a stable missing-only reference bin for an entirely empty field.
        return _distribution_index(
            reference.map(lambda value: "<MISSING>" if pd.isna(value) else "<OBSERVED>"),
            current.map(lambda value: "<MISSING>" if pd.isna(value) else "<OBSERVED>"),
        )
    edges = np.unique(np.quantile(observed, np.linspace(0, 1, 11)))
    if len(edges) == 1:
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
    ref_bins = pd.cut(reference, bins=edges, labels=list(range(len(edges) - 1)))
    cur_bins = pd.cut(current, bins=edges, labels=list(range(len(edges) - 1)))
    ref_values = ref_bins.astype(object).where(reference.notna(), "<MISSING>").astype(str)
    cur_values = cur_bins.astype(object).where(current.notna(), "<MISSING>").astype(str)
    return _distribution_index(ref_values, cur_values)


def compare_batches(
    reference_path: str | Path,
    current_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Compare feature distributions and optionally save a JSON report."""
    reference = _load_features(reference_path, "Reference")
    current = _load_features(current_path, "Current")
    rows: list[dict[str, object]] = []
    for feature in FEATURE_COLUMNS:
        if feature in NUMERIC_FEATURES:
            psi = _numeric_index(reference[feature], current[feature])
        elif feature in CATEGORICAL_FEATURES:
            psi = _distribution_index(reference[feature], current[feature])
        else:  # Guard against schema additions without an explicit monitor policy.
            raise ValueError(f"No drift calculation is configured for {feature!r}.")
        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "drift_status": _drift_status(psi),
                "reference_missing_rate": float(reference[feature].isna().mean()),
                "current_missing_rate": float(current[feature].isna().mean()),
            }
        )
    report: dict[str, object] = {
        "reference_file": Path(reference_path).name,
        "current_file": Path(current_path).name,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "psi_thresholds": {"low_below": 0.1, "moderate_below": 0.2},
        "features": rows,
        "high_drift_features": [row["feature"] for row in rows if row["drift_status"] == "high"],
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Reference shipment CSV.")
    parser.add_argument("--current", required=True, help="New shipment batch CSV.")
    parser.add_argument("--output", default="outputs/drift_report.json", help="JSON report path.")
    args = parser.parse_args()
    print(json.dumps(compare_batches(args.reference, args.current, args.output), indent=2))


if __name__ == "__main__":
    main()
