"""Reproducible model training, comparison, tuning, and artifact generation."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from mlops_pipeline.schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from mlops_pipeline.validation import load_training_data

RANDOM_STATE = 42
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing that is fitted only inside each training fold."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )


def candidate_models() -> dict[str, object]:
    """Return all baseline classifiers with imbalance handling where supported."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "DecisionTree": DecisionTreeClassifier(
            class_weight="balanced",
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }


def parameter_grids() -> dict[str, dict[str, list[object]]]:
    """Keep the tuning search bounded while covering each model's main controls."""
    return {
        "LogisticRegression": {
            "classifier__C": [0.1, 1.0, 10.0],
        },
        "DecisionTree": {
            "classifier__max_depth": [None, 8, 16],
            "classifier__min_samples_leaf": [1, 5],
        },
        "RandomForest": {
            "classifier__n_estimators": [150, 300],
            "classifier__max_depth": [None, 12],
            "classifier__min_samples_leaf": [1, 3],
        },
        "GradientBoosting": {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [2, 3],
        },
    }


def build_pipeline(classifier: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def compare_models(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare candidates on stratified CV, using only the training partition."""
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    rows: list[dict[str, object]] = []
    models = candidate_models()
    for name, classifier in models.items():
        result = cross_validate(
            build_pipeline(classifier),
            features,
            target,
            scoring=SCORING,
            cv=cross_validation,
            n_jobs=1,
            error_score="raise",
        )
        row: dict[str, object] = {"model": name}
        for metric in SCORING:
            values = result[f"test_{metric}"]
            row[f"mean_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = float(values.std())
        rows.append(row)

    results = pd.DataFrame(rows).sort_values(
        "mean_roc_auc", ascending=False, ignore_index=True
    )
    return results, models


def _feature_importance(model: Pipeline) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = abs(classifier.coef_[0])
    else:
        raise TypeError(
            f"Feature importance is unavailable for {type(classifier).__name__}."
        )
    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False, ignore_index=True)
    )


def _save_feature_importance(importance: pd.DataFrame, output_dir: Path) -> None:
    top_features = importance.head(15).sort_values("importance", ascending=True)
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(top_features["feature"], top_features["importance"])
    axis.set_xlabel("Importance")
    axis.set_title("Top Shipment Delivery Features")
    figure.tight_layout()
    figure.savefig(output_dir / "feature_importance.png", dpi=140)
    plt.close(figure)


def _save_plots(
    target: pd.Series,
    predictions: pd.Series,
    probabilities: pd.Series,
    output_dir: Path,
) -> None:
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["On time (0)", "Late (1)"],
    ).plot(cmap="Blues", values_format="d")
    plt.title("Shipment Delivery Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=140)
    plt.close()

    false_positive_rate, true_positive_rate, _ = roc_curve(target, probabilities)
    auc = roc_auc_score(target, probabilities)
    plt.figure(figsize=(7, 5))
    plt.plot(false_positive_rate, true_positive_rate, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate (late shipments)")
    plt.title("Shipment Late-Delivery ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=140)
    plt.close()


def train(
    data_path: str | Path,
    output_dir: str | Path = "outputs",
    model_dir: str | Path = "models",
) -> dict[str, object]:
    """Run the full training workflow and write model and evaluation artifacts."""
    source_path = Path(data_path)
    output_path = Path(output_dir)
    model_path = Path(model_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)
    frame = load_training_data(source_path)
    features = frame.loc[:, list(FEATURE_COLUMNS)].copy()
    target = frame[TARGET_COLUMN].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    comparison, models = compare_models(x_train, y_train)
    comparison.to_csv(output_path / "model_comparison.csv", index=False)
    best_name = str(comparison.iloc[0]["model"])
    search = GridSearchCV(
        estimator=build_pipeline(models[best_name]),
        param_grid=parameter_grids()[best_name],
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    final_model = search.best_estimator_

    predictions = final_model.predict(x_test)
    probabilities = final_model.predict_proba(x_test)[:, 1]
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "positive_class": 1,
        "positive_class_meaning": "late shipment",
        "selected_model": best_name,
        "best_parameters": search.best_params_,
        "test_rows": int(len(y_test)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["on_time", "late"],
            output_dict=True,
            zero_division=0,
        ),
    }

    (output_path / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _save_plots(
        y_test,
        pd.Series(predictions, index=y_test.index),
        pd.Series(probabilities, index=y_test.index),
        output_path,
    )

    importance = _feature_importance(final_model)
    importance.to_csv(output_path / "feature_importance.csv", index=False)
    _save_feature_importance(importance, output_path)
    joblib.dump(final_model, model_path / "shipments_pipeline.joblib")

    metadata = {
        "model": best_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": source_path.name,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "target_column": TARGET_COLUMN,
        "positive_class": 1,
        "positive_class_meaning": "late shipment",
        "feature_columns": list(FEATURE_COLUMNS),
        "best_parameters": search.best_params_,
        "test_metrics": {
            name: metrics[name]
            for name in ("accuracy", "precision", "recall", "f1", "roc_auc")
        },
    }
    (model_path / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Train.csv", help="Labeled training CSV.")
    parser.add_argument("--output", default="outputs", help="Evaluation output directory.")
    parser.add_argument("--model-dir", default="models", help="Model artifact directory.")
    arguments = parser.parse_args()
    results = train(arguments.data, arguments.output, arguments.model_dir)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
