"""Select a classification threshold using training-set OOF predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline

from train import (
    RANDOM_STATE,
    build_preprocessor,
    load_modeling_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
)

INTERIM_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "interim"
)

BEST_PARAMETERS_PATH = (
    OUTPUT_DIRECTORY
    / "random_forest_best_parameters.json"
)

THRESHOLD_CURVE_PATH = (
    OUTPUT_DIRECTORY
    / "threshold_precision_recall_curve.csv"
)

THRESHOLD_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "threshold_selection_summary.csv"
)

SELECTED_THRESHOLD_PATH = (
    OUTPUT_DIRECTORY
    / "selected_threshold.json"
)

OOF_PREDICTIONS_PATH = (
    INTERIM_DIRECTORY
    / "random_forest_oof_predictions.csv"
)


NUMBER_OF_FOLDS = 5
DEFAULT_THRESHOLD = 0.50


def load_best_parameters() -> dict[str, Any]:
    """Load the tuned Random Forest pipeline parameters."""
    if not BEST_PARAMETERS_PATH.exists():
        raise FileNotFoundError(
            f"Best-parameter file not found: "
            f"{BEST_PARAMETERS_PATH}\n"
            "Run: python src\\tune.py"
        )

    parameters = json.loads(
        BEST_PARAMETERS_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not parameters:
        raise ValueError(
            "The best-parameter file is empty."
        )

    return parameters


def build_tuned_pipeline(
    best_parameters: dict[str, Any],
) -> Pipeline:
    """Build the tuned Random Forest modeling pipeline."""
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )

    pipeline.set_params(**best_parameters)

    return pipeline


def generate_oof_probabilities(
    model_pipeline: Pipeline,
    train_features: pd.DataFrame,
    train_target: pd.Series,
) -> np.ndarray:
    """Generate out-of-fold probabilities for the positive class."""
    cross_validator = StratifiedKFold(
        n_splits=NUMBER_OF_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    probability_matrix = cross_val_predict(
        estimator=model_pipeline,
        X=train_features,
        y=train_target,
        cv=cross_validator,
        method="predict_proba",
        n_jobs=-1,
    )

    if probability_matrix.shape != (
        len(train_features),
        2,
    ):
        raise ValueError(
            "Unexpected OOF probability-matrix shape."
        )

    positive_probabilities = (
        probability_matrix[:, 1]
    )

    if not np.isfinite(
        positive_probabilities
    ).all():
        raise ValueError(
            "Non-finite OOF probabilities were detected."
        )

    if not (
        (positive_probabilities >= 0)
        & (positive_probabilities <= 1)
    ).all():
        raise ValueError(
            "OOF probabilities must be between 0 and 1."
        )

    return positive_probabilities


def calculate_f_beta(
    precision: np.ndarray,
    recall: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Calculate F-beta scores safely."""
    beta_squared = beta**2

    numerator = (
        (1 + beta_squared)
        * precision
        * recall
    )

    denominator = (
        beta_squared * precision
        + recall
    )

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(
            numerator,
            dtype=float,
        ),
        where=denominator != 0,
    )


def build_threshold_curve(
    target: pd.Series,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Calculate threshold-level precision, recall, F1, and F2."""
    (
        precision_values,
        recall_values,
        threshold_values,
    ) = precision_recall_curve(
        target,
        probabilities,
    )

    # precision_recall_curve returns one extra precision/recall
    # point that has no corresponding threshold.
    precision_values = precision_values[:-1]
    recall_values = recall_values[:-1]

    f1_values = calculate_f_beta(
        precision=precision_values,
        recall=recall_values,
        beta=1.0,
    )

    f2_values = calculate_f_beta(
        precision=precision_values,
        recall=recall_values,
        beta=2.0,
    )

    predicted_positive_rates = np.array(
        [
            float(
                np.mean(
                    probabilities >= threshold
                )
            )
            for threshold in threshold_values
        ]
    )

    curve = pd.DataFrame(
        {
            "threshold": threshold_values,
            "precision": precision_values,
            "recall": recall_values,
            "f1": f1_values,
            "f2": f2_values,
            "predicted_positive_rate": (
                predicted_positive_rates
            ),
        }
    )

    curve["predicted_positive_percentage"] = (
        curve["predicted_positive_rate"]
        .mul(100)
    )

    return curve


def select_f1_threshold(
    threshold_curve: pd.DataFrame,
) -> pd.Series:
    """Select the F1-maximizing threshold."""
    ranked_thresholds = (
        threshold_curve
        .sort_values(
            by=[
                "f1",
                "recall",
                "precision",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return ranked_thresholds.iloc[0]


def evaluate_threshold(
    target: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    threshold_name: str,
) -> dict[str, object]:
    """Evaluate one threshold using OOF probabilities."""
    predictions = (
        probabilities >= threshold
    ).astype(int)

    (
        true_negatives,
        false_positives,
        false_negatives,
        true_positives,
    ) = confusion_matrix(
        target,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        target,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        target,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        target,
        predictions,
        zero_division=0,
    )

    specificity = (
        true_negatives
        / (
            true_negatives
            + false_positives
        )
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2

    return {
        "threshold_name": threshold_name,
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "predicted_positive_count": int(
            predictions.sum()
        ),
        "predicted_positive_percentage": float(
            100 * predictions.mean()
        ),
        "true_negatives": int(
            true_negatives
        ),
        "false_positives": int(
            false_positives
        ),
        "false_negatives": int(
            false_negatives
        ),
        "true_positives": int(
            true_positives
        ),
    }


def build_threshold_summary(
    target: pd.Series,
    probabilities: np.ndarray,
    selected_threshold: float,
) -> pd.DataFrame:
    """Compare selected and default thresholds."""
    records = [
        evaluate_threshold(
            target=target,
            probabilities=probabilities,
            threshold=selected_threshold,
            threshold_name="selected_max_f1",
        ),
        evaluate_threshold(
            target=target,
            probabilities=probabilities,
            threshold=DEFAULT_THRESHOLD,
            threshold_name="default_0_50",
        ),
    ]

    summary = pd.DataFrame(records)

    summary["oof_roc_auc"] = (
        roc_auc_score(
            target,
            probabilities,
        )
    )

    summary["oof_average_precision"] = (
        average_precision_score(
            target,
            probabilities,
        )
    )

    return summary


def validate_results(
    threshold_curve: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    selected_threshold: float,
    train_target: pd.Series,
    probabilities: np.ndarray,
) -> None:
    """Validate threshold-selection outputs."""
    if threshold_curve.empty:
        raise ValueError(
            "The threshold curve is empty."
        )

    if not 0 <= selected_threshold <= 1:
        raise ValueError(
            "The selected threshold is outside 0-1."
        )

    probability_metric_columns = [
        "precision",
        "recall",
        "f1",
        "f2",
        "predicted_positive_rate",
    ]

    for metric_column in probability_metric_columns:
        if not threshold_curve[
            metric_column
        ].between(0, 1).all():
            raise ValueError(
                f"Invalid threshold metric: "
                f"{metric_column}"
            )

    if len(threshold_summary) != 2:
        raise ValueError(
            "Expected two threshold-summary rows."
        )

    if set(
        threshold_summary["threshold_name"]
    ) != {
        "selected_max_f1",
        "default_0_50",
    }:
        raise ValueError(
            "Unexpected threshold-summary names."
        )

    selected_f1 = float(
        threshold_summary.loc[
            threshold_summary[
                "threshold_name"
            ] == "selected_max_f1",
            "f1",
        ].iloc[0]
    )

    default_f1 = float(
        threshold_summary.loc[
            threshold_summary[
                "threshold_name"
            ] == "default_0_50",
            "f1",
        ].iloc[0]
    )

    if selected_f1 < default_f1:
        raise ValueError(
            "The selected F1 threshold underperformed "
            "the default threshold."
        )

    if len(probabilities) != len(train_target):
        raise ValueError(
            "OOF probabilities and targets are misaligned."
        )

    if int(train_target.sum()) != 4_231:
        raise ValueError(
            "Unexpected training subscriber count."
        )


def save_outputs(
    threshold_curve: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    selected_threshold: float,
    selected_row: pd.Series,
    train_target: pd.Series,
    probabilities: np.ndarray,
) -> None:
    """Save threshold-selection reports and OOF predictions."""
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    INTERIM_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_curve.to_csv(
        THRESHOLD_CURVE_PATH,
        index=False,
    )

    threshold_summary.to_csv(
        THRESHOLD_SUMMARY_PATH,
        index=False,
    )

    selected_payload = {
        "selection_rule": "maximize_oof_f1",
        "threshold": float(
            selected_threshold
        ),
        "oof_precision": float(
            selected_row["precision"]
        ),
        "oof_recall": float(
            selected_row["recall"]
        ),
        "oof_f1": float(
            selected_row["f1"]
        ),
        "oof_f2": float(
            selected_row["f2"]
        ),
        "predicted_positive_percentage": float(
            selected_row[
                "predicted_positive_percentage"
            ]
        ),
        "number_of_folds": NUMBER_OF_FOLDS,
        "test_set_used": False,
    }

    SELECTED_THRESHOLD_PATH.write_text(
        json.dumps(
            selected_payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    oof_predictions = pd.DataFrame(
        {
            "actual_subscribed": (
                train_target.to_numpy()
            ),
            "oof_probability": probabilities,
            "selected_threshold_prediction": (
                probabilities
                >= selected_threshold
            ).astype(int),
            "default_threshold_prediction": (
                probabilities
                >= DEFAULT_THRESHOLD
            ).astype(int),
        }
    )

    oof_predictions.to_csv(
        OOF_PREDICTIONS_PATH,
        index=False,
    )


def display_results(
    threshold_summary: pd.DataFrame,
    selected_threshold: float,
) -> None:
    """Display threshold-selection results."""
    print("\nThreshold comparison:")

    display_columns = [
        "threshold_name",
        "threshold",
        "precision",
        "recall",
        "f1",
        "specificity",
        "balanced_accuracy",
        "predicted_positive_percentage",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
        "oof_roc_auc",
        "oof_average_precision",
    ]

    print(
        threshold_summary[
            display_columns
        ].to_string(index=False)
    )

    print(
        "\nSelected classification threshold:",
        round(selected_threshold, 6),
    )


def main() -> None:
    """Select a threshold using OOF training probabilities."""
    print("=" * 70)
    print("BANK MARKETING — THRESHOLD SELECTION")
    print("=" * 70)

    (
        train_features,
        train_target,
        test_features,
        test_target,
    ) = load_modeling_data()

    print(f"\nTraining feature shape: {train_features.shape}")
    print(f"Test feature shape: {test_features.shape}")

    print(
        "\nThe test set is loaded only for schema validation."
    )

    print(
        "No test predictions or metrics are calculated "
        "during threshold selection."
    )

    best_parameters = load_best_parameters()

    tuned_pipeline = build_tuned_pipeline(
        best_parameters
    )

    print(
        f"\nGenerating {NUMBER_OF_FOLDS}-fold "
        "out-of-fold training probabilities..."
    )

    oof_probabilities = (
        generate_oof_probabilities(
            model_pipeline=tuned_pipeline,
            train_features=train_features,
            train_target=train_target,
        )
    )

    threshold_curve = build_threshold_curve(
        target=train_target,
        probabilities=oof_probabilities,
    )

    selected_row = select_f1_threshold(
        threshold_curve
    )

    selected_threshold = float(
        selected_row["threshold"]
    )

    threshold_summary = build_threshold_summary(
        target=train_target,
        probabilities=oof_probabilities,
        selected_threshold=selected_threshold,
    )

    validate_results(
        threshold_curve=threshold_curve,
        threshold_summary=threshold_summary,
        selected_threshold=selected_threshold,
        train_target=train_target,
        probabilities=oof_probabilities,
    )

    save_outputs(
        threshold_curve=threshold_curve,
        threshold_summary=threshold_summary,
        selected_threshold=selected_threshold,
        selected_row=selected_row,
        train_target=train_target,
        probabilities=oof_probabilities,
    )

    display_results(
        threshold_summary=threshold_summary,
        selected_threshold=selected_threshold,
    )

    print("\nGenerated threshold-selection files:")
    print(f"- {THRESHOLD_CURVE_PATH}")
    print(f"- {THRESHOLD_SUMMARY_PATH}")
    print(f"- {SELECTED_THRESHOLD_PATH}")
    print(f"- {OOF_PREDICTIONS_PATH}")

    print(
        "\nThreshold selection completed successfully "
        "using training data only."
    )


if __name__ == "__main__":
    main()
