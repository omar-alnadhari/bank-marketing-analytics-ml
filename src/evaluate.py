"""Train and evaluate the final model on the held-out test set."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from train import (
    MODEL_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    build_preprocessor,
    load_modeling_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELING_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
)

INTERIM_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "interim"
)

MODEL_DIRECTORY = PROJECT_ROOT / "models"

BEST_PARAMETERS_PATH = (
    MODELING_REPORT_DIRECTORY
    / "random_forest_best_parameters.json"
)

SELECTED_THRESHOLD_PATH = (
    MODELING_REPORT_DIRECTORY
    / "selected_threshold.json"
)

PROBABILITY_METRICS_PATH = (
    MODELING_REPORT_DIRECTORY
    / "final_test_probability_metrics.csv"
)

THRESHOLD_METRICS_PATH = (
    MODELING_REPORT_DIRECTORY
    / "final_test_threshold_metrics.csv"
)

CONFUSION_MATRICES_PATH = (
    MODELING_REPORT_DIRECTORY
    / "final_test_confusion_matrices.csv"
)

CLASSIFICATION_REPORT_PATH = (
    MODELING_REPORT_DIRECTORY
    / "final_test_classification_report.csv"
)

ROC_CURVE_PATH = (
    MODELING_REPORT_DIRECTORY
    / "final_test_roc_curve.csv"
)

PRECISION_RECALL_CURVE_PATH = (
    MODELING_REPORT_DIRECTORY
    / "final_test_precision_recall_curve.csv"
)

PERMUTATION_IMPORTANCE_PATH = (
    MODELING_REPORT_DIRECTORY
    / "test_permutation_importance.csv"
)

TARGETING_LIFT_PATH = (
    MODELING_REPORT_DIRECTORY
    / "final_targeting_lift_table.csv"
)

MODEL_METADATA_PATH = (
    MODELING_REPORT_DIRECTORY
    / "final_model_metadata.json"
)

TEST_PREDICTIONS_PATH = (
    INTERIM_DIRECTORY
    / "final_test_predictions.csv"
)

FINAL_MODEL_PATH = (
    MODEL_DIRECTORY
    / "bank_marketing_pre_campaign_random_forest.joblib"
)


DEFAULT_THRESHOLD = 0.50

PERMUTATION_REPEATS = 10

TARGETING_FRACTIONS = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
    1.00,
]


def load_json_file(
    path: Path,
    description: str,
) -> dict[str, Any]:
    """Load and validate a required JSON file."""
    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found: {path}"
        )

    content = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(content, dict) or not content:
        raise ValueError(
            f"{description} is empty or invalid."
        )

    return content


def load_model_configuration() -> tuple[
    dict[str, Any],
    dict[str, Any],
    float,
]:
    """Load tuned parameters and selected threshold."""
    best_parameters = load_json_file(
        BEST_PARAMETERS_PATH,
        "Best Random Forest parameter file",
    )

    threshold_payload = load_json_file(
        SELECTED_THRESHOLD_PATH,
        "Selected-threshold file",
    )

    if "threshold" not in threshold_payload:
        raise ValueError(
            "The selected-threshold file contains "
            "no threshold value."
        )

    selected_threshold = float(
        threshold_payload["threshold"]
    )

    if not 0 <= selected_threshold <= 1:
        raise ValueError(
            "The selected threshold must be between 0 and 1."
        )

    if threshold_payload.get("test_set_used") is not False:
        raise ValueError(
            "The threshold-selection metadata does not confirm "
            "that the test set remained unused."
        )

    return (
        best_parameters,
        threshold_payload,
        selected_threshold,
    )


def build_final_pipeline(
    best_parameters: dict[str, Any],
) -> Pipeline:
    """Build the tuned final modeling pipeline."""
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
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.set_params(**best_parameters)

    return pipeline


def calculate_probability_metrics(
    test_target: pd.Series,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Calculate threshold-independent probability metrics."""
    prevalence = float(test_target.mean())

    metrics = {
        "test_records": len(test_target),
        "test_subscribers": int(test_target.sum()),
        "positive_class_prevalence": prevalence,
        "roc_auc": float(
            roc_auc_score(
                test_target,
                probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                test_target,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                test_target,
                probabilities,
            )
        ),
        "log_loss": float(
            log_loss(
                test_target,
                probabilities,
                labels=[0, 1],
            )
        ),
    }

    return pd.DataFrame([metrics])


def calculate_threshold_metrics(
    test_target: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    threshold_name: str,
) -> dict[str, object]:
    """Calculate classification metrics at one threshold."""
    predictions = (
        probabilities >= threshold
    ).astype(int)

    (
        true_negatives,
        false_positives,
        false_negatives,
        true_positives,
    ) = confusion_matrix(
        test_target,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        test_target,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        test_target,
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

    negative_predictive_value = (
        true_negatives
        / (
            true_negatives
            + false_negatives
        )
        if (
            true_negatives
            + false_negatives
        ) > 0
        else 0.0
    )

    prevalence = float(
        test_target.mean()
    )

    lift = (
        precision / prevalence
        if prevalence > 0
        else 0.0
    )

    return {
        "threshold_name": threshold_name,
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(
                test_target,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                test_target,
                predictions,
            )
        ),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "negative_predictive_value": float(
            negative_predictive_value
        ),
        "f1": float(
            f1_score(
                test_target,
                predictions,
                zero_division=0,
            )
        ),
        "f2": float(
            fbeta_score(
                test_target,
                predictions,
                beta=2,
                zero_division=0,
            )
        ),
        "lift_over_baseline": float(lift),
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


def build_threshold_reports(
    test_target: pd.Series,
    probabilities: np.ndarray,
    selected_threshold: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build selected/default threshold reports."""
    threshold_records = [
        calculate_threshold_metrics(
            test_target=test_target,
            probabilities=probabilities,
            threshold=selected_threshold,
            threshold_name="selected_training_oof_f1",
        ),
        calculate_threshold_metrics(
            test_target=test_target,
            probabilities=probabilities,
            threshold=DEFAULT_THRESHOLD,
            threshold_name="default_0_50",
        ),
    ]

    threshold_metrics = pd.DataFrame(
        threshold_records
    )

    confusion_matrices = threshold_metrics[
        [
            "threshold_name",
            "threshold",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "true_positives",
        ]
    ].copy()

    classification_reports: list[pd.DataFrame] = []

    for threshold_name, threshold in [
        (
            "selected_training_oof_f1",
            selected_threshold,
        ),
        (
            "default_0_50",
            DEFAULT_THRESHOLD,
        ),
    ]:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        report = classification_report(
            test_target,
            predictions,
            labels=[0, 1],
            target_names=[
                "non_subscriber",
                "subscriber",
            ],
            output_dict=True,
            zero_division=0,
        )

        report_frame = (
            pd.DataFrame(report)
            .transpose()
            .reset_index()
            .rename(
                columns={
                    "index": "class_or_average",
                }
            )
        )

        report_frame.insert(
            0,
            "threshold_name",
            threshold_name,
        )

        report_frame.insert(
            1,
            "threshold",
            threshold,
        )

        classification_reports.append(
            report_frame
        )

    classification_report_frame = pd.concat(
        classification_reports,
        ignore_index=True,
    )

    return (
        threshold_metrics,
        confusion_matrices,
        classification_report_frame,
    )


def build_probability_curves(
    test_target: pd.Series,
    probabilities: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ROC and precision-recall curve tables."""
    (
        false_positive_rate,
        true_positive_rate,
        roc_thresholds,
    ) = roc_curve(
        test_target,
        probabilities,
    )

    roc_curve_data = pd.DataFrame(
        {
            "false_positive_rate": (
                false_positive_rate
            ),
            "true_positive_rate": (
                true_positive_rate
            ),
            "threshold": roc_thresholds,
        }
    )

    (
        precision_values,
        recall_values,
        pr_thresholds,
    ) = precision_recall_curve(
        test_target,
        probabilities,
    )

    padded_thresholds = np.append(
        pr_thresholds,
        np.nan,
    )

    precision_recall_data = pd.DataFrame(
        {
            "precision": precision_values,
            "recall": recall_values,
            "threshold": padded_thresholds,
        }
    )

    return (
        roc_curve_data,
        precision_recall_data,
    )


def build_targeting_lift_table(
    test_target: pd.Series,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Evaluate campaign value when targeting top-ranked clients."""
    ranking_data = pd.DataFrame(
        {
            "actual_subscribed": (
                test_target.to_numpy()
            ),
            "probability": probabilities,
        }
    )

    ranking_data = (
        ranking_data
        .sort_values(
            "probability",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    total_clients = len(ranking_data)

    total_subscribers = int(
        ranking_data[
            "actual_subscribed"
        ].sum()
    )

    baseline_rate = (
        total_subscribers
        / total_clients
    )

    result_records: list[dict[str, object]] = []

    for targeting_fraction in TARGETING_FRACTIONS:
        targeted_clients = int(
            np.ceil(
                total_clients
                * targeting_fraction
            )
        )

        targeted_segment = (
            ranking_data.iloc[
                :targeted_clients
            ]
        )

        subscribers_captured = int(
            targeted_segment[
                "actual_subscribed"
            ].sum()
        )

        targeting_precision = (
            subscribers_captured
            / targeted_clients
        )

        capture_rate = (
            subscribers_captured
            / total_subscribers
        )

        lift = (
            targeting_precision
            / baseline_rate
        )

        expected_random_subscribers = (
            targeted_clients
            * baseline_rate
        )

        result_records.append(
            {
                "targeting_fraction": (
                    targeting_fraction
                ),
                "targeting_percentage": (
                    targeting_fraction * 100
                ),
                "targeted_clients": (
                    targeted_clients
                ),
                "subscribers_captured": (
                    subscribers_captured
                ),
                "subscriber_capture_rate": (
                    capture_rate
                ),
                "subscriber_capture_percentage": (
                    capture_rate * 100
                ),
                "targeting_precision": (
                    targeting_precision
                ),
                "targeting_precision_percentage": (
                    targeting_precision * 100
                ),
                "baseline_subscription_rate": (
                    baseline_rate
                ),
                "lift_over_random_targeting": (
                    lift
                ),
                "expected_subscribers_under_random_targeting": (
                    expected_random_subscribers
                ),
                "additional_subscribers_vs_random_expectation": (
                    subscribers_captured
                    - expected_random_subscribers
                ),
            }
        )

    return pd.DataFrame(result_records)


def calculate_permutation_importance(
    final_pipeline: Pipeline,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> pd.DataFrame:
    """Calculate held-out permutation importance using AP."""
    results = permutation_importance(
        estimator=final_pipeline,
        X=test_features,
        y=test_target,
        scoring="average_precision",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_data = pd.DataFrame(
        {
            "feature": test_features.columns,
            "importance_mean": (
                results.importances_mean
            ),
            "importance_std": (
                results.importances_std
            ),
        }
    )

    importance_data["absolute_importance_mean"] = (
        importance_data[
            "importance_mean"
        ].abs()
    )

    importance_data = (
        importance_data
        .sort_values(
            "importance_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_data["importance_rank"] = (
        np.arange(
            1,
            len(importance_data) + 1,
        )
    )

    return importance_data


def build_test_predictions(
    test_target: pd.Series,
    probabilities: np.ndarray,
    selected_threshold: float,
) -> pd.DataFrame:
    """Build the final row-level test prediction report."""
    return pd.DataFrame(
        {
            "actual_subscribed": (
                test_target.to_numpy()
            ),
            "predicted_probability": (
                probabilities
            ),
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


def calculate_file_sha256(
    path: Path,
) -> str:
    """Calculate a SHA-256 checksum for a file."""
    sha256_hash = hashlib.sha256()

    with path.open("rb") as file_handle:
        for block in iter(
            lambda: file_handle.read(1_048_576),
            b"",
        ):
            sha256_hash.update(block)

    return sha256_hash.hexdigest()


def save_model_artifact(
    final_pipeline: Pipeline,
    selected_threshold: float,
) -> dict[str, object]:
    """Persist the fitted pipeline and prediction contract."""
    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "pipeline": final_pipeline,
        "threshold": selected_threshold,
        "model_features": MODEL_FEATURES,
        "target_column": TARGET_COLUMN,
        "positive_class": 1,
        "prediction_contract": (
            "Pre-campaign client prioritization "
            "before current-campaign contact begins."
        ),
        "excluded_current_campaign_features": [
            "contact",
            "day",
            "month",
            "duration",
            "campaign",
        ],
    }

    joblib.dump(
        artifact,
        FINAL_MODEL_PATH,
        compress=3,
    )

    reloaded_artifact = joblib.load(
        FINAL_MODEL_PATH
    )

    if "pipeline" not in reloaded_artifact:
        raise ValueError(
            "The reloaded model artifact contains no pipeline."
        )

    if float(
        reloaded_artifact["threshold"]
    ) != float(selected_threshold):
        raise ValueError(
            "The reloaded threshold does not match."
        )

    return artifact


def validate_model_reload(
    final_pipeline: Pipeline,
    test_features: pd.DataFrame,
) -> None:
    """Validate that the saved model reproduces predictions."""
    saved_artifact = joblib.load(
        FINAL_MODEL_PATH
    )

    reloaded_pipeline = (
        saved_artifact["pipeline"]
    )

    sample_features = test_features.head(50)

    original_probabilities = (
        final_pipeline.predict_proba(
            sample_features
        )[:, 1]
    )

    reloaded_probabilities = (
        reloaded_pipeline.predict_proba(
            sample_features
        )[:, 1]
    )

    if not np.allclose(
        original_probabilities,
        reloaded_probabilities,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Reloaded model predictions do not match "
            "the original fitted model."
        )


def validate_results(
    probability_metrics: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    confusion_matrices: pd.DataFrame,
    roc_curve_data: pd.DataFrame,
    precision_recall_data: pd.DataFrame,
    targeting_lift: pd.DataFrame,
    permutation_importance_data: pd.DataFrame,
    test_target: pd.Series,
    probabilities: np.ndarray,
) -> None:
    """Validate all final evaluation outputs."""
    if len(test_target) != 9_043:
        raise ValueError(
            "Unexpected final test-set size."
        )

    if int(test_target.sum()) != 1_058:
        raise ValueError(
            "Unexpected test subscriber count."
        )

    if len(probabilities) != len(test_target):
        raise ValueError(
            "Test probabilities and targets are misaligned."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Non-finite test probabilities were detected."
        )

    if not (
        (probabilities >= 0)
        & (probabilities <= 1)
    ).all():
        raise ValueError(
            "Test probabilities must be between 0 and 1."
        )

    probability_columns = [
        "positive_class_prevalence",
        "roc_auc",
        "average_precision",
        "brier_score",
    ]

    for column in probability_columns:
        if not probability_metrics[
            column
        ].between(0, 1).all():
            raise ValueError(
                f"Invalid probability metric: {column}"
            )

    threshold_metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "negative_predictive_value",
        "f1",
        "f2",
    ]

    for column in threshold_metric_columns:
        if not threshold_metrics[
            column
        ].between(0, 1).all():
            raise ValueError(
                f"Invalid threshold metric: {column}"
            )

    if len(threshold_metrics) != 2:
        raise ValueError(
            "Expected selected and default threshold results."
        )

    if len(confusion_matrices) != 2:
        raise ValueError(
            "Expected two confusion matrices."
        )

    confusion_totals = (
        confusion_matrices[
            [
                "true_negatives",
                "false_positives",
                "false_negatives",
                "true_positives",
            ]
        ]
        .sum(axis=1)
    )

    if not (
        confusion_totals
        == len(test_target)
    ).all():
        raise ValueError(
            "A confusion matrix does not cover all test rows."
        )

    if roc_curve_data.empty:
        raise ValueError(
            "The ROC curve is empty."
        )

    if precision_recall_data.empty:
        raise ValueError(
            "The precision-recall curve is empty."
        )

    if len(targeting_lift) != len(
        TARGETING_FRACTIONS
    ):
        raise ValueError(
            "Unexpected targeting-lift table length."
        )

    final_lift = targeting_lift.loc[
        targeting_lift[
            "targeting_fraction"
        ] == 1.0,
        "lift_over_random_targeting",
    ].iloc[0]

    if not np.isclose(
        final_lift,
        1.0,
    ):
        raise ValueError(
            "Targeting 100% of clients should have lift 1."
        )

    if len(permutation_importance_data) != len(
        MODEL_FEATURES
    ):
        raise ValueError(
            "Unexpected permutation-importance length."
        )


def save_reports(
    probability_metrics: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    confusion_matrices: pd.DataFrame,
    classification_report_data: pd.DataFrame,
    roc_curve_data: pd.DataFrame,
    precision_recall_data: pd.DataFrame,
    targeting_lift: pd.DataFrame,
    permutation_importance_data: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> None:
    """Save final evaluation reports."""
    MODELING_REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    INTERIM_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    probability_metrics.to_csv(
        PROBABILITY_METRICS_PATH,
        index=False,
    )

    threshold_metrics.to_csv(
        THRESHOLD_METRICS_PATH,
        index=False,
    )

    confusion_matrices.to_csv(
        CONFUSION_MATRICES_PATH,
        index=False,
    )

    classification_report_data.to_csv(
        CLASSIFICATION_REPORT_PATH,
        index=False,
    )

    roc_curve_data.to_csv(
        ROC_CURVE_PATH,
        index=False,
    )

    precision_recall_data.to_csv(
        PRECISION_RECALL_CURVE_PATH,
        index=False,
    )

    targeting_lift.to_csv(
        TARGETING_LIFT_PATH,
        index=False,
    )

    permutation_importance_data.to_csv(
        PERMUTATION_IMPORTANCE_PATH,
        index=False,
    )

    test_predictions.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
    )


def save_model_metadata(
    best_parameters: dict[str, Any],
    threshold_payload: dict[str, Any],
    selected_threshold: float,
    probability_metrics: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    train_rows: int,
    test_rows: int,
) -> None:
    """Save reproducibility and model-artifact metadata."""
    selected_metrics = (
        threshold_metrics.loc[
            threshold_metrics[
                "threshold_name"
            ]
            == "selected_training_oof_f1"
        ]
        .iloc[0]
    )

    probability_result = (
        probability_metrics.iloc[0]
    )

    metadata = {
        "model_name": (
            "Bank Marketing Pre-Campaign "
            "Balanced Random Forest"
        ),
        "model_type": "RandomForestClassifier",
        "prediction_contract": (
            "Prioritize clients before current-campaign "
            "contacts begin."
        ),
        "target_column": TARGET_COLUMN,
        "positive_class": 1,
        "training_rows": train_rows,
        "test_rows": test_rows,
        "model_features": MODEL_FEATURES,
        "selected_threshold": selected_threshold,
        "threshold_selection_metadata": (
            threshold_payload
        ),
        "best_parameters": best_parameters,
        "final_test_metrics": {
            "roc_auc": float(
                probability_result["roc_auc"]
            ),
            "average_precision": float(
                probability_result[
                    "average_precision"
                ]
            ),
            "brier_score": float(
                probability_result[
                    "brier_score"
                ]
            ),
            "log_loss": float(
                probability_result["log_loss"]
            ),
            "precision": float(
                selected_metrics["precision"]
            ),
            "recall": float(
                selected_metrics["recall"]
            ),
            "f1": float(
                selected_metrics["f1"]
            ),
            "f2": float(
                selected_metrics["f2"]
            ),
            "balanced_accuracy": float(
                selected_metrics[
                    "balanced_accuracy"
                ]
            ),
        },
        "software_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "model_file": str(
            FINAL_MODEL_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "model_sha256": calculate_file_sha256(
            FINAL_MODEL_PATH
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "test_set_used_for_model_selection": False,
        "test_set_used_for_final_reporting": True,
    }

    MODEL_METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def display_results(
    probability_metrics: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    targeting_lift: pd.DataFrame,
    permutation_importance_data: pd.DataFrame,
) -> None:
    """Display concise final evaluation results."""
    print("\nFinal test probability metrics:")
    print(
        probability_metrics.to_string(
            index=False
        )
    )

    print("\nFinal threshold comparison:")

    threshold_columns = [
        "threshold_name",
        "threshold",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
        "lift_over_baseline",
        "predicted_positive_percentage",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
    ]

    print(
        threshold_metrics[
            threshold_columns
        ].to_string(index=False)
    )

    print("\nTargeting lift:")
    print(
        targeting_lift[
            [
                "targeting_percentage",
                "targeted_clients",
                "subscribers_captured",
                "subscriber_capture_percentage",
                "targeting_precision_percentage",
                "lift_over_random_targeting",
            ]
        ].to_string(index=False)
    )

    print("\nTop permutation importances:")
    print(
        permutation_importance_data[
            [
                "importance_rank",
                "feature",
                "importance_mean",
                "importance_std",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


def main() -> None:
    """Train and evaluate the final held-out model."""
    print("=" * 70)
    print("BANK MARKETING — FINAL HELD-OUT TEST EVALUATION")
    print("=" * 70)

    (
        train_features,
        train_target,
        test_features,
        test_target,
    ) = load_modeling_data()

    (
        best_parameters,
        threshold_payload,
        selected_threshold,
    ) = load_model_configuration()

    print(f"\nTraining feature shape: {train_features.shape}")
    print(f"Final test feature shape: {test_features.shape}")

    print(
        "\nSelected threshold from training OOF predictions:",
        round(selected_threshold, 6),
    )

    print(
        "\nTraining the tuned final pipeline "
        "on the complete training set..."
    )

    final_pipeline = build_final_pipeline(
        best_parameters
    )

    final_pipeline.fit(
        train_features,
        train_target,
    )

    print(
        "\nGenerating final held-out test probabilities..."
    )

    test_probabilities = (
        final_pipeline.predict_proba(
            test_features
        )[:, 1]
    )

    probability_metrics = (
        calculate_probability_metrics(
            test_target=test_target,
            probabilities=test_probabilities,
        )
    )

    (
        threshold_metrics,
        confusion_matrices,
        classification_report_data,
    ) = build_threshold_reports(
        test_target=test_target,
        probabilities=test_probabilities,
        selected_threshold=selected_threshold,
    )

    (
        roc_curve_data,
        precision_recall_data,
    ) = build_probability_curves(
        test_target=test_target,
        probabilities=test_probabilities,
    )

    targeting_lift = build_targeting_lift_table(
        test_target=test_target,
        probabilities=test_probabilities,
    )

    print(
        "\nCalculating held-out permutation importance..."
    )

    permutation_importance_data = (
        calculate_permutation_importance(
            final_pipeline=final_pipeline,
            test_features=test_features,
            test_target=test_target,
        )
    )

    test_predictions = build_test_predictions(
        test_target=test_target,
        probabilities=test_probabilities,
        selected_threshold=selected_threshold,
    )

    validate_results(
        probability_metrics=probability_metrics,
        threshold_metrics=threshold_metrics,
        confusion_matrices=confusion_matrices,
        roc_curve_data=roc_curve_data,
        precision_recall_data=precision_recall_data,
        targeting_lift=targeting_lift,
        permutation_importance_data=(
            permutation_importance_data
        ),
        test_target=test_target,
        probabilities=test_probabilities,
    )

    save_reports(
        probability_metrics=probability_metrics,
        threshold_metrics=threshold_metrics,
        confusion_matrices=confusion_matrices,
        classification_report_data=(
            classification_report_data
        ),
        roc_curve_data=roc_curve_data,
        precision_recall_data=(
            precision_recall_data
        ),
        targeting_lift=targeting_lift,
        permutation_importance_data=(
            permutation_importance_data
        ),
        test_predictions=test_predictions,
    )

    save_model_artifact(
        final_pipeline=final_pipeline,
        selected_threshold=selected_threshold,
    )

    validate_model_reload(
        final_pipeline=final_pipeline,
        test_features=test_features,
    )

    save_model_metadata(
        best_parameters=best_parameters,
        threshold_payload=threshold_payload,
        selected_threshold=selected_threshold,
        probability_metrics=probability_metrics,
        threshold_metrics=threshold_metrics,
        train_rows=len(train_features),
        test_rows=len(test_features),
    )

    display_results(
        probability_metrics=probability_metrics,
        threshold_metrics=threshold_metrics,
        targeting_lift=targeting_lift,
        permutation_importance_data=(
            permutation_importance_data
        ),
    )

    print("\nGenerated final evaluation files:")
    print(f"- {PROBABILITY_METRICS_PATH}")
    print(f"- {THRESHOLD_METRICS_PATH}")
    print(f"- {CONFUSION_MATRICES_PATH}")
    print(f"- {CLASSIFICATION_REPORT_PATH}")
    print(f"- {ROC_CURVE_PATH}")
    print(f"- {PRECISION_RECALL_CURVE_PATH}")
    print(f"- {PERMUTATION_IMPORTANCE_PATH}")
    print(f"- {TARGETING_LIFT_PATH}")
    print(f"- {MODEL_METADATA_PATH}")
    print(f"- {TEST_PREDICTIONS_PATH}")
    print(f"- {FINAL_MODEL_PATH}")

    print(
        "\nFinal held-out test evaluation and model "
        "persistence completed successfully."
    )

    print(
        "\nThe test set is now considered consumed. "
        "Do not tune the model or threshold using these results."
    )


if __name__ == "__main__":
    main()
