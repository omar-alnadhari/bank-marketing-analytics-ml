"""Tune the balanced Random Forest using training data only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
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

TUNING_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "random_forest_tuning_results.csv"
)

BEST_PARAMETERS_PATH = (
    OUTPUT_DIRECTORY
    / "random_forest_best_parameters.json"
)

BEST_CV_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "random_forest_best_cv_summary.csv"
)

BASELINE_COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "model_cv_comparison.csv"
)


NUMBER_OF_FOLDS = 5
NUMBER_OF_CANDIDATES = 16


def build_random_forest_pipeline() -> Pipeline:
    """Build the preprocessing and Random Forest pipeline."""
    return Pipeline(
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


def build_parameter_distributions() -> dict[str, list[Any]]:
    """Return the Random Forest tuning search space."""
    return {
        "classifier__n_estimators": [
            200,
            300,
            450,
            600,
        ],
        "classifier__max_depth": [
            8,
            10,
            12,
            16,
            20,
            None,
        ],
        "classifier__min_samples_split": [
            2,
            5,
            10,
            20,
        ],
        "classifier__min_samples_leaf": [
            1,
            2,
            5,
            10,
            20,
        ],
        "classifier__max_features": [
            "sqrt",
            "log2",
            0.5,
        ],
        "classifier__class_weight": [
            "balanced",
            "balanced_subsample",
        ],
        "classifier__max_samples": [
            None,
            0.70,
            0.85,
        ],
    }


def build_scoring_metrics() -> dict[str, Any]:
    """Return the metrics calculated during tuning."""
    return {
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
        "balanced_accuracy": "balanced_accuracy",
        "precision": make_scorer(
            precision_score,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0,
        ),
    }


def run_randomized_search(
    train_features: pd.DataFrame,
    train_target: pd.Series,
) -> RandomizedSearchCV:
    """Tune the Random Forest using stratified cross-validation."""
    cross_validator = StratifiedKFold(
        n_splits=NUMBER_OF_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=build_random_forest_pipeline(),
        param_distributions=(
            build_parameter_distributions()
        ),
        n_iter=NUMBER_OF_CANDIDATES,
        scoring=build_scoring_metrics(),
        refit="average_precision",
        cv=cross_validator,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=2,
        return_train_score=False,
        error_score="raise",
    )

    search.fit(
        train_features,
        train_target,
    )

    return search


def build_tuning_results(
    search: RandomizedSearchCV,
) -> pd.DataFrame:
    """Create a concise table from RandomizedSearchCV results."""
    complete_results = pd.DataFrame(
        search.cv_results_
    )

    parameter_columns = sorted(
        column
        for column in complete_results.columns
        if column.startswith("param_")
    )

    metric_columns = [
        "rank_test_average_precision",
        "mean_test_average_precision",
        "std_test_average_precision",
        "mean_test_roc_auc",
        "std_test_roc_auc",
        "mean_test_balanced_accuracy",
        "mean_test_precision",
        "mean_test_recall",
        "mean_test_f1",
        "mean_fit_time",
        "std_fit_time",
    ]

    tuning_results = complete_results[
        parameter_columns
        + metric_columns
    ].copy()

    tuning_results = (
        tuning_results
        .sort_values(
            "rank_test_average_precision"
        )
        .reset_index(drop=True)
    )

    return tuning_results


def build_best_cv_summary(
    search: RandomizedSearchCV,
) -> pd.DataFrame:
    """Return the cross-validation metrics for the best candidate."""
    results = search.cv_results_
    best_index = search.best_index_

    summary = {
        "selection_metric": "average_precision",
        "best_average_precision": float(
            results[
                "mean_test_average_precision"
            ][best_index]
        ),
        "std_average_precision": float(
            results[
                "std_test_average_precision"
            ][best_index]
        ),
        "mean_roc_auc": float(
            results[
                "mean_test_roc_auc"
            ][best_index]
        ),
        "std_roc_auc": float(
            results[
                "std_test_roc_auc"
            ][best_index]
        ),
        "mean_balanced_accuracy": float(
            results[
                "mean_test_balanced_accuracy"
            ][best_index]
        ),
        "mean_precision": float(
            results[
                "mean_test_precision"
            ][best_index]
        ),
        "mean_recall": float(
            results[
                "mean_test_recall"
            ][best_index]
        ),
        "mean_f1": float(
            results[
                "mean_test_f1"
            ][best_index]
        ),
        "number_of_candidates": (
            NUMBER_OF_CANDIDATES
        ),
        "number_of_folds": NUMBER_OF_FOLDS,
    }

    return pd.DataFrame([summary])


def load_baseline_random_forest_result() -> pd.Series:
    """Load the original Random Forest benchmark result."""
    if not BASELINE_COMPARISON_PATH.exists():
        raise FileNotFoundError(
            "Baseline model-comparison report was not found. "
            "Run: python src\\train.py"
        )

    baseline_results = pd.read_csv(
        BASELINE_COMPARISON_PATH
    )

    matching_rows = baseline_results.loc[
        baseline_results["model"]
        == "random_forest_balanced"
    ]

    if len(matching_rows) != 1:
        raise ValueError(
            "Expected exactly one baseline "
            "Random Forest result."
        )

    return matching_rows.iloc[0]


def validate_results(
    search: RandomizedSearchCV,
    tuning_results: pd.DataFrame,
    best_summary: pd.DataFrame,
) -> None:
    """Validate the generated tuning outputs."""
    if len(tuning_results) != NUMBER_OF_CANDIDATES:
        raise ValueError(
            "Unexpected number of tuning candidates."
        )

    if int(
        tuning_results.loc[
            0,
            "rank_test_average_precision",
        ]
    ) != 1:
        raise ValueError(
            "The first tuning result is not ranked first."
        )

    metric_columns = [
        "mean_test_average_precision",
        "mean_test_roc_auc",
        "mean_test_balanced_accuracy",
        "mean_test_precision",
        "mean_test_recall",
        "mean_test_f1",
    ]

    for metric_column in metric_columns:
        if not tuning_results[
            metric_column
        ].between(0, 1).all():
            raise ValueError(
                f"Invalid tuning metric: {metric_column}"
            )

    if search.best_estimator_ is None:
        raise ValueError(
            "The tuning process produced no best estimator."
        )

    if not search.best_params_:
        raise ValueError(
            "The tuning process produced no best parameters."
        )

    best_average_precision = float(
        best_summary.loc[
            0,
            "best_average_precision",
        ]
    )

    if best_average_precision <= 0:
        raise ValueError(
            "Invalid best average-precision score."
        )


def save_outputs(
    search: RandomizedSearchCV,
    tuning_results: pd.DataFrame,
    best_summary: pd.DataFrame,
) -> None:
    """Save tuning results and best parameters."""
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    tuning_results.to_csv(
        TUNING_RESULTS_PATH,
        index=False,
    )

    best_summary.to_csv(
        BEST_CV_SUMMARY_PATH,
        index=False,
    )

    BEST_PARAMETERS_PATH.write_text(
        json.dumps(
            search.best_params_,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def display_results(
    search: RandomizedSearchCV,
    best_summary: pd.DataFrame,
    baseline_result: pd.Series,
) -> None:
    """Display the best candidate and baseline comparison."""
    print("\nBest Random Forest parameters:")

    for parameter, value in sorted(
        search.best_params_.items()
    ):
        print(f"- {parameter}: {value}")

    print("\nBest cross-validation summary:")
    print(
        best_summary.to_string(
            index=False
        )
    )

    baseline_average_precision = float(
        baseline_result[
            "mean_average_precision"
        ]
    )

    tuned_average_precision = float(
        best_summary.loc[
            0,
            "best_average_precision",
        ]
    )

    print("\nAverage-precision comparison:")
    print(
        "- Baseline Random Forest:",
        round(
            baseline_average_precision,
            6,
        ),
    )

    print(
        "- Tuned Random Forest:",
        round(
            tuned_average_precision,
            6,
        ),
    )

    print(
        "- Difference:",
        round(
            tuned_average_precision
            - baseline_average_precision,
            6,
        ),
    )


def main() -> None:
    """Run Random Forest hyperparameter tuning."""
    print("=" * 70)
    print("BANK MARKETING — RANDOM FOREST TUNING")
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
        "No test predictions or metrics are "
        "calculated during tuning."
    )

    print(
        f"\nTesting {NUMBER_OF_CANDIDATES} "
        f"parameter combinations with "
        f"{NUMBER_OF_FOLDS}-fold cross-validation."
    )

    search = run_randomized_search(
        train_features=train_features,
        train_target=train_target,
    )

    tuning_results = build_tuning_results(
        search
    )

    best_summary = build_best_cv_summary(
        search
    )

    baseline_result = (
        load_baseline_random_forest_result()
    )

    validate_results(
        search=search,
        tuning_results=tuning_results,
        best_summary=best_summary,
    )

    save_outputs(
        search=search,
        tuning_results=tuning_results,
        best_summary=best_summary,
    )

    display_results(
        search=search,
        best_summary=best_summary,
        baseline_result=baseline_result,
    )

    print("\nGenerated tuning reports:")
    print(f"- {TUNING_RESULTS_PATH}")
    print(f"- {BEST_PARAMETERS_PATH}")
    print(f"- {BEST_CV_SUMMARY_PATH}")

    print(
        "\nRandom Forest tuning completed successfully "
        "without evaluating the test set."
    )


if __name__ == "__main__":
    main()
