"""Benchmark baseline classification models using cross-validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "model_train.csv"
)

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "model_test.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
)

CV_COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "model_cv_comparison.csv"
)

CV_FOLD_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "model_cv_fold_results.csv"
)

PREPROCESSING_CONTRACT_PATH = (
    OUTPUT_DIRECTORY
    / "preprocessing_contract.csv"
)


TARGET_COLUMN = "subscribed"

RANDOM_STATE = 42
NUMBER_OF_FOLDS = 5


NUMERICAL_FEATURES = [
    "age",
    "balance",
    "previous",
    "pdays_since_previous_contact",
]


CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "poutcome",
    "previously_contacted",
    "negative_balance",
]


MODEL_FEATURES = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)


EXPECTED_TRAIN_ROWS = 36_168
EXPECTED_TEST_ROWS = 9_043
EXPECTED_TRAIN_SUBSCRIBERS = 4_231
EXPECTED_TEST_SUBSCRIBERS = 1_058


def load_modeling_data(
    train_path: Path = TRAIN_DATA_PATH,
    test_path: Path = TEST_DATA_PATH,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """Load and validate the modeling train and test datasets."""
    missing_files = [
        path
        for path in [train_path, test_path]
        if not path.exists()
    ]

    if missing_files:
        formatted_paths = "\n".join(
            f"- {path}"
            for path in missing_files
        )

        raise FileNotFoundError(
            "Required modeling files were not found:\n"
            f"{formatted_paths}\n"
            "Run: python src\\features.py"
        )

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    expected_columns = [
        *MODEL_FEATURES,
        TARGET_COLUMN,
    ]

    if train_data.columns.tolist() != expected_columns:
        raise ValueError(
            "Unexpected training-data schema."
        )

    if test_data.columns.tolist() != expected_columns:
        raise ValueError(
            "Unexpected test-data schema."
        )

    if len(train_data) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"Unexpected training row count: "
            f"{len(train_data)}."
        )

    if len(test_data) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Unexpected test row count: "
            f"{len(test_data)}."
        )

    train_features = train_data[
        MODEL_FEATURES
    ].copy()

    test_features = test_data[
        MODEL_FEATURES
    ].copy()

    train_target = train_data[
        TARGET_COLUMN
    ].astype("int8")

    test_target = test_data[
        TARGET_COLUMN
    ].astype("int8")

    if set(train_target.unique()) != {0, 1}:
        raise ValueError(
            "Training target must contain only 0 and 1."
        )

    if set(test_target.unique()) != {0, 1}:
        raise ValueError(
            "Test target must contain only 0 and 1."
        )

    if int(train_target.sum()) != EXPECTED_TRAIN_SUBSCRIBERS:
        raise ValueError(
            "Unexpected training subscriber count."
        )

    if int(test_target.sum()) != EXPECTED_TEST_SUBSCRIBERS:
        raise ValueError(
            "Unexpected test subscriber count."
        )

    unexpected_train_missing = [
        column
        for column in MODEL_FEATURES
        if (
            column
            != "pdays_since_previous_contact"
            and train_features[column].isna().any()
        )
    ]

    unexpected_test_missing = [
        column
        for column in MODEL_FEATURES
        if (
            column
            != "pdays_since_previous_contact"
            and test_features[column].isna().any()
        )
    ]

    if unexpected_train_missing:
        raise ValueError(
            "Unexpected training missing values in: "
            f"{unexpected_train_missing}"
        )

    if unexpected_test_missing:
        raise ValueError(
            "Unexpected test missing values in: "
            f"{unexpected_test_missing}"
        )

    return (
        train_features,
        train_target,
        test_features,
        test_target,
    )


def build_preprocessor() -> ColumnTransformer:
    """Build leakage-safe numerical and categorical preprocessing."""
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def build_model_pipelines() -> dict[str, Pipeline]:
    """Build initial candidate-model pipelines."""
    dummy_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                DummyClassifier(
                    strategy="prior",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    logistic_regression_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight=None,
                    max_iter=2_000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    balanced_logistic_regression_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2_000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    balanced_random_forest_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=12,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )

    return {
        "dummy_prior": dummy_pipeline,
        "logistic_regression": (
            logistic_regression_pipeline
        ),
        "logistic_regression_balanced": (
            balanced_logistic_regression_pipeline
        ),
        "random_forest_balanced": (
            balanced_random_forest_pipeline
        ),
    }


def build_scoring_metrics() -> dict[str, Any]:
    """Return metrics used for cross-validation."""
    return {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
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


def run_cross_validation(
    models: dict[str, Pipeline],
    train_features: pd.DataFrame,
    train_target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate models using stratified cross-validation."""
    cross_validator = StratifiedKFold(
        n_splits=NUMBER_OF_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring_metrics = build_scoring_metrics()

    fold_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []

    for model_name, model_pipeline in models.items():
        print(f"\nEvaluating model: {model_name}")

        cross_validation_results = cross_validate(
            estimator=model_pipeline,
            X=train_features,
            y=train_target,
            scoring=scoring_metrics,
            cv=cross_validator,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )

        for fold_number in range(NUMBER_OF_FOLDS):
            fold_records.append(
                {
                    "model": model_name,
                    "fold": fold_number + 1,
                    "roc_auc": float(
                        cross_validation_results[
                            "test_roc_auc"
                        ][fold_number]
                    ),
                    "average_precision": float(
                        cross_validation_results[
                            "test_average_precision"
                        ][fold_number]
                    ),
                    "balanced_accuracy": float(
                        cross_validation_results[
                            "test_balanced_accuracy"
                        ][fold_number]
                    ),
                    "precision": float(
                        cross_validation_results[
                            "test_precision"
                        ][fold_number]
                    ),
                    "recall": float(
                        cross_validation_results[
                            "test_recall"
                        ][fold_number]
                    ),
                    "f1": float(
                        cross_validation_results[
                            "test_f1"
                        ][fold_number]
                    ),
                    "fit_time_seconds": float(
                        cross_validation_results[
                            "fit_time"
                        ][fold_number]
                    ),
                    "score_time_seconds": float(
                        cross_validation_results[
                            "score_time"
                        ][fold_number]
                    ),
                }
            )

        summary_record: dict[str, object] = {
            "model": model_name,
        }

        for metric_name in scoring_metrics:
            metric_values = (
                cross_validation_results[
                    f"test_{metric_name}"
                ]
            )

            summary_record[
                f"mean_{metric_name}"
            ] = float(metric_values.mean())

            summary_record[
                f"std_{metric_name}"
            ] = float(metric_values.std(ddof=1))

        summary_record["mean_fit_time_seconds"] = float(
            cross_validation_results[
                "fit_time"
            ].mean()
        )

        summary_record["mean_score_time_seconds"] = float(
            cross_validation_results[
                "score_time"
            ].mean()
        )

        summary_records.append(summary_record)

    fold_results = pd.DataFrame(fold_records)

    comparison_results = (
        pd.DataFrame(summary_records)
        .sort_values(
            [
                "mean_average_precision",
                "mean_roc_auc",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return fold_results, comparison_results


def build_preprocessing_contract() -> pd.DataFrame:
    """Document all preprocessing operations."""
    records = [
        {
            "feature_group": "numerical",
            "features": ", ".join(
                NUMERICAL_FEATURES
            ),
            "missing_value_strategy": (
                "Median imputation learned within each "
                "training fold"
            ),
            "encoding_or_scaling": (
                "StandardScaler learned within each "
                "training fold"
            ),
            "notes": (
                "A missingness indicator is added for "
                "pdays_since_previous_contact."
            ),
        },
        {
            "feature_group": "categorical",
            "features": ", ".join(
                CATEGORICAL_FEATURES
            ),
            "missing_value_strategy": (
                "Most-frequent imputation learned within "
                "each training fold"
            ),
            "encoding_or_scaling": (
                "OneHotEncoder with "
                "handle_unknown='ignore'"
            ),
            "notes": (
                "Unseen categories in validation or test "
                "data do not cause a prediction failure."
            ),
        },
    ]

    return pd.DataFrame(records)


def validate_cv_results(
    fold_results: pd.DataFrame,
    comparison_results: pd.DataFrame,
) -> None:
    """Validate cross-validation outputs."""
    expected_models = {
        "dummy_prior",
        "logistic_regression",
        "logistic_regression_balanced",
        "random_forest_balanced",
    }

    if set(comparison_results["model"]) != expected_models:
        raise ValueError(
            "Unexpected models in the comparison results."
        )

    expected_fold_rows = (
        len(expected_models)
        * NUMBER_OF_FOLDS
    )

    if len(fold_results) != expected_fold_rows:
        raise ValueError(
            "Unexpected number of fold-level results."
        )

    metric_columns = [
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
    ]

    for metric_column in metric_columns:
        if not fold_results[
            metric_column
        ].between(0, 1).all():
            raise ValueError(
                f"Invalid values found for "
                f"{metric_column}."
            )

    dummy_result = (
        comparison_results.loc[
            comparison_results["model"]
            == "dummy_prior"
        ]
        .iloc[0]
    )

    candidate_results = (
        comparison_results.loc[
            comparison_results["model"]
            != "dummy_prior"
        ]
    )

    dummy_average_precision = float(
        dummy_result["mean_average_precision"]
    )

    dummy_roc_auc = float(
        dummy_result["mean_roc_auc"]
    )

    if not (
        candidate_results["mean_average_precision"]
        > dummy_average_precision
    ).all():
        raise ValueError(
            "At least one candidate model did not "
            "outperform the dummy model on "
            "average precision."
        )

    if not (
        candidate_results["mean_roc_auc"]
        > dummy_roc_auc
    ).all():
        raise ValueError(
            "At least one candidate model did not "
            "outperform the dummy model on ROC-AUC."
        )

    if abs(dummy_roc_auc - 0.5) > 1e-9:
        raise ValueError(
            "The prior dummy classifier should have "
            "ROC-AUC equal to 0.5."
        )


def save_outputs(
    fold_results: pd.DataFrame,
    comparison_results: pd.DataFrame,
    preprocessing_contract: pd.DataFrame,
) -> None:
    """Save cross-validation and preprocessing reports."""
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_results.to_csv(
        CV_FOLD_RESULTS_PATH,
        index=False,
    )

    comparison_results.to_csv(
        CV_COMPARISON_PATH,
        index=False,
    )

    preprocessing_contract.to_csv(
        PREPROCESSING_CONTRACT_PATH,
        index=False,
    )


def display_results(
    comparison_results: pd.DataFrame,
) -> None:
    """Display a concise cross-validation comparison."""
    display_columns = [
        "model",
        "mean_roc_auc",
        "std_roc_auc",
        "mean_average_precision",
        "std_average_precision",
        "mean_balanced_accuracy",
        "mean_precision",
        "mean_recall",
        "mean_f1",
        "mean_fit_time_seconds",
    ]

    print("\nCross-validation model comparison:")

    print(
        comparison_results[
            display_columns
        ].to_string(index=False)
    )


def main() -> None:
    """Run initial model benchmarking."""
    print("=" * 70)
    print("BANK MARKETING — CANDIDATE MODEL BENCHMARKING")
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
        "Training subscription rate:",
        round(
            100 * train_target.mean(),
            4,
        ),
        "%",
    )

    print(
        "Test subscription rate:",
        round(
            100 * test_target.mean(),
            4,
        ),
        "%",
    )

    print(
        "\nThe test set is loaded only for schema validation."
    )

    print(
        "No test-set predictions or metrics are "
        "calculated in this stage."
    )

    models = build_model_pipelines()

    (
        fold_results,
        comparison_results,
    ) = run_cross_validation(
        models=models,
        train_features=train_features,
        train_target=train_target,
    )

    preprocessing_contract = (
        build_preprocessing_contract()
    )

    validate_cv_results(
        fold_results=fold_results,
        comparison_results=comparison_results,
    )

    save_outputs(
        fold_results=fold_results,
        comparison_results=comparison_results,
        preprocessing_contract=preprocessing_contract,
    )

    display_results(comparison_results)

    print("\nGenerated modeling reports:")
    print(f"- {CV_COMPARISON_PATH}")
    print(f"- {CV_FOLD_RESULTS_PATH}")
    print(f"- {PREPROCESSING_CONTRACT_PATH}")

    print(
    "\nCandidate-model cross-validation completed "
    "successfully without evaluating the test set."
    )


if __name__ == "__main__":
    main()
