"""Prepare leakage-safe features and train/test splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bank_marketing_clean.csv"
)

INTERIM_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "interim"
)

MODELING_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
)

TRAIN_DATA_PATH = (
    INTERIM_DATA_DIRECTORY
    / "model_train.csv"
)

TEST_DATA_PATH = (
    INTERIM_DATA_DIRECTORY
    / "model_test.csv"
)

FEATURE_CONTRACT_PATH = (
    MODELING_REPORT_DIRECTORY
    / "feature_contract.csv"
)

SPLIT_SUMMARY_PATH = (
    MODELING_REPORT_DIRECTORY
    / "train_test_split_summary.csv"
)


TARGET_COLUMN = "subscribed"

TEST_SIZE = 0.20
RANDOM_STATE = 42


EXPECTED_SOURCE_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "subscribed",
]


CURRENT_CAMPAIGN_COLUMNS = [
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
]


PRIMARY_MODEL_RAW_FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "pdays",
    "previous",
    "poutcome",
]


NUMERICAL_MODEL_FEATURES = [
    "age",
    "balance",
    "previous",
    "pdays_since_previous_contact",
]


CATEGORICAL_MODEL_FEATURES = [
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
    NUMERICAL_MODEL_FEATURES
    + CATEGORICAL_MODEL_FEATURES
)


def load_clean_data(
    path: Path = DATA_PATH,
) -> pd.DataFrame:
    """Load and validate the cleaned source dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {path}\n"
            "Run: python src\\data_cleaning.py"
        )

    data = pd.read_csv(path)

    if data.columns.tolist() != EXPECTED_SOURCE_COLUMNS:
        raise ValueError(
            "The cleaned dataset schema does not match "
            "the expected feature-engineering schema."
        )

    if len(data) != 45_211:
        raise ValueError(
            f"Unexpected row count: {len(data)}."
        )

    if data.isna().any().any():
        raise ValueError(
            "The cleaned source dataset contains missing values."
        )

    if set(data[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError(
            "The target column must contain only 0 and 1."
        )

    return data


def engineer_features(
    raw_features: pd.DataFrame,
) -> pd.DataFrame:
    """Create deterministic features available before the campaign."""
    missing_raw_features = sorted(
        set(PRIMARY_MODEL_RAW_FEATURES)
        - set(raw_features.columns)
    )

    if missing_raw_features:
        raise ValueError(
            "Required raw features are missing: "
            f"{missing_raw_features}"
        )

    engineered_features = raw_features[
        PRIMARY_MODEL_RAW_FEATURES
    ].copy()

    engineered_features["previously_contacted"] = (
        np.where(
            engineered_features["pdays"] >= 0,
            "yes",
            "no",
        )
    )

    engineered_features[
        "pdays_since_previous_contact"
    ] = (
        engineered_features["pdays"]
        .where(
            engineered_features["pdays"] >= 0,
            np.nan,
        )
    )

    engineered_features["negative_balance"] = (
        np.where(
            engineered_features["balance"] < 0,
            "yes",
            "no",
        )
    )

    engineered_features = (
        engineered_features
        .drop(columns=["pdays"])
    )

    engineered_features = engineered_features[
        MODEL_FEATURES
    ]

    validate_engineered_features(
        engineered_features
    )

    return engineered_features


def validate_engineered_features(
    features: pd.DataFrame,
) -> None:
    """Validate the leakage-safe engineered feature matrix."""
    if features.columns.tolist() != MODEL_FEATURES:
        raise ValueError(
            "Engineered feature columns do not match "
            "the expected primary-model feature list."
        )

    forbidden_columns_found = sorted(
        set(CURRENT_CAMPAIGN_COLUMNS)
        & set(features.columns)
    )

    if forbidden_columns_found:
        raise ValueError(
            "Current-campaign features were found "
            "in the primary model: "
            f"{forbidden_columns_found}"
        )

    if TARGET_COLUMN in features.columns:
        raise ValueError(
            "The target column was included in the features."
        )

    allowed_missing_column = (
        "pdays_since_previous_contact"
    )

    unexpected_missing_columns = [
        column
        for column in features.columns
        if (
            column != allowed_missing_column
            and features[column].isna().any()
        )
    ]

    if unexpected_missing_columns:
        raise ValueError(
            "Unexpected missing values were found in: "
            f"{unexpected_missing_columns}"
        )

    expected_missing_mask = (
        features["previously_contacted"] == "no"
    )

    actual_missing_mask = (
        features[
            "pdays_since_previous_contact"
        ].isna()
    )

    if not actual_missing_mask.equals(
        expected_missing_mask
    ):
        raise ValueError(
            "Missing pdays values do not match "
            "the previously-contacted indicator."
        )

    contacted_values = features.loc[
        features["previously_contacted"] == "yes",
        "pdays_since_previous_contact",
    ]

    if not (contacted_values >= 0).all():
        raise ValueError(
            "Previously contacted clients have "
            "invalid pdays values."
        )

    if set(
        features["previously_contacted"].unique()
    ) != {"yes", "no"}:
        raise ValueError(
            "Unexpected previously_contacted values."
        )

    if set(
        features["negative_balance"].unique()
    ) != {"yes", "no"}:
        raise ValueError(
            "Unexpected negative_balance values."
        )


def create_train_test_split(
    data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Index,
    pd.Index,
]:
    """Split raw records before applying feature engineering."""
    raw_features = data.drop(
        columns=[TARGET_COLUMN]
    )

    target = data[TARGET_COLUMN].astype(
        "int8"
    )

    (
        raw_train_features,
        raw_test_features,
        train_target,
        test_target,
    ) = train_test_split(
        raw_features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    train_indices = raw_train_features.index
    test_indices = raw_test_features.index

    if not set(train_indices).isdisjoint(
        set(test_indices)
    ):
        raise ValueError(
            "Train and test row indices overlap."
        )

    train_features = engineer_features(
        raw_train_features
    )

    test_features = engineer_features(
        raw_test_features
    )

    return (
        train_features,
        test_features,
        train_target,
        test_target,
        train_indices,
        test_indices,
    )


def build_split_summary(
    full_target: pd.Series,
    train_target: pd.Series,
    test_target: pd.Series,
) -> pd.DataFrame:
    """Summarize class distributions across all splits."""
    result_records: list[dict[str, object]] = []

    split_targets = {
        "full_dataset": full_target,
        "train": train_target,
        "test": test_target,
    }

    for split_name, target in split_targets.items():
        total_records = len(target)

        subscriber_count = int(
            (target == 1).sum()
        )

        non_subscriber_count = int(
            (target == 0).sum()
        )

        result_records.append(
            {
                "split": split_name,
                "record_count": total_records,
                "non_subscribers": (
                    non_subscriber_count
                ),
                "subscribers": subscriber_count,
                "subscription_rate_percentage": round(
                    100
                    * subscriber_count
                    / total_records,
                    4,
                ),
            }
        )

    return pd.DataFrame(result_records)


def build_feature_contract() -> pd.DataFrame:
    """Document feature availability and modeling decisions."""
    feature_records = [
        {
            "feature": "age",
            "source_or_derived": "source",
            "primary_model_status": "included",
            "model_representation": "numeric",
            "reason": (
                "Customer attribute available before the campaign; "
                "requires fairness review."
            ),
        },
        {
            "feature": "job",
            "source_or_derived": "source",
            "primary_model_status": "included",
            "model_representation": "categorical",
            "reason": (
                "Customer attribute available before the campaign."
            ),
        },
        {
            "feature": "marital",
            "source_or_derived": "source",
            "primary_model_status": "included",
            "model_representation": "categorical",
            "reason": (
                "Available before the campaign; "
                "requires fairness review."
            ),
        },
        {
            "feature": "education",
            "source_or_derived": "source",
            "primary_model_status": "included",
            "model_representation": "categorical",
            "reason": (
                "Customer attribute available before the campaign."
            ),
        },
        {
            "feature": "default",
            "source_or_derived": "source",
            "primary_model_status": "included",
            "model_representation": "categorical",
            "reason": (
                "Existing financial-status information."
            ),
        },
        {
            "feature": "balance",
            "source_or_derived": "source",
            "primary_model_status": "included",
            "model_representation": "numeric",
            "reason": (
                "Existing account information available "
                "before campaign contact."
            ),
        },
        {
            "feature": "housing",
            "source_or_derived": "source",
            "primary_model_status": "included",
            "model_representation": "categorical",
            "reason": (
                "Existing financial-obligation information."
            ),
        },
        {
            "feature": "loan",
            "source_or_derived": "source",
            "primary_model_status": "included",
            "model_representation": "categorical",
            "reason": (
                "Existing financial-obligation information."
            ),
        },
        {
            "feature": "contact",
            "source_or_derived": "source",
            "primary_model_status": "excluded",
            "model_representation": "not used",
            "reason": (
                "Current-campaign execution information."
            ),
        },
        {
            "feature": "day",
            "source_or_derived": "source",
            "primary_model_status": "excluded",
            "model_representation": "not used",
            "reason": (
                "Current-campaign execution date."
            ),
        },
        {
            "feature": "month",
            "source_or_derived": "source",
            "primary_model_status": "excluded",
            "model_representation": "not used",
            "reason": (
                "Current-campaign execution timing."
            ),
        },
        {
            "feature": "duration",
            "source_or_derived": "source",
            "primary_model_status": "excluded",
            "model_representation": "not used",
            "reason": (
                "Known only after the call ends; "
                "temporal data leakage."
            ),
        },
        {
            "feature": "campaign",
            "source_or_derived": "source",
            "primary_model_status": "excluded",
            "model_representation": "not used",
            "reason": (
                "Current-campaign contact count may accumulate "
                "after the targeting decision."
            ),
        },
        {
            "feature": "pdays",
            "source_or_derived": "source",
            "primary_model_status": "transformed",
            "model_representation": "not used directly",
            "reason": (
                "Sentinel value -1 is separated from elapsed days."
            ),
        },
        {
            "feature": "previous",
            "source_or_derived": "source",
            "primary_model_status": "included",
            "model_representation": "numeric",
            "reason": (
                "Historical campaign-contact count."
            ),
        },
        {
            "feature": "poutcome",
            "source_or_derived": "source",
            "primary_model_status": "included",
            "model_representation": "categorical",
            "reason": (
                "Historical campaign outcome."
            ),
        },
        {
            "feature": "previously_contacted",
            "source_or_derived": "derived",
            "primary_model_status": "included",
            "model_representation": "categorical",
            "reason": (
                "Separates no prior contact from valid pdays values."
            ),
        },
        {
            "feature": "pdays_since_previous_contact",
            "source_or_derived": "derived",
            "primary_model_status": "included",
            "model_representation": "numeric",
            "reason": (
                "Elapsed days only for previously contacted clients."
            ),
        },
        {
            "feature": "negative_balance",
            "source_or_derived": "derived",
            "primary_model_status": "included",
            "model_representation": "categorical",
            "reason": (
                "Captures whether the account balance is negative."
            ),
        },
    ]

    return pd.DataFrame(feature_records)


def combine_features_and_target(
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """Combine features and target while preserving alignment."""
    if not features.index.equals(target.index):
        raise ValueError(
            "Feature and target indices are not aligned."
        )

    combined_data = features.copy()

    combined_data[TARGET_COLUMN] = target

    return combined_data.reset_index(drop=True)


def validate_split_outputs(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_target: pd.Series,
    test_target: pd.Series,
    train_indices: pd.Index,
    test_indices: pd.Index,
) -> None:
    """Validate sizes, schemas, and class distributions."""
    if len(train_features) != 36_168:
        raise ValueError(
            f"Unexpected training size: {len(train_features)}."
        )

    if len(test_features) != 9_043:
        raise ValueError(
            f"Unexpected test size: {len(test_features)}."
        )

    if (
        len(train_features)
        + len(test_features)
        != 45_211
    ):
        raise ValueError(
            "Train and test sizes do not cover all records."
        )

    if train_features.columns.tolist() != MODEL_FEATURES:
        raise ValueError(
            "Unexpected training feature schema."
        )

    if test_features.columns.tolist() != MODEL_FEATURES:
        raise ValueError(
            "Unexpected test feature schema."
        )

    if set(train_indices) & set(test_indices):
        raise ValueError(
            "Train and test indices overlap."
        )

    if int(train_target.sum()) != 4_231:
        raise ValueError(
            "Unexpected training subscriber count."
        )

    if int(test_target.sum()) != 1_058:
        raise ValueError(
            "Unexpected test subscriber count."
        )

    if int((train_target == 0).sum()) != 31_937:
        raise ValueError(
            "Unexpected training non-subscriber count."
        )

    if int((test_target == 0).sum()) != 7_985:
        raise ValueError(
            "Unexpected test non-subscriber count."
        )

    validate_engineered_features(
        train_features
    )

    validate_engineered_features(
        test_features
    )


def save_outputs(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    split_summary: pd.DataFrame,
    feature_contract: pd.DataFrame,
) -> None:
    """Save reproducible modeling inputs and reports."""
    INTERIM_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELING_REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_data.to_csv(
        TRAIN_DATA_PATH,
        index=False,
    )

    test_data.to_csv(
        TEST_DATA_PATH,
        index=False,
    )

    split_summary.to_csv(
        SPLIT_SUMMARY_PATH,
        index=False,
    )

    feature_contract.to_csv(
        FEATURE_CONTRACT_PATH,
        index=False,
    )


def main() -> None:
    """Create leakage-safe model features and data splits."""
    print("=" * 70)
    print("BANK MARKETING — MODEL FEATURE PREPARATION")
    print("=" * 70)

    data = load_clean_data()

    (
        train_features,
        test_features,
        train_target,
        test_target,
        train_indices,
        test_indices,
    ) = create_train_test_split(data)

    validate_split_outputs(
        train_features=train_features,
        test_features=test_features,
        train_target=train_target,
        test_target=test_target,
        train_indices=train_indices,
        test_indices=test_indices,
    )

    train_data = combine_features_and_target(
        train_features,
        train_target,
    )

    test_data = combine_features_and_target(
        test_features,
        test_target,
    )

    split_summary = build_split_summary(
        full_target=data[TARGET_COLUMN],
        train_target=train_target,
        test_target=test_target,
    )

    feature_contract = build_feature_contract()

    save_outputs(
        train_data=train_data,
        test_data=test_data,
        split_summary=split_summary,
        feature_contract=feature_contract,
    )

    print(f"\nSource dataset shape: {data.shape}")

    print(
        "Primary model numerical features:",
        len(NUMERICAL_MODEL_FEATURES),
    )

    print(
        "Primary model categorical features:",
        len(CATEGORICAL_MODEL_FEATURES),
    )

    print(
        "Total primary model features:",
        len(MODEL_FEATURES),
    )

    print("\nExcluded current-campaign features:")

    for feature in CURRENT_CAMPAIGN_COLUMNS:
        print(f"- {feature}")

    print("\nTrain/test split summary:")
    print(split_summary.to_string(index=False))

    print("\nTraining data shape:")
    print(train_data.shape)

    print("\nTest data shape:")
    print(test_data.shape)

    print(
        "\nEngineered pdays missing values "
        "(intentional):"
    )

    print(
        "- Train:",
        int(
            train_features[
                "pdays_since_previous_contact"
            ].isna().sum()
        ),
    )

    print(
        "- Test:",
        int(
            test_features[
                "pdays_since_previous_contact"
            ].isna().sum()
        ),
    )

    print("\nGenerated files:")
    print(f"- {TRAIN_DATA_PATH}")
    print(f"- {TEST_DATA_PATH}")
    print(f"- {FEATURE_CONTRACT_PATH}")
    print(f"- {SPLIT_SUMMARY_PATH}")

    print(
        "\nLeakage-safe feature preparation and "
        "stratified train/test split completed successfully."
    )


if __name__ == "__main__":
    main()