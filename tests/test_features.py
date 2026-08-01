"""Tests for leakage-safe feature engineering and split helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from src import features


def build_raw_feature_frame() -> pd.DataFrame:
    """Create a small source-like frame with all raw features."""
    return pd.DataFrame(
        {
            "age": [33, 48, 27, 61, 40, 52, 29, 36, 44, 57],
            "job": [
                "technician",
                "management",
                "services",
                "retired",
                "admin.",
                "blue-collar",
                "student",
                "entrepreneur",
                "housemaid",
                "self-employed",
            ],
            "marital": [
                "single",
                "married",
                "single",
                "married",
                "divorced",
                "married",
                "single",
                "married",
                "divorced",
                "married",
            ],
            "education": [
                "secondary",
                "tertiary",
                "secondary",
                "primary",
                "secondary",
                "primary",
                "tertiary",
                "tertiary",
                "primary",
                "secondary",
            ],
            "default": [
                "no",
                "no",
                "no",
                "no",
                "yes",
                "no",
                "no",
                "no",
                "yes",
                "no",
            ],
            "balance": [1500, -20, 350, 4200, -5, 80, 700, 150, -300, 930],
            "housing": ["yes", "no", "yes", "no", "yes", "yes", "no", "no", "yes", "no"],
            "loan": ["no", "no", "yes", "no", "yes", "no", "no", "yes", "no", "no"],
            "contact": ["cellular", "unknown", "cellular", "telephone", "unknown", "cellular", "cellular", "telephone", "unknown", "cellular"],
            "day": [5, 12, 18, 22, 3, 14, 7, 28, 11, 19],
            "month": ["may", "jun", "jul", "aug", "nov", "apr", "feb", "oct", "may", "dec"],
            "duration": [120, 65, 300, 90, 45, 240, 600, 35, 180, 220],
            "campaign": [1, 2, 1, 3, 4, 1, 2, 1, 5, 2],
            "pdays": [-1, 10, -1, 200, 0, -1, 45, 300, -1, 15],
            "previous": [0, 2, 0, 3, 1, 0, 4, 5, 0, 1],
            "poutcome": [
                "unknown",
                "success",
                "unknown",
                "failure",
                "other",
                "unknown",
                "success",
                "failure",
                "unknown",
                "other",
            ],
            "subscribed": [0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
        }
    )


def test_engineer_features_creates_leakage_safe_schema() -> None:
    raw_features = build_raw_feature_frame().drop(columns=[features.TARGET_COLUMN])

    engineered = features.engineer_features(raw_features)

    assert engineered.columns.tolist() == features.MODEL_FEATURES
    assert features.TARGET_COLUMN not in engineered.columns
    assert not set(features.CURRENT_CAMPAIGN_COLUMNS) & set(engineered.columns)
    assert engineered["previously_contacted"].tolist() == [
        "no",
        "yes",
        "no",
        "yes",
        "yes",
        "no",
        "yes",
        "yes",
        "no",
        "yes",
    ]
    assert engineered["negative_balance"].tolist() == [
        "no",
        "yes",
        "no",
        "no",
        "yes",
        "no",
        "no",
        "no",
        "yes",
        "no",
    ]
    assert engineered.loc[
        engineered["previously_contacted"] == "no",
        "pdays_since_previous_contact",
    ].isna().all()
    assert engineered.loc[
        engineered["previously_contacted"] == "yes",
        "pdays_since_previous_contact",
    ].ge(0).all()


def test_engineer_features_requires_all_raw_model_inputs() -> None:
    raw_features = build_raw_feature_frame().drop(
        columns=[features.TARGET_COLUMN, "job"]
    )

    with pytest.raises(ValueError, match="Required raw features"):
        features.engineer_features(raw_features)


def test_create_train_test_split_is_stratified_and_disjoint() -> None:
    data = build_raw_feature_frame()

    (
        train_features,
        test_features,
        train_target,
        test_target,
        train_indices,
        test_indices,
    ) = features.create_train_test_split(data)

    assert len(train_features) == 8
    assert len(test_features) == 2
    assert train_features.columns.tolist() == features.MODEL_FEATURES
    assert test_features.columns.tolist() == features.MODEL_FEATURES
    assert set(train_indices).isdisjoint(set(test_indices))
    assert train_target.sum() == 3
    assert test_target.sum() == 1


def test_combine_features_and_target_preserves_alignment() -> None:
    raw_features = build_raw_feature_frame().drop(columns=[features.TARGET_COLUMN])
    engineered = features.engineer_features(raw_features)
    target = build_raw_feature_frame()[features.TARGET_COLUMN]

    combined = features.combine_features_and_target(engineered, target)

    assert combined.columns.tolist() == [
        *features.MODEL_FEATURES,
        features.TARGET_COLUMN,
    ]
    assert combined[features.TARGET_COLUMN].tolist() == target.tolist()


def test_combine_features_and_target_rejects_misalignment() -> None:
    raw_features = build_raw_feature_frame().drop(columns=[features.TARGET_COLUMN])
    engineered = features.engineer_features(raw_features)
    target = build_raw_feature_frame()[features.TARGET_COLUMN].sample(
        frac=1,
        random_state=0,
    )

    with pytest.raises(ValueError, match="indices are not aligned"):
        features.combine_features_and_target(engineered, target)


def test_feature_contract_documents_current_campaign_exclusions() -> None:
    contract = features.build_feature_contract()

    excluded_features = set(
        contract.loc[
            contract["primary_model_status"] == "excluded",
            "feature",
        ]
    )

    assert set(features.CURRENT_CAMPAIGN_COLUMNS).issubset(excluded_features)
    assert "duration" in excluded_features
    assert set(features.MODEL_FEATURES).issubset(set(contract["feature"]))
