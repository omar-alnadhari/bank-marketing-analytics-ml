"""Tests for saved modeling reports and final model artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src import features


REQUIRED_MODELING_REPORTS = {
    "feature_contract": "feature_contract.csv",
    "split_summary": "train_test_split_summary.csv",
    "preprocessing_contract": "preprocessing_contract.csv",
    "model_cv_comparison": "model_cv_comparison.csv",
    "model_cv_fold_results": "model_cv_fold_results.csv",
    "tuning_results": "random_forest_tuning_results.csv",
    "best_cv_summary": "random_forest_best_cv_summary.csv",
    "threshold_summary": "threshold_selection_summary.csv",
    "final_probability_metrics": "final_test_probability_metrics.csv",
    "final_threshold_metrics": "final_test_threshold_metrics.csv",
    "confusion_matrices": "final_test_confusion_matrices.csv",
    "classification_report": "final_test_classification_report.csv",
    "roc_curve": "final_test_roc_curve.csv",
    "precision_recall_curve": "final_test_precision_recall_curve.csv",
    "targeting_lift": "final_targeting_lift_table.csv",
    "permutation_importance": "test_permutation_importance.csv",
}

REQUIRED_MODELING_JSON = {
    "best_parameters": "random_forest_best_parameters.json",
    "selected_threshold": "selected_threshold.json",
    "model_metadata": "final_model_metadata.json",
}

EXPECTED_MODELS = {
    "dummy_prior",
    "logistic_regression",
    "logistic_regression_balanced",
    "random_forest_balanced",
}


@pytest.fixture(scope="module")
def modeling_reports(modeling_report_directory) -> dict[str, pd.DataFrame]:
    missing_files = [
        file_name
        for file_name in REQUIRED_MODELING_REPORTS.values()
        if not (modeling_report_directory / file_name).exists()
    ]

    if missing_files:
        pytest.skip(f"Generated modeling reports are missing: {missing_files}")

    return {
        report_name: pd.read_csv(modeling_report_directory / file_name)
        for report_name, file_name in REQUIRED_MODELING_REPORTS.items()
    }


@pytest.fixture(scope="module")
def modeling_json(modeling_report_directory) -> dict[str, dict[str, object]]:
    missing_files = [
        file_name
        for file_name in REQUIRED_MODELING_JSON.values()
        if not (modeling_report_directory / file_name).exists()
    ]

    if missing_files:
        pytest.skip(f"Generated modeling metadata files are missing: {missing_files}")

    return {
        json_name: json.loads(
            (modeling_report_directory / file_name).read_text(encoding="utf-8")
        )
        for json_name, file_name in REQUIRED_MODELING_JSON.items()
    }


def calculate_sha256(path: Path) -> str:
    sha256_hash = hashlib.sha256()

    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1_048_576), b""):
            sha256_hash.update(block)

    return sha256_hash.hexdigest()


def test_all_required_modeling_reports_are_present_and_non_empty(
    modeling_report_directory,
) -> None:
    for file_name in [
        *REQUIRED_MODELING_REPORTS.values(),
        *REQUIRED_MODELING_JSON.values(),
    ]:
        report_path = modeling_report_directory / file_name
        assert report_path.exists(), f"Missing report: {file_name}"
        assert report_path.stat().st_size > 0, f"Empty report: {file_name}"


def test_feature_contract_excludes_current_campaign_leakage(
    modeling_reports,
) -> None:
    feature_contract = modeling_reports["feature_contract"]
    excluded_features = set(
        feature_contract.loc[
            feature_contract["primary_model_status"] == "excluded",
            "feature",
        ]
    )
    included_or_transformed_features = set(
        feature_contract.loc[
            feature_contract["primary_model_status"].isin(
                ["included", "transformed"]
            ),
            "feature",
        ]
    )

    assert set(features.CURRENT_CAMPAIGN_COLUMNS).issubset(excluded_features)
    assert "duration" in excluded_features
    assert not set(features.CURRENT_CAMPAIGN_COLUMNS) & set(features.MODEL_FEATURES)
    assert set(features.MODEL_FEATURES).issubset(included_or_transformed_features)


def test_train_test_split_report_matches_expected_counts(
    modeling_reports,
) -> None:
    split_summary = modeling_reports["split_summary"].set_index("split")

    assert split_summary.loc["full_dataset", "record_count"] == 45_211
    assert split_summary.loc["train", "record_count"] == 36_168
    assert split_summary.loc["test", "record_count"] == 9_043
    assert split_summary.loc["full_dataset", "subscribers"] == 5_289
    assert split_summary.loc["train", "subscribers"] == 4_231
    assert split_summary.loc["test", "subscribers"] == 1_058
    assert split_summary.loc["train", "record_count"] + split_summary.loc[
        "test",
        "record_count",
    ] == split_summary.loc["full_dataset", "record_count"]


def test_cross_validation_reports_rank_candidates_above_dummy(
    modeling_reports,
) -> None:
    comparison = modeling_reports["model_cv_comparison"]
    fold_results = modeling_reports["model_cv_fold_results"]

    assert set(comparison["model"]) == EXPECTED_MODELS
    assert len(fold_results) == len(EXPECTED_MODELS) * 5

    metric_columns = [
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
    ]
    for metric_column in metric_columns:
        assert fold_results[metric_column].between(0, 1).all()

    dummy_result = comparison.loc[comparison["model"] == "dummy_prior"].iloc[0]
    candidate_results = comparison.loc[comparison["model"] != "dummy_prior"]

    assert np.isclose(dummy_result["mean_roc_auc"], 0.5)
    assert (
        candidate_results["mean_average_precision"]
        > dummy_result["mean_average_precision"]
    ).all()
    assert (candidate_results["mean_roc_auc"] > dummy_result["mean_roc_auc"]).all()
    assert comparison.iloc[0]["model"] == "random_forest_balanced"


def test_final_probability_metrics_are_stronger_than_baseline(
    modeling_reports,
) -> None:
    probability_metrics = modeling_reports["final_probability_metrics"]

    assert len(probability_metrics) == 1

    result = probability_metrics.iloc[0]

    assert result["test_records"] == 9_043
    assert result["test_subscribers"] == 1_058
    assert 0 < result["positive_class_prevalence"] < 1
    assert result["roc_auc"] > 0.5
    assert result["average_precision"] > result["positive_class_prevalence"]
    assert 0 <= result["brier_score"] <= 1
    assert result["log_loss"] > 0


def test_selected_threshold_is_consistent_across_reports_and_metadata(
    modeling_reports,
    modeling_json,
) -> None:
    selected_threshold = float(modeling_json["selected_threshold"]["threshold"])
    metadata_threshold = float(modeling_json["model_metadata"]["selected_threshold"])
    threshold_metrics = modeling_reports["final_threshold_metrics"]
    selected_metric = threshold_metrics.loc[
        threshold_metrics["threshold_name"] == "selected_training_oof_f1"
    ].iloc[0]

    assert 0 <= selected_threshold <= 1
    assert modeling_json["selected_threshold"]["test_set_used"] is False
    assert modeling_json["model_metadata"]["test_set_used_for_model_selection"] is False
    assert modeling_json["model_metadata"]["test_set_used_for_final_reporting"] is True
    assert metadata_threshold == pytest.approx(selected_threshold)
    assert selected_metric["threshold"] == pytest.approx(selected_threshold)
    assert selected_metric["f1"] >= threshold_metrics.loc[
        threshold_metrics["threshold_name"] == "default_0_50",
        "f1",
    ].iloc[0]


def test_confusion_matrices_match_threshold_metrics(
    modeling_reports,
) -> None:
    confusion_matrices = modeling_reports["confusion_matrices"]
    threshold_metrics = modeling_reports["final_threshold_metrics"].set_index(
        "threshold_name"
    )
    test_records = int(
        modeling_reports["final_probability_metrics"].iloc[0]["test_records"]
    )

    for _, matrix in confusion_matrices.iterrows():
        threshold_name = matrix["threshold_name"]
        matrix_counts = matrix[
            [
                "true_negatives",
                "false_positives",
                "false_negatives",
                "true_positives",
            ]
        ]
        assert int(matrix_counts.sum()) == test_records
        for count_column in matrix_counts.index:
            assert matrix[count_column] == threshold_metrics.loc[
                threshold_name,
                count_column,
            ]


def test_curve_and_lift_reports_are_well_formed(modeling_reports) -> None:
    roc_curve = modeling_reports["roc_curve"]
    precision_recall_curve = modeling_reports["precision_recall_curve"]
    targeting_lift = modeling_reports["targeting_lift"]

    roc_rate_columns = roc_curve[["false_positive_rate", "true_positive_rate"]]
    precision_recall_columns = precision_recall_curve[["precision", "recall"]]

    assert roc_rate_columns.ge(0).all().all()
    assert roc_rate_columns.le(1).all().all()
    assert precision_recall_columns.ge(0).all().all()
    assert precision_recall_columns.le(1).all().all()
    assert targeting_lift["targeting_fraction"].is_monotonic_increasing
    assert targeting_lift["targeted_clients"].is_monotonic_increasing
    assert targeting_lift.loc[
        targeting_lift["targeting_fraction"] == 1.0,
        "lift_over_random_targeting",
    ].iloc[0] == pytest.approx(1.0)


def test_permutation_importance_matches_model_feature_contract(
    modeling_reports,
) -> None:
    permutation_importance = modeling_reports["permutation_importance"]

    assert len(permutation_importance) == len(features.MODEL_FEATURES)
    assert set(permutation_importance["feature"]) == set(features.MODEL_FEATURES)
    assert not set(features.CURRENT_CAMPAIGN_COLUMNS) & set(
        permutation_importance["feature"]
    )
    assert permutation_importance["importance_rank"].tolist() == list(
        range(1, len(features.MODEL_FEATURES) + 1)
    )


def test_final_model_artifact_matches_metadata(project_root, modeling_json) -> None:
    metadata = modeling_json["model_metadata"]
    model_path = project_root / metadata["model_file"]

    if not model_path.exists():
        pytest.skip("Generated final model artifact is not present.")

    assert calculate_sha256(model_path) == metadata["model_sha256"]

    artifact = joblib.load(model_path)

    assert set(artifact) >= {
        "pipeline",
        "threshold",
        "model_features",
        "target_column",
        "positive_class",
        "excluded_current_campaign_features",
    }
    assert artifact["threshold"] == pytest.approx(metadata["selected_threshold"])
    assert artifact["model_features"] == features.MODEL_FEATURES
    assert artifact["target_column"] == features.TARGET_COLUMN
    assert set(artifact["excluded_current_campaign_features"]) == set(
        features.CURRENT_CAMPAIGN_COLUMNS
    )
