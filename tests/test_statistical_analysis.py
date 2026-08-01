"""Tests for statistical-analysis helpers and saved reports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import statistical_analysis as stats


def build_statistical_sample() -> pd.DataFrame:
    """Create a small analytical dataset for deterministic tests."""
    return pd.DataFrame(
        {
            "subscribed": [0, 1, 0, 1, 0, 1, 0, 1],
            "age": [30, 42, 36, 55, 24, 61, 45, 39],
            "balance": [100, 800, -50, 1200, 20, 2300, 300, 700],
            "duration": [60, 300, 80, 250, 40, 500, 90, 420],
            "campaign": [2, 1, 3, 1, 4, 1, 2, 1],
            "previous": [0, 2, 0, 3, 1, 4, 0, 2],
            "pdays": [-1, 10, -1, 200, 0, 45, -1, 15],
            "poutcome": [
                "unknown",
                "success",
                "unknown",
                "failure",
                "other",
                "success",
                "unknown",
                "other",
            ],
            "contact": [
                "unknown",
                "cellular",
                "unknown",
                "cellular",
                "telephone",
                "cellular",
                "unknown",
                "cellular",
            ],
        }
    )


def test_benjamini_hochberg_adjusts_p_values_monotonically() -> None:
    adjusted = stats.benjamini_hochberg(pd.Series([0.01, 0.04, 0.03]))

    np.testing.assert_allclose(adjusted, np.array([0.03, 0.04, 0.04]))


def test_benjamini_hochberg_rejects_invalid_p_values() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        stats.benjamini_hochberg(pd.Series([0.01, -0.2, 0.5]))


@pytest.mark.parametrize(
    ("effect_size", "expected_label"),
    [
        (0.01, "Negligible"),
        (0.15, "Small"),
        (-0.35, "Moderate"),
        (0.75, "Large"),
    ],
)
def test_classify_effect_size(effect_size: float, expected_label: str) -> None:
    assert stats.classify_effect_size(effect_size) == expected_label


def test_wilson_proportion_interval_contains_observed_rate() -> None:
    rate, lower_bound, upper_bound = stats.wilson_proportion_interval(
        successes=12,
        total=100,
    )

    assert rate == 0.12
    assert lower_bound < rate < upper_bound
    assert 0 <= lower_bound <= upper_bound <= 1


def test_wilson_proportion_interval_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="between zero and total"):
        stats.wilson_proportion_interval(successes=5, total=4)


def test_difference_in_proportions_interval_rejects_overlapping_groups() -> None:
    data = build_statistical_sample()
    group_a_mask = data["contact"] == "cellular"
    group_b_mask = data["subscribed"] == 1

    with pytest.raises(ValueError, match="overlap"):
        stats.difference_in_proportions_interval(
            data=data,
            comparison_name="overlapping comparison",
            group_a_name="cellular",
            group_a_mask=group_a_mask,
            group_b_name="subscriber",
            group_b_mask=group_b_mask,
        )


def test_difference_in_proportions_interval_returns_expected_fields() -> None:
    data = build_statistical_sample()
    group_a_mask = data["contact"] == "cellular"
    group_b_mask = data["contact"] == "unknown"

    interval = stats.difference_in_proportions_interval(
        data=data,
        comparison_name="cellular vs unknown",
        group_a_name="cellular",
        group_a_mask=group_a_mask,
        group_b_name="unknown",
        group_b_mask=group_b_mask,
    )

    assert interval["group_a_size"] == 4
    assert interval["group_b_size"] == 3
    assert interval["group_a_subscribers"] == 4
    assert interval["group_b_subscribers"] == 0
    assert interval["difference_percentage_points"] == pytest.approx(100.0)
    assert (
        interval["ci_lower_percentage_points"]
        <= interval["difference_percentage_points"]
        <= interval["ci_upper_percentage_points"]
    )


def test_run_numerical_group_tests_returns_all_configured_tests() -> None:
    results = stats.run_numerical_group_tests(build_statistical_sample())

    assert set(results["analysis_name"]) == {
        specification["analysis_name"]
        for specification in stats.NUMERICAL_TEST_SPECIFICATIONS
    }
    assert results["p_value"].between(0, 1).all()
    assert results["adjusted_p_value_bh"].between(0, 1).all()
    assert results["rank_biserial_correlation"].between(-1, 1).all()
    assert bool(
        results.loc[
            results["analysis_name"] == "duration",
            "leakage_sensitive",
        ].iloc[0]
    ) is True


def test_saved_statistical_reports_match_expected_contract(
    statistics_report_directory,
) -> None:
    expected_report_files = {
        "categorical_association_tests.csv",
        "numerical_group_tests.csv",
        "overall_subscription_rate_ci.csv",
        "proportion_difference_confidence_intervals.csv",
    }

    missing_files = [
        report_file
        for report_file in expected_report_files
        if not (statistics_report_directory / report_file).exists()
    ]

    if missing_files:
        pytest.skip(f"Generated statistical reports are missing: {missing_files}")

    categorical_results = pd.read_csv(
        statistics_report_directory / "categorical_association_tests.csv"
    )
    numerical_results = pd.read_csv(
        statistics_report_directory / "numerical_group_tests.csv"
    )
    overall_interval = pd.read_csv(
        statistics_report_directory / "overall_subscription_rate_ci.csv"
    )
    comparison_intervals = pd.read_csv(
        statistics_report_directory / "proportion_difference_confidence_intervals.csv"
    )

    stats.validate_results(
        categorical_results=categorical_results,
        numerical_results=numerical_results,
        overall_interval=overall_interval,
        comparison_intervals=comparison_intervals,
    )
