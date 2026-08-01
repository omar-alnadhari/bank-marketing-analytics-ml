"""Run statistical analyses for the Bank Marketing dataset."""

from __future__ import annotations

from statistics import NormalDist
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bank_marketing_clean.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "statistics"
)

TARGET_COLUMN = "subscribed"

EXPECTED_ROW_COUNT = 45_211
EXPECTED_SUBSCRIBERS = 5_289
EXPECTED_NON_SUBSCRIBERS = 39_922

ALPHA = 0.05
CONFIDENCE_LEVEL = 0.95


CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
    "previously_contacted",
]


NUMERICAL_TEST_SPECIFICATIONS = [
    {
        "feature": "age",
        "analysis_name": "age",
        "subset": "all_clients",
        "leakage_sensitive": False,
    },
    {
        "feature": "balance",
        "analysis_name": "balance",
        "subset": "all_clients",
        "leakage_sensitive": False,
    },
    {
        "feature": "duration",
        "analysis_name": "duration",
        "subset": "all_clients",
        "leakage_sensitive": True,
    },
    {
        "feature": "campaign",
        "analysis_name": "campaign",
        "subset": "all_clients",
        "leakage_sensitive": False,
    },
    {
        "feature": "previous",
        "analysis_name": "previous",
        "subset": "all_clients",
        "leakage_sensitive": False,
    },
    {
        "feature": "pdays",
        "analysis_name": "pdays_previously_contacted",
        "subset": "previously_contacted_clients",
        "leakage_sensitive": False,
    },
]


def load_analysis_data(
    path: Path = DATA_PATH,
) -> pd.DataFrame:
    """Load and validate the cleaned analytical dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {path}\n"
            "Run: python src\\data_cleaning.py"
        )

    data = pd.read_csv(path)

    if len(data) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Unexpected row count: {len(data)}. "
            f"Expected: {EXPECTED_ROW_COUNT}."
        )

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Target column not found: {TARGET_COLUMN}"
        )

    if set(data[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError(
            "The target column must contain only 0 and 1."
        )

    if data.isna().any().any():
        raise ValueError(
            "The analytical dataset contains missing values."
        )

    analysis_data = data.copy()

    analysis_data["previously_contacted"] = np.where(
        analysis_data["pdays"] == -1,
        "no",
        "yes",
    )

    return analysis_data


def benjamini_hochberg(
    p_values: pd.Series,
) -> np.ndarray:
    """Adjust p-values using the Benjamini-Hochberg procedure."""
    values = np.asarray(
        p_values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "P-values must be one-dimensional."
        )

    if len(values) == 0:
        return np.array([], dtype=float)

    if (
        np.isnan(values).any()
        or (values < 0).any()
        or (values > 1).any()
    ):
        raise ValueError(
            "All p-values must be valid values between 0 and 1."
        )

    number_of_tests = len(values)

    sort_order = np.argsort(values)
    sorted_p_values = values[sort_order]

    ranks = np.arange(
        1,
        number_of_tests + 1,
    )

    adjusted_sorted = (
        sorted_p_values
        * number_of_tests
        / ranks
    )

    adjusted_sorted = np.minimum.accumulate(
        adjusted_sorted[::-1]
    )[::-1]

    adjusted_sorted = np.clip(
        adjusted_sorted,
        0,
        1,
    )

    adjusted_values = np.empty_like(
        adjusted_sorted
    )

    adjusted_values[sort_order] = (
        adjusted_sorted
    )

    return adjusted_values


def classify_effect_size(
    effect_size: float,
) -> str:
    """Provide a practical descriptive effect-size label."""
    absolute_effect = abs(effect_size)

    if absolute_effect < 0.10:
        return "Negligible"

    if absolute_effect < 0.30:
        return "Small"

    if absolute_effect < 0.50:
        return "Moderate"

    return "Large"


def calculate_cramers_v(
    chi_square_statistic: float,
    contingency_table: pd.DataFrame,
) -> float:
    """Calculate Cramér's V from a contingency table."""
    sample_size = contingency_table.to_numpy().sum()

    number_of_rows = contingency_table.shape[0]
    number_of_columns = contingency_table.shape[1]

    minimum_dimension = min(
        number_of_rows - 1,
        number_of_columns - 1,
    )

    if sample_size <= 0 or minimum_dimension <= 0:
        raise ValueError(
            "Cramér's V requires a non-empty "
            "table with at least two categories."
        )

    return float(
        np.sqrt(
            chi_square_statistic
            / (
                sample_size
                * minimum_dimension
            )
        )
    )


def run_categorical_association_tests(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Run chi-square tests for categorical features."""
    result_records: list[dict[str, object]] = []

    for feature in CATEGORICAL_FEATURES:
        contingency_table = pd.crosstab(
            data[feature],
            data[TARGET_COLUMN],
            dropna=False,
        )

        (
            chi_square_statistic,
            p_value,
            degrees_of_freedom,
            expected_frequencies,
        ) = chi2_contingency(
            contingency_table,
            correction=False,
        )

        cramers_v = calculate_cramers_v(
            chi_square_statistic,
            contingency_table,
        )

        expected_frequencies = np.asarray(
            expected_frequencies,
            dtype=float,
        )

        cells_below_five = int(
            (expected_frequencies < 5).sum()
        )

        percentage_below_five = round(
            100
            * cells_below_five
            / expected_frequencies.size,
            2,
        )

        minimum_expected_frequency = float(
            expected_frequencies.min()
        )

        assumption_warning = bool(
            minimum_expected_frequency < 1
            or percentage_below_five > 20
        )

        result_records.append(
            {
                "feature": feature,
                "number_of_categories": (
                    contingency_table.shape[0]
                ),
                "sample_size": int(
                    contingency_table.to_numpy().sum()
                ),
                "chi_square_statistic": float(
                    chi_square_statistic
                ),
                "degrees_of_freedom": int(
                    degrees_of_freedom
                ),
                "p_value": float(p_value),
                "cramers_v": cramers_v,
                "effect_size_interpretation": (
                    classify_effect_size(cramers_v)
                ),
                "minimum_expected_frequency": (
                    minimum_expected_frequency
                ),
                "expected_cells_below_five": (
                    cells_below_five
                ),
                "expected_cells_below_five_percentage": (
                    percentage_below_five
                ),
                "assumption_warning": assumption_warning,
            }
        )

    results = pd.DataFrame(result_records)

    results["adjusted_p_value_bh"] = (
        benjamini_hochberg(
            results["p_value"]
        )
    )

    results["reject_null_fdr_0_05"] = (
        results["adjusted_p_value_bh"]
        < ALPHA
    )

    results = (
        results
        .sort_values(
            "cramers_v",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return results


def get_numerical_test_subset(
    data: pd.DataFrame,
    subset_name: str,
) -> pd.DataFrame:
    """Return the required subset for a numerical test."""
    if subset_name == "all_clients":
        return data

    if subset_name == "previously_contacted_clients":
        subset = data.loc[
            data["pdays"] >= 0
        ].copy()

        if subset.empty:
            raise ValueError(
                "No previously contacted clients were found."
            )

        return subset

    raise ValueError(
        f"Unknown numerical-test subset: {subset_name}"
    )


def run_numerical_group_tests(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Run Mann-Whitney U tests for numerical features."""
    result_records: list[dict[str, object]] = []

    for specification in NUMERICAL_TEST_SPECIFICATIONS:
        feature = str(specification["feature"])
        analysis_name = str(
            specification["analysis_name"]
        )
        subset_name = str(specification["subset"])

        test_data = get_numerical_test_subset(
            data,
            subset_name,
        )

        subscriber_values = (
            test_data.loc[
                test_data[TARGET_COLUMN] == 1,
                feature,
            ]
            .astype(float)
            .to_numpy()
        )

        non_subscriber_values = (
            test_data.loc[
                test_data[TARGET_COLUMN] == 0,
                feature,
            ]
            .astype(float)
            .to_numpy()
        )

        if (
            len(subscriber_values) == 0
            or len(non_subscriber_values) == 0
        ):
            raise ValueError(
                f"Feature '{analysis_name}' has an empty group."
            )

        test_result = mannwhitneyu(
            subscriber_values,
            non_subscriber_values,
            alternative="two-sided",
            method="asymptotic",
        )

        u_statistic = float(
            test_result.statistic
        )

        rank_biserial_correlation = (
            2
            * u_statistic
            / (
                len(subscriber_values)
                * len(non_subscriber_values)
            )
            - 1
        )

        subscriber_median = float(
            np.median(subscriber_values)
        )

        non_subscriber_median = float(
            np.median(non_subscriber_values)
        )

        median_difference = (
            subscriber_median
            - non_subscriber_median
        )

        if median_difference > 0:
            median_direction = (
                "Higher among subscribers"
            )
        elif median_difference < 0:
            median_direction = (
                "Lower among subscribers"
            )
        else:
            median_direction = "Equal medians"

        result_records.append(
            {
                "analysis_name": analysis_name,
                "source_feature": feature,
                "subset": subset_name,
                "leakage_sensitive": bool(
                    specification["leakage_sensitive"]
                ),
                "subscriber_sample_size": len(
                    subscriber_values
                ),
                "non_subscriber_sample_size": len(
                    non_subscriber_values
                ),
                "subscriber_mean": float(
                    np.mean(subscriber_values)
                ),
                "non_subscriber_mean": float(
                    np.mean(non_subscriber_values)
                ),
                "subscriber_median": (
                    subscriber_median
                ),
                "non_subscriber_median": (
                    non_subscriber_median
                ),
                "median_difference": (
                    median_difference
                ),
                "median_direction": median_direction,
                "mann_whitney_u": u_statistic,
                "p_value": float(
                    test_result.pvalue
                ),
                "rank_biserial_correlation": (
                    rank_biserial_correlation
                ),
                "effect_size_interpretation": (
                    classify_effect_size(
                        rank_biserial_correlation
                    )
                ),
            }
        )

    results = pd.DataFrame(result_records)

    results["adjusted_p_value_bh"] = (
        benjamini_hochberg(
            results["p_value"]
        )
    )

    results["reject_null_fdr_0_05"] = (
        results["adjusted_p_value_bh"]
        < ALPHA
    )

    results["absolute_effect_size"] = (
        results[
            "rank_biserial_correlation"
        ].abs()
    )

    results = (
        results
        .sort_values(
            "absolute_effect_size",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return results


def wilson_proportion_interval(
    successes: int,
    total: int,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> tuple[float, float, float]:
    """Calculate a Wilson confidence interval for a proportion."""
    if total <= 0:
        raise ValueError(
            "The total sample size must be positive."
        )

    if successes < 0 or successes > total:
        raise ValueError(
            "Successes must be between zero and total."
        )

    alpha = 1 - confidence_level

    z_score = NormalDist().inv_cdf(
        1 - alpha / 2
    )

    estimated_proportion = successes / total

    z_squared = z_score**2

    denominator = 1 + z_squared / total

    center = (
        estimated_proportion
        + z_squared / (2 * total)
    ) / denominator

    margin = (
        z_score
        / denominator
        * np.sqrt(
            estimated_proportion
            * (1 - estimated_proportion)
            / total
            + z_squared
            / (4 * total**2)
        )
    )

    return (
        estimated_proportion,
        center - margin,
        center + margin,
    )


def difference_in_proportions_interval(
    data: pd.DataFrame,
    comparison_name: str,
    group_a_name: str,
    group_a_mask: pd.Series,
    group_b_name: str,
    group_b_mask: pd.Series,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> dict[str, object]:
    """Calculate a large-sample CI for a difference in proportions."""
    if not group_a_mask.index.equals(data.index):
        raise ValueError(
            "Group A mask does not match the dataset index."
        )

    if not group_b_mask.index.equals(data.index):
        raise ValueError(
            "Group B mask does not match the dataset index."
        )

    if (group_a_mask & group_b_mask).any():
        raise ValueError(
            f"Comparison groups overlap: {comparison_name}"
        )

    group_a_target = data.loc[
        group_a_mask,
        TARGET_COLUMN,
    ]

    group_b_target = data.loc[
        group_b_mask,
        TARGET_COLUMN,
    ]

    group_a_size = len(group_a_target)
    group_b_size = len(group_b_target)

    if group_a_size == 0 or group_b_size == 0:
        raise ValueError(
            f"An empty comparison group was found: "
            f"{comparison_name}"
        )

    group_a_subscribers = int(
        group_a_target.sum()
    )

    group_b_subscribers = int(
        group_b_target.sum()
    )

    group_a_rate = (
        group_a_subscribers / group_a_size
    )

    group_b_rate = (
        group_b_subscribers / group_b_size
    )

    rate_difference = (
        group_a_rate - group_b_rate
    )

    alpha = 1 - confidence_level

    z_score = NormalDist().inv_cdf(
        1 - alpha / 2
    )

    standard_error = np.sqrt(
        group_a_rate
        * (1 - group_a_rate)
        / group_a_size
        + group_b_rate
        * (1 - group_b_rate)
        / group_b_size
    )

    lower_bound = (
        rate_difference
        - z_score * standard_error
    )

    upper_bound = (
        rate_difference
        + z_score * standard_error
    )

    return {
        "comparison": comparison_name,
        "group_a": group_a_name,
        "group_b": group_b_name,
        "group_a_size": group_a_size,
        "group_b_size": group_b_size,
        "group_a_subscribers": (
            group_a_subscribers
        ),
        "group_b_subscribers": (
            group_b_subscribers
        ),
        "group_a_rate_percentage": (
            group_a_rate * 100
        ),
        "group_b_rate_percentage": (
            group_b_rate * 100
        ),
        "difference_percentage_points": (
            rate_difference * 100
        ),
        "ci_lower_percentage_points": (
            lower_bound * 100
        ),
        "ci_upper_percentage_points": (
            upper_bound * 100
        ),
        "confidence_level": confidence_level,
    }


def build_confidence_interval_reports(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build overall and comparison confidence intervals."""
    total_clients = len(data)

    subscribers = int(
        data[TARGET_COLUMN].sum()
    )

    (
        estimated_rate,
        lower_rate,
        upper_rate,
    ) = wilson_proportion_interval(
        successes=subscribers,
        total=total_clients,
    )

    overall_interval = pd.DataFrame(
        [
            {
                "total_clients": total_clients,
                "subscribers": subscribers,
                "estimated_subscription_rate_percentage": (
                    estimated_rate * 100
                ),
                "ci_lower_percentage": (
                    lower_rate * 100
                ),
                "ci_upper_percentage": (
                    upper_rate * 100
                ),
                "confidence_level": (
                    CONFIDENCE_LEVEL
                ),
                "method": "Wilson score interval",
            }
        ]
    )

    comparison_records = [
        difference_in_proportions_interval(
            data=data,
            comparison_name=(
                "Previous success vs other outcomes"
            ),
            group_a_name="Previous outcome: success",
            group_a_mask=(
                data["poutcome"] == "success"
            ),
            group_b_name="Other previous outcomes",
            group_b_mask=(
                data["poutcome"] != "success"
            ),
        ),
        difference_in_proportions_interval(
            data=data,
            comparison_name=(
                "Cellular vs unknown contact"
            ),
            group_a_name="Cellular contact",
            group_a_mask=(
                data["contact"] == "cellular"
            ),
            group_b_name="Unknown contact",
            group_b_mask=(
                data["contact"] == "unknown"
            ),
        ),
        difference_in_proportions_interval(
            data=data,
            comparison_name=(
                "One campaign contact vs multiple contacts"
            ),
            group_a_name="One campaign contact",
            group_a_mask=(
                data["campaign"] == 1
            ),
            group_b_name="Multiple campaign contacts",
            group_b_mask=(
                data["campaign"] > 1
            ),
        ),
    ]

    comparison_intervals = pd.DataFrame(
        comparison_records
    )

    return (
        overall_interval,
        comparison_intervals,
    )


def save_results(
    categorical_results: pd.DataFrame,
    numerical_results: pd.DataFrame,
    overall_interval: pd.DataFrame,
    comparison_intervals: pd.DataFrame,
) -> None:
    """Save statistical-analysis outputs."""
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    categorical_results.to_csv(
        OUTPUT_DIRECTORY
        / "categorical_association_tests.csv",
        index=False,
    )

    numerical_results.to_csv(
        OUTPUT_DIRECTORY
        / "numerical_group_tests.csv",
        index=False,
    )

    overall_interval.to_csv(
        OUTPUT_DIRECTORY
        / "overall_subscription_rate_ci.csv",
        index=False,
    )

    comparison_intervals.to_csv(
        OUTPUT_DIRECTORY
        / "proportion_difference_confidence_intervals.csv",
        index=False,
    )


def validate_results(
    categorical_results: pd.DataFrame,
    numerical_results: pd.DataFrame,
    overall_interval: pd.DataFrame,
    comparison_intervals: pd.DataFrame,
) -> None:
    """Validate the generated statistical results."""
    if len(categorical_results) != len(
        CATEGORICAL_FEATURES
    ):
        raise ValueError(
            "Unexpected number of categorical tests."
        )

    if len(numerical_results) != len(
        NUMERICAL_TEST_SPECIFICATIONS
    ):
        raise ValueError(
            "Unexpected number of numerical tests."
        )

    if len(overall_interval) != 1:
        raise ValueError(
            "The overall confidence-interval report "
            "must contain one row."
        )

    if len(comparison_intervals) != 3:
        raise ValueError(
            "Unexpected number of proportion comparisons."
        )

    p_value_columns = [
        categorical_results["p_value"],
        categorical_results[
            "adjusted_p_value_bh"
        ],
        numerical_results["p_value"],
        numerical_results[
            "adjusted_p_value_bh"
        ],
    ]

    for p_values in p_value_columns:
        if not p_values.between(0, 1).all():
            raise ValueError(
                "Invalid p-values were detected."
            )

    if not categorical_results[
        "cramers_v"
    ].between(0, 1).all():
        raise ValueError(
            "Invalid Cramér's V values were detected."
        )

    if not numerical_results[
        "rank_biserial_correlation"
    ].between(-1, 1).all():
        raise ValueError(
            "Invalid rank-biserial correlations "
            "were detected."
        )

    overall_result = overall_interval.iloc[0]

    if int(
        overall_result["total_clients"]
    ) != EXPECTED_ROW_COUNT:
        raise ValueError(
            "Unexpected overall sample size."
        )

    if int(
        overall_result["subscribers"]
    ) != EXPECTED_SUBSCRIBERS:
        raise ValueError(
            "Unexpected subscriber count."
        )

    estimated_rate = float(
        overall_result[
            "estimated_subscription_rate_percentage"
        ]
    )

    lower_bound = float(
        overall_result["ci_lower_percentage"]
    )

    upper_bound = float(
        overall_result["ci_upper_percentage"]
    )

    if not (
        lower_bound
        <= estimated_rate
        <= upper_bound
    ):
        raise ValueError(
            "The estimated rate is outside "
            "its confidence interval."
        )

    if not (
        comparison_intervals[
            "ci_lower_percentage_points"
        ]
        <= comparison_intervals[
            "difference_percentage_points"
        ]
    ).all():
        raise ValueError(
            "Invalid lower confidence bounds."
        )

    if not (
        comparison_intervals[
            "difference_percentage_points"
        ]
        <= comparison_intervals[
            "ci_upper_percentage_points"
        ]
    ).all():
        raise ValueError(
            "Invalid upper confidence bounds."
        )


def display_results(
    categorical_results: pd.DataFrame,
    numerical_results: pd.DataFrame,
    overall_interval: pd.DataFrame,
    comparison_intervals: pd.DataFrame,
) -> None:
    """Display concise statistical-analysis results."""
    print("\nCategorical association tests:")
    print(
        categorical_results[
            [
                "feature",
                "chi_square_statistic",
                "adjusted_p_value_bh",
                "cramers_v",
                "effect_size_interpretation",
                "reject_null_fdr_0_05",
                "assumption_warning",
            ]
        ].to_string(index=False)
    )

    print("\nNumerical group comparisons:")
    print(
        numerical_results[
            [
                "analysis_name",
                "subscriber_median",
                "non_subscriber_median",
                "median_difference",
                "adjusted_p_value_bh",
                "rank_biserial_correlation",
                "effect_size_interpretation",
                "leakage_sensitive",
            ]
        ].to_string(index=False)
    )

    print("\nOverall subscription-rate confidence interval:")
    print(
        overall_interval.to_string(
            index=False
        )
    )

    print("\nSelected differences in proportions:")
    print(
        comparison_intervals[
            [
                "comparison",
                "group_a_rate_percentage",
                "group_b_rate_percentage",
                "difference_percentage_points",
                "ci_lower_percentage_points",
                "ci_upper_percentage_points",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    """Run the complete statistical-analysis workflow."""
    print("=" * 70)
    print("BANK MARKETING — STATISTICAL ANALYSIS")
    print("=" * 70)

    data = load_analysis_data()

    print(f"\nAnalysis dataset shape: {data.shape}")

    print(
        "Subscribers:",
        int(data[TARGET_COLUMN].sum()),
    )

    print(
        "Non-subscribers:",
        int(
            (data[TARGET_COLUMN] == 0).sum()
        ),
    )

    categorical_results = (
        run_categorical_association_tests(data)
    )

    numerical_results = (
        run_numerical_group_tests(data)
    )

    (
        overall_interval,
        comparison_intervals,
    ) = build_confidence_interval_reports(data)

    validate_results(
        categorical_results=categorical_results,
        numerical_results=numerical_results,
        overall_interval=overall_interval,
        comparison_intervals=comparison_intervals,
    )

    save_results(
        categorical_results=categorical_results,
        numerical_results=numerical_results,
        overall_interval=overall_interval,
        comparison_intervals=comparison_intervals,
    )

    display_results(
        categorical_results=categorical_results,
        numerical_results=numerical_results,
        overall_interval=overall_interval,
        comparison_intervals=comparison_intervals,
    )

    print("\nGenerated statistical reports:")

    for output_path in sorted(
        OUTPUT_DIRECTORY.glob("*.csv")
    ):
        print(f"- {output_path}")

    print(
        "\nAll statistical analyses executed "
        "and validated successfully."
    )


if __name__ == "__main__":
    main()
