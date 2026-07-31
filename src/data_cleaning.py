"""Assess and clean the UCI Bank Marketing dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bank-full.csv"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bank_marketing_clean.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

OVERVIEW_REPORT_PATH = (
    REPORTS_DIR
    / "data_quality_overview.csv"
)

UNKNOWN_REPORT_PATH = (
    REPORTS_DIR
    / "unknown_values_summary.csv"
)

OUTLIER_REPORT_PATH = (
    REPORTS_DIR
    / "numeric_outlier_summary.csv"
)


EXPECTED_COLUMNS = [
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
    "y",
]


CATEGORICAL_COLUMNS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]


NUMERIC_COLUMNS = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]


TARGET_SOURCE_COLUMN = "y"
TARGET_COLUMN = "subscribed"


def load_raw_data(
    path: Path = RAW_DATA_PATH,
) -> pd.DataFrame:
    """Load the semicolon-separated raw dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {path}\n"
            "Run: python src\\download_data.py"
        )

    return pd.read_csv(path, sep=";")


def validate_raw_schema(
    data: pd.DataFrame,
) -> None:
    """Validate the expected raw columns and target values."""
    actual_columns = data.columns.tolist()

    if actual_columns != EXPECTED_COLUMNS:
        missing_columns = sorted(
            set(EXPECTED_COLUMNS) - set(actual_columns)
        )

        unexpected_columns = sorted(
            set(actual_columns) - set(EXPECTED_COLUMNS)
        )

        raise ValueError(
            "Unexpected raw-data schema.\n"
            f"Missing columns: {missing_columns}\n"
            f"Unexpected columns: {unexpected_columns}"
        )

    target_values = set(
        data[TARGET_SOURCE_COLUMN]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    expected_target_values = {"yes", "no"}

    if target_values != expected_target_values:
        raise ValueError(
            f"Unexpected target values: {target_values}. "
            f"Expected: {expected_target_values}."
        )


def validate_numeric_ranges(
    data: pd.DataFrame,
) -> None:
    """Check for impossible or structurally invalid values."""
    problems: list[str] = []

    if (data["age"] <= 0).any():
        problems.append(
            "The age column contains values less than "
            "or equal to zero."
        )

    if not data["day"].between(1, 31).all():
        problems.append(
            "The day column contains values outside 1-31."
        )

    if (data["duration"] < 0).any():
        problems.append(
            "The duration column contains negative values."
        )

    if (data["campaign"] < 1).any():
        problems.append(
            "The campaign column contains values below 1."
        )

    if (data["pdays"] < -1).any():
        problems.append(
            "The pdays column contains values below -1."
        )

    if (data["previous"] < 0).any():
        problems.append(
            "The previous column contains negative values."
        )

    if problems:
        formatted_problems = "\n".join(
            f"- {problem}"
            for problem in problems
        )

        raise ValueError(
            "Invalid numeric values were detected:\n"
            f"{formatted_problems}"
        )


def summarize_missing_values(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize standard Pandas missing values."""
    missing_counts = data.isna().sum()

    missing_percentages = (
        missing_counts
        .div(len(data))
        .mul(100)
        .round(2)
    )

    summary = pd.DataFrame(
        {
            "column": data.columns,
            "missing_count": missing_counts.values,
            "missing_percentage": (
                missing_percentages.values
            ),
        }
    )

    return summary


def summarize_unknown_values(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Count explicit 'unknown' categories."""
    records: list[dict[str, object]] = []

    for column in CATEGORICAL_COLUMNS:
        normalized_values = (
            data[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        unknown_count = int(
            normalized_values.eq("unknown").sum()
        )

        unknown_percentage = round(
            unknown_count / len(data) * 100,
            2,
        )

        records.append(
            {
                "column": column,
                "unknown_count": unknown_count,
                "unknown_percentage": (
                    unknown_percentage
                ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            by="unknown_count",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def summarize_numeric_outliers(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Identify potential outliers using the IQR rule."""
    records: list[dict[str, object]] = []

    for column in NUMERIC_COLUMNS:
        numeric_values = pd.to_numeric(
            data[column],
            errors="coerce",
        ).dropna()

        first_quartile = numeric_values.quantile(0.25)
        third_quartile = numeric_values.quantile(0.75)

        interquartile_range = (
            third_quartile - first_quartile
        )

        lower_bound = (
            first_quartile
            - 1.5 * interquartile_range
        )

        upper_bound = (
            third_quartile
            + 1.5 * interquartile_range
        )

        outlier_mask = (
            (numeric_values < lower_bound)
            | (numeric_values > upper_bound)
        )

        outlier_count = int(outlier_mask.sum())

        outlier_percentage = round(
            outlier_count
            / len(numeric_values)
            * 100,
            2,
        )

        records.append(
            {
                "column": column,
                "q1": round(first_quartile, 2),
                "q3": round(third_quartile, 2),
                "iqr": round(interquartile_range, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "potential_outlier_count": (
                    outlier_count
                ),
                "potential_outlier_percentage": (
                    outlier_percentage
                ),
            }
        )

    return pd.DataFrame(records)


def normalize_text_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize categorical and target text values."""
    normalized_data = data.copy()

    text_columns = (
        CATEGORICAL_COLUMNS
        + [TARGET_SOURCE_COLUMN]
    )

    for column in text_columns:
        normalized_data[column] = (
            normalized_data[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    return normalized_data


def clean_data(
    raw_data: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Apply conservative, reproducible cleaning rules."""
    validate_raw_schema(raw_data)

    cleaned_data = raw_data.copy()

    cleaned_data.columns = [
        column.strip().lower()
        for column in cleaned_data.columns
    ]

    cleaned_data = normalize_text_columns(
        cleaned_data
    )

    for column in NUMERIC_COLUMNS:
        cleaned_data[column] = pd.to_numeric(
            cleaned_data[column],
            errors="raise",
        )

    validate_numeric_ranges(cleaned_data)

    duplicate_count = int(
        cleaned_data.duplicated().sum()
    )

    cleaned_data = (
        cleaned_data
        .drop_duplicates()
        .reset_index(drop=True)
    )

    cleaned_data[TARGET_COLUMN] = (
        cleaned_data[TARGET_SOURCE_COLUMN]
        .map(
            {
                "no": 0,
                "yes": 1,
            }
        )
    )

    if cleaned_data[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Target encoding produced missing values."
        )

    cleaned_data[TARGET_COLUMN] = (
        cleaned_data[TARGET_COLUMN]
        .astype("int8")
    )

    cleaned_data = cleaned_data.drop(
        columns=[TARGET_SOURCE_COLUMN]
    )

    return cleaned_data, duplicate_count


def build_overview_report(
    raw_data: pd.DataFrame,
    cleaned_data: pd.DataFrame,
    duplicate_count: int,
    unknown_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build a high-level data-quality report."""
    target_counts = (
        cleaned_data[TARGET_COLUMN]
        .value_counts()
        .to_dict()
    )

    total_unknown_values = int(
        unknown_summary["unknown_count"].sum()
    )

    report_values = {
        "raw_rows": len(raw_data),
        "raw_columns": raw_data.shape[1],
        "cleaned_rows": len(cleaned_data),
        "cleaned_columns": cleaned_data.shape[1],
        "exact_duplicates_found": (
            duplicate_count
        ),
        "exact_duplicates_removed": (
            duplicate_count
        ),
        "raw_missing_cells": int(
            raw_data.isna().sum().sum()
        ),
        "cleaned_missing_cells": int(
            cleaned_data.isna().sum().sum()
        ),
        "explicit_unknown_values": (
            total_unknown_values
        ),
        "non_subscribers": int(
            target_counts.get(0, 0)
        ),
        "subscribers": int(
            target_counts.get(1, 0)
        ),
    }

    return pd.DataFrame(
        {
            "metric": report_values.keys(),
            "value": report_values.values(),
        }
    )


def save_outputs(
    cleaned_data: pd.DataFrame,
    overview_report: pd.DataFrame,
    unknown_report: pd.DataFrame,
    outlier_report: pd.DataFrame,
) -> None:
    """Save cleaned data and generated reports."""
    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned_data.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    overview_report.to_csv(
        OVERVIEW_REPORT_PATH,
        index=False,
    )

    unknown_report.to_csv(
        UNKNOWN_REPORT_PATH,
        index=False,
    )

    outlier_report.to_csv(
        OUTLIER_REPORT_PATH,
        index=False,
    )


def main() -> None:
    """Run data-quality assessment and cleaning."""
    raw_data = load_raw_data()

    validate_raw_schema(raw_data)

    missing_summary = summarize_missing_values(
        raw_data
    )

    unknown_summary = summarize_unknown_values(
        raw_data
    )

    outlier_summary = summarize_numeric_outliers(
        raw_data
    )

    cleaned_data, duplicate_count = clean_data(
        raw_data
    )

    overview_report = build_overview_report(
        raw_data=raw_data,
        cleaned_data=cleaned_data,
        duplicate_count=duplicate_count,
        unknown_summary=unknown_summary,
    )

    save_outputs(
        cleaned_data=cleaned_data,
        overview_report=overview_report,
        unknown_report=unknown_summary,
        outlier_report=outlier_summary,
    )

    print("=" * 70)
    print("BANK MARKETING — DATA QUALITY AND CLEANING")
    print("=" * 70)

    print(f"\nRaw shape: {raw_data.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")

    print(
        "\nExact duplicate rows found:",
        duplicate_count,
    )

    print(
        "Pandas missing cells:",
        int(missing_summary["missing_count"].sum()),
    )

    print("\nExplicit 'unknown' values:")
    print(
        unknown_summary.to_string(
            index=False
        )
    )

    print("\nPotential numeric outliers using IQR:")
    print(
        outlier_summary[
            [
                "column",
                "potential_outlier_count",
                "potential_outlier_percentage",
            ]
        ].to_string(index=False)
    )

    print("\nCleaned target distribution:")
    target_distribution = (
        cleaned_data[TARGET_COLUMN]
        .value_counts()
        .rename_axis(TARGET_COLUMN)
        .reset_index(name="count")
        .sort_values(TARGET_COLUMN)
    )

    target_distribution["percentage"] = (
        target_distribution["count"]
        .div(len(cleaned_data))
        .mul(100)
        .round(2)
    )

    print(
        target_distribution.to_string(
            index=False
        )
    )

    print("\nGenerated files:")
    print(f"- {PROCESSED_DATA_PATH}")
    print(f"- {OVERVIEW_REPORT_PATH}")
    print(f"- {UNKNOWN_REPORT_PATH}")
    print(f"- {OUTLIER_REPORT_PATH}")

    print(
        "\nImportant decisions:"
        "\n- Explicit 'unknown' values were retained."
        "\n- Potential outliers were reported but not removed."
        "\n- The duration column was retained for EDA."
        "\n- The duration column will be excluded from "
        "the deployable pre-call model."
        "\n- The pdays value -1 was retained as a "
        "meaningful sentinel value."
    )

    print(
        "\nData-quality assessment and cleaning "
        "completed successfully."
    )


if __name__ == "__main__":
    main()
