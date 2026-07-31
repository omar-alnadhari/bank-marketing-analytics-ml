"""Validate the raw UCI Bank Marketing dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank-full.csv"

EXPECTED_SHAPE = (45_211, 17)

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

EXPECTED_TARGET_COUNTS = {
    "no": 39_922,
    "yes": 5_289,
}


def load_raw_data() -> pd.DataFrame:
    """Load the semicolon-separated raw dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run: python src\\download_data.py"
        )

    return pd.read_csv(DATA_PATH, sep=";")


def build_target_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return target counts and percentages."""
    target_counts = (
        data["y"]
        .value_counts()
        .reindex(["no", "yes"])
    )

    target_percentages = (
        data["y"]
        .value_counts(normalize=True)
        .mul(100)
        .reindex(["no", "yes"])
        .round(2)
    )

    return pd.DataFrame(
        {
            "count": target_counts,
            "percentage": target_percentages,
        }
    )


def validate_data(data: pd.DataFrame) -> list[str]:
    """Return validation problems found in the raw dataset."""
    problems: list[str] = []

    if data.shape != EXPECTED_SHAPE:
        problems.append(
            f"Unexpected shape: {data.shape}. "
            f"Expected: {EXPECTED_SHAPE}."
        )

    if data.columns.tolist() != EXPECTED_COLUMNS:
        problems.append(
            "The dataset columns or their order do not match "
            "the expected UCI bank-full.csv schema."
        )

    actual_target_values = set(data["y"].dropna().unique())
    expected_target_values = {"no", "yes"}

    if actual_target_values != expected_target_values:
        problems.append(
            f"Unexpected target values: {actual_target_values}. "
            f"Expected: {expected_target_values}."
        )

    actual_target_counts = data["y"].value_counts().to_dict()

    if actual_target_counts != EXPECTED_TARGET_COUNTS:
        problems.append(
            f"Unexpected target counts: {actual_target_counts}. "
            f"Expected: {EXPECTED_TARGET_COUNTS}."
        )

    return problems


def main() -> None:
    """Load, summarize, and validate the raw dataset."""
    data = load_raw_data()
    target_summary = build_target_summary(data)
    problems = validate_data(data)

    print("=" * 70)
    print("UCI BANK MARKETING — RAW DATA VALIDATION")
    print("=" * 70)

    print(f"\nDataset path:\n{DATA_PATH}")

    print(f"\nShape:\n{data.shape}")

    print("\nColumns:")
    for column_number, column_name in enumerate(
        data.columns,
        start=1,
    ):
        print(f"{column_number:>2}. {column_name}")

    print("\nTarget distribution:")
    print(target_summary.to_string())

    print(
        "\nPandas missing cells:",
        int(data.isna().sum().sum()),
    )

    print("\nFirst three records:")
    print(data.head(3).to_string(index=False))

    if problems:
        print("\nVALIDATION FAILED:")

        for problem in problems:
            print(f"- {problem}")

        raise SystemExit(1)

    print("\nRaw-data validation passed successfully.")


if __name__ == "__main__":
    main()