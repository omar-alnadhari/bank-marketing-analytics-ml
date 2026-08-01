"""Create and validate the SQLite Bank Marketing database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bank_marketing_clean.csv"
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "bank_marketing.db"
)

SCHEMA_PATH = (
    PROJECT_ROOT
    / "sql"
    / "create_tables.sql"
)

TABLE_NAME = "bank_marketing"


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


DATABASE_COLUMN_MAP = {
    "default": "default_status",
    "housing": "housing_loan",
    "loan": "personal_loan",
    "day": "contact_day",
    "month": "contact_month",
    "duration": "call_duration_seconds",
    "campaign": "campaign_contacts",
    "pdays": "days_since_previous_contact",
    "previous": "previous_contacts",
    "poutcome": "previous_outcome",
}


EXPECTED_DATABASE_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default_status",
    "balance",
    "housing_loan",
    "personal_loan",
    "contact",
    "contact_day",
    "contact_month",
    "call_duration_seconds",
    "campaign_contacts",
    "days_since_previous_contact",
    "previous_contacts",
    "previous_outcome",
    "subscribed",
]


EXPECTED_ROW_COUNT = 45_211
EXPECTED_NON_SUBSCRIBERS = 39_922
EXPECTED_SUBSCRIBERS = 5_289


def load_clean_data(
    path: Path = CLEAN_DATA_PATH,
) -> pd.DataFrame:
    """Load and validate the cleaned CSV dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {path}\n"
            "Run: python src\\data_cleaning.py"
        )

    data = pd.read_csv(path)

    if data.columns.tolist() != EXPECTED_SOURCE_COLUMNS:
        raise ValueError(
            "The cleaned dataset schema does not match "
            "the expected source columns."
        )

    if len(data) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Unexpected row count: {len(data)}. "
            f"Expected: {EXPECTED_ROW_COUNT}."
        )

    if data.isna().any().any():
        raise ValueError(
            "The cleaned dataset contains missing values."
        )

    if set(data["subscribed"].unique()) != {0, 1}:
        raise ValueError(
            "The subscribed column must contain only 0 and 1."
        )

    return data


def prepare_database_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Rename cleaned-data columns for the SQL schema."""
    database_data = data.rename(
        columns=DATABASE_COLUMN_MAP
    ).copy()

    if (
        database_data.columns.tolist()
        != EXPECTED_DATABASE_COLUMNS
    ):
        raise ValueError(
            "The renamed database columns do not match "
            "the expected SQL schema."
        )

    return database_data


def read_schema_sql(
    path: Path = SCHEMA_PATH,
) -> str:
    """Read the SQL schema file."""
    if not path.exists():
        raise FileNotFoundError(
            f"SQL schema file not found: {path}"
        )

    return path.read_text(encoding="utf-8")


def create_database(
    database_data: pd.DataFrame,
) -> None:
    """Create the database schema and load all records."""
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    schema_sql = read_schema_sql()

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            "PRAGMA foreign_keys = ON;"
        )

        connection.executescript(schema_sql)

        database_data.to_sql(
            TABLE_NAME,
            connection,
            if_exists="append",
            index=False,
            chunksize=1_000,
        )


def get_table_columns(
    connection: sqlite3.Connection,
) -> list[str]:
    """Return the ordered database-table column names."""
    table_information = connection.execute(
        f"PRAGMA table_info({TABLE_NAME});"
    ).fetchall()

    return [
        row[1]
        for row in table_information
    ]


def validate_database() -> dict[str, object]:
    """Validate database integrity, schema, and row counts."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        integrity_result = connection.execute(
            "PRAGMA integrity_check;"
        ).fetchone()[0]

        table_columns = get_table_columns(connection)

        expected_table_columns = [
            "campaign_record_id",
            *EXPECTED_DATABASE_COLUMNS,
        ]

        if table_columns != expected_table_columns:
            raise ValueError(
                "Database table columns do not match "
                "the expected schema."
            )

        row_count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {TABLE_NAME};
            """
        ).fetchone()[0]

        non_subscriber_count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {TABLE_NAME}
            WHERE subscribed = 0;
            """
        ).fetchone()[0]

        subscriber_count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {TABLE_NAME}
            WHERE subscribed = 1;
            """
        ).fetchone()[0]

        duplicate_id_count = connection.execute(
            f"""
            SELECT
                COUNT(*) - COUNT(DISTINCT campaign_record_id)
            FROM {TABLE_NAME};
            """
        ).fetchone()[0]

        null_subscribed_count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {TABLE_NAME}
            WHERE subscribed IS NULL;
            """
        ).fetchone()[0]

    validation_results = {
        "integrity_check": integrity_result,
        "row_count": row_count,
        "non_subscribers": non_subscriber_count,
        "subscribers": subscriber_count,
        "duplicate_primary_keys": duplicate_id_count,
        "null_target_values": null_subscribed_count,
    }

    if integrity_result != "ok":
        raise ValueError(
            f"SQLite integrity check failed: {integrity_result}"
        )

    if row_count != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Unexpected database row count: {row_count}."
        )

    if (
        non_subscriber_count
        != EXPECTED_NON_SUBSCRIBERS
    ):
        raise ValueError(
            "Unexpected non-subscriber count: "
            f"{non_subscriber_count}."
        )

    if subscriber_count != EXPECTED_SUBSCRIBERS:
        raise ValueError(
            "Unexpected subscriber count: "
            f"{subscriber_count}."
        )

    if duplicate_id_count != 0:
        raise ValueError(
            "Duplicate primary-key values were detected."
        )

    if null_subscribed_count != 0:
        raise ValueError(
            "Null target values were detected."
        )

    return validation_results


def display_database_preview() -> None:
    """Display a short database preview and target summary."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        preview = pd.read_sql_query(
            f"""
            SELECT *
            FROM {TABLE_NAME}
            ORDER BY campaign_record_id
            LIMIT 5;
            """,
            connection,
        )

        target_summary = pd.read_sql_query(
            f"""
            SELECT
                subscribed,
                COUNT(*) AS client_count,
                ROUND(
                    100.0 * COUNT(*) /
                    SUM(COUNT(*)) OVER (),
                    2
                ) AS percentage
            FROM {TABLE_NAME}
            GROUP BY subscribed
            ORDER BY subscribed;
            """,
            connection,
        )

    print("\nFirst five database records:")
    print(preview.to_string(index=False))

    print("\nDatabase target distribution:")
    print(target_summary.to_string(index=False))


def main() -> None:
    """Create, load, and validate the SQLite database."""
    print("=" * 70)
    print("BANK MARKETING — SQLITE DATABASE SETUP")
    print("=" * 70)

    clean_data = load_clean_data()

    database_data = prepare_database_data(
        clean_data
    )

    print(f"\nCleaned CSV shape: {clean_data.shape}")
    print(f"Database input shape: {database_data.shape}")

    print("\nCreating SQLite database...")
    create_database(database_data)

    validation_results = validate_database()

    print("\nDatabase validation results:")

    for metric, value in validation_results.items():
        print(f"- {metric}: {value}")

    display_database_preview()

    print(f"\nDatabase path:\n{DATABASE_PATH}")

    print(
        "\nSQLite database creation and validation "
        "completed successfully."
    )


if __name__ == "__main__":
    main()