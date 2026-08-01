"""Execute and validate named SQL business-analysis queries."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "bank_marketing.db"
)

SQL_FILE_PATH = (
    PROJECT_ROOT
    / "sql"
    / "business_analysis.sql"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "sql"
)

QUERY_MARKER = "-- name:"


REQUIRED_QUERY_NAMES = {
    "dataset_overview",
    "subscription_by_job",
    "subscription_by_education",
    "subscription_by_contact_type",
    "subscription_by_month",
    "subscription_by_previous_outcome",
    "average_balance_by_subscription",
    "conversion_by_age_group",
    "conversion_by_campaign_contacts",
    "job_performance_ranking",
    "scalable_segment_ranking",
}


def load_named_queries(
    path: Path = SQL_FILE_PATH,
) -> dict[str, str]:
    """Load SQL statements identified by '-- name:' markers."""
    if not path.exists():
        raise FileNotFoundError(
            f"SQL analysis file not found: {path}"
        )

    queries: dict[str, str] = {}

    current_name: str | None = None
    current_lines: list[str] = []

    def store_current_query() -> None:
        """Store the query currently being parsed."""
        if current_name is None:
            return

        query_text = "\n".join(
            current_lines
        ).strip()

        if not query_text:
            raise ValueError(
                f"Query '{current_name}' is empty."
            )

        if current_name in queries:
            raise ValueError(
                f"Duplicate query name: {current_name}"
            )

        queries[current_name] = query_text

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        stripped_line = line.strip()

        if stripped_line.lower().startswith(
            QUERY_MARKER
        ):
            store_current_query()

            current_name = stripped_line[
                len(QUERY_MARKER):
            ].strip()

            if not current_name:
                raise ValueError(
                    "A SQL query marker has no name."
                )

            current_lines = []
            continue

        if current_name is not None:
            current_lines.append(line)

    store_current_query()

    missing_queries = (
        REQUIRED_QUERY_NAMES - set(queries)
    )

    unexpected_queries = (
        set(queries) - REQUIRED_QUERY_NAMES
    )

    if missing_queries:
        raise ValueError(
            "Missing required SQL queries: "
            f"{sorted(missing_queries)}"
        )

    if unexpected_queries:
        raise ValueError(
            "Unexpected SQL queries: "
            f"{sorted(unexpected_queries)}"
        )

    return queries


def execute_queries(
    queries: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Execute every named SQL query."""
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"SQLite database not found: {DATABASE_PATH}\n"
            "Run: python src\\database.py"
        )

    results: dict[str, pd.DataFrame] = {}

    with sqlite3.connect(DATABASE_PATH) as connection:
        for query_name, query_text in queries.items():
            result = pd.read_sql_query(
                query_text,
                connection,
            )

            if result.empty:
                raise ValueError(
                    f"Query '{query_name}' returned no rows."
                )

            results[query_name] = result

    return results


def validate_results(
    results: dict[str, pd.DataFrame],
) -> None:
    """Validate key totals produced by SQL."""
    overview = results["dataset_overview"].iloc[0]

    if int(overview["total_clients"]) != 45_211:
        raise ValueError(
            "Unexpected total client count."
        )

    if int(overview["subscribers"]) != 5_289:
        raise ValueError(
            "Unexpected subscriber count."
        )

    if int(overview["non_subscribers"]) != 39_922:
        raise ValueError(
            "Unexpected non-subscriber count."
        )

    job_total = int(
        results["subscription_by_job"][
            "client_count"
        ].sum()
    )

    month_total = int(
        results["subscription_by_month"][
            "client_count"
        ].sum()
    )

    age_total = int(
        results["conversion_by_age_group"][
            "client_count"
        ].sum()
    )

    campaign_total = int(
        results[
            "conversion_by_campaign_contacts"
        ]["client_count"].sum()
    )

    expected_total = 45_211

    if job_total != expected_total:
        raise ValueError(
            "Job summary does not cover all clients."
        )

    if month_total != expected_total:
        raise ValueError(
            "Month summary does not cover all clients."
        )

    if age_total != expected_total:
        raise ValueError(
            "Age summary does not cover all clients."
        )

    if campaign_total != expected_total:
        raise ValueError(
            "Campaign summary does not cover all clients."
        )

    for query_name, result in results.items():
        if "subscription_rate" in result.columns:
            if not result[
                "subscription_rate"
            ].between(0, 100).all():
                raise ValueError(
                    "Invalid subscription rate in "
                    f"query '{query_name}'."
                )


def save_results(
    results: dict[str, pd.DataFrame],
) -> None:
    """Save each SQL query result as a CSV report."""
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    for query_name, result in results.items():
        output_path = (
            OUTPUT_DIRECTORY
            / f"{query_name}.csv"
        )

        result.to_csv(
            output_path,
            index=False,
        )


def display_results(
    results: dict[str, pd.DataFrame],
) -> None:
    """Print a concise preview of each SQL result."""
    for query_name, result in results.items():
        print("\n" + "=" * 70)
        print(query_name.upper())
        print("=" * 70)

        print(
            result.head(10).to_string(
                index=False
            )
        )

        if len(result) > 10:
            print(
                f"\nShowing 10 of "
                f"{len(result)} rows."
            )


def main() -> None:
    """Run and validate the SQL business analysis."""
    print("=" * 70)
    print("BANK MARKETING — SQL BUSINESS ANALYSIS")
    print("=" * 70)

    queries = load_named_queries()

    print(
        f"\nNamed SQL queries loaded: "
        f"{len(queries)}"
    )

    results = execute_queries(queries)

    validate_results(results)

    save_results(results)

    display_results(results)

    print("\nGenerated SQL reports:")

    for output_path in sorted(
        OUTPUT_DIRECTORY.glob("*.csv")
    ):
        print(f"- {output_path}")

    print(
        "\nAll SQL business-analysis queries "
        "executed and validated successfully."
    )


if __name__ == "__main__":
    main()
