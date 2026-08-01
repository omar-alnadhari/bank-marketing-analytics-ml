"""Tests for SQLite database preparation and validation."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from src import database


def build_clean_database_source() -> pd.DataFrame:
    """Create a small cleaned dataset matching the project schema."""
    return pd.DataFrame(
        {
            "age": [33, 48],
            "job": ["technician", "management"],
            "marital": ["single", "married"],
            "education": ["secondary", "tertiary"],
            "default": ["no", "no"],
            "balance": [1500, -20],
            "housing": ["yes", "no"],
            "loan": ["no", "no"],
            "contact": ["cellular", "unknown"],
            "day": [5, 12],
            "month": ["may", "jun"],
            "duration": [120, 65],
            "campaign": [1, 2],
            "pdays": [-1, 10],
            "previous": [0, 2],
            "poutcome": ["unknown", "success"],
            "subscribed": [0, 1],
        }
    )


def test_prepare_database_data_renames_columns_to_sql_contract() -> None:
    source_data = build_clean_database_source()

    database_data = database.prepare_database_data(source_data)

    assert database_data.columns.tolist() == database.EXPECTED_DATABASE_COLUMNS
    assert "default_status" in database_data.columns
    assert "call_duration_seconds" in database_data.columns
    assert "default" not in database_data.columns
    assert "duration" not in database_data.columns


def test_prepare_database_data_rejects_unexpected_schema() -> None:
    source_data = build_clean_database_source().drop(columns=["job"])

    with pytest.raises(ValueError, match="expected SQL schema"):
        database.prepare_database_data(source_data)


def test_read_schema_sql_defines_expected_table() -> None:
    schema_sql = database.read_schema_sql()

    assert f"CREATE TABLE {database.TABLE_NAME}" in schema_sql
    assert "campaign_record_id INTEGER PRIMARY KEY AUTOINCREMENT" in schema_sql


def test_create_database_loads_records_into_temporary_sqlite_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    database_path = tmp_path / "bank_marketing_test.db"
    database_data = database.prepare_database_data(build_clean_database_source())

    monkeypatch.setattr(database, "DATABASE_PATH", database_path)

    database.create_database(database_data)

    with sqlite3.connect(database_path) as connection:
        table_columns = database.get_table_columns(connection)
        row_count = connection.execute(
            f"SELECT COUNT(*) FROM {database.TABLE_NAME};"
        ).fetchone()[0]
        id_values = connection.execute(
            f"SELECT campaign_record_id FROM {database.TABLE_NAME} ORDER BY campaign_record_id;"
        ).fetchall()

    assert table_columns == [
        "campaign_record_id",
        *database.EXPECTED_DATABASE_COLUMNS,
    ]
    assert row_count == len(database_data)
    assert id_values == [(1,), (2,)]


def test_existing_generated_database_matches_validation_contract() -> None:
    if not database.DATABASE_PATH.exists():
        pytest.skip("Generated SQLite database is not present.")

    validation_results = database.validate_database()

    assert validation_results == {
        "integrity_check": "ok",
        "row_count": database.EXPECTED_ROW_COUNT,
        "non_subscribers": database.EXPECTED_NON_SUBSCRIBERS,
        "subscribers": database.EXPECTED_SUBSCRIBERS,
        "duplicate_primary_keys": 0,
        "null_target_values": 0,
    }
