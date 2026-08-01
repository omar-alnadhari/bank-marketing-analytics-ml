"""Shared pytest configuration for the Bank Marketing project."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

for import_path in [PROJECT_ROOT, SRC_DIRECTORY]:
    import_path_text = str(import_path)

    if import_path_text not in sys.path:
        sys.path.insert(0, import_path_text)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the repository root used by the tests."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def modeling_report_directory(project_root: Path) -> Path:
    """Return the directory containing modeling reports."""
    return project_root / "reports" / "modeling"


@pytest.fixture(scope="session")
def statistics_report_directory(project_root: Path) -> Path:
    """Return the directory containing statistical reports."""
    return project_root / "reports" / "statistics"
