"""Download and extract the official UCI Bank Marketing dataset."""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/222/"
    "bank%2Bmarketing.zip"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

ARCHIVE_PATH = RAW_DATA_DIR / "bank_marketing_uci.zip"
CSV_PATH = RAW_DATA_DIR / "bank-full.csv"


def find_archive_member(
    member_names: list[str],
    expected_filename: str,
) -> str | None:
    """Find a file inside an archive regardless of its parent directory."""
    expected_filename = expected_filename.lower()

    for member_name in member_names:
        if Path(member_name).name.lower() == expected_filename:
            return member_name

    return None


def download_archive(force: bool = False) -> None:
    """Download the official UCI archive into the raw-data directory."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if ARCHIVE_PATH.exists() and not force:
        print(f"Archive already exists: {ARCHIVE_PATH}")
        return

    temporary_path = ARCHIVE_PATH.with_suffix(".zip.part")

    if temporary_path.exists():
        temporary_path.unlink()

    request = urllib.request.Request(
        DATASET_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 Bank-Marketing-Analytics-ML/1.0"
            )
        },
    )

    print("Downloading the official UCI Bank Marketing archive...")

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary_path.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)

        temporary_path.replace(ARCHIVE_PATH)

    except (urllib.error.URLError, TimeoutError, OSError) as error:
        if temporary_path.exists():
            temporary_path.unlink()

        raise RuntimeError(
            f"Dataset download failed: {error}"
        ) from error

    print(f"Archive saved to: {ARCHIVE_PATH}")


def copy_member_to_file(
    archive: zipfile.ZipFile,
    member_name: str,
    destination: Path,
) -> None:
    """Copy one archive member to a local file."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    with archive.open(member_name) as source_file:
        with destination.open("wb") as destination_file:
            shutil.copyfileobj(source_file, destination_file)


def extract_bank_full_csv() -> None:
    """Extract bank-full.csv from the downloaded UCI archive."""
    if CSV_PATH.exists():
        print(f"Dataset already exists: {CSV_PATH}")
        return

    print("Extracting bank-full.csv...")

    try:
        with zipfile.ZipFile(ARCHIVE_PATH) as outer_archive:
            outer_members = outer_archive.namelist()

            direct_csv_member = find_archive_member(
                outer_members,
                "bank-full.csv",
            )

            if direct_csv_member is not None:
                copy_member_to_file(
                    outer_archive,
                    direct_csv_member,
                    CSV_PATH,
                )
                print(f"Dataset extracted to: {CSV_PATH}")
                return

            nested_bank_zip = find_archive_member(
                outer_members,
                "bank.zip",
            )

            if nested_bank_zip is None:
                raise FileNotFoundError(
                    "Neither bank-full.csv nor bank.zip was found "
                    "inside the downloaded UCI archive."
                )

            nested_archive_bytes = outer_archive.read(nested_bank_zip)

        with zipfile.ZipFile(
            io.BytesIO(nested_archive_bytes)
        ) as nested_archive:
            nested_members = nested_archive.namelist()

            csv_member = find_archive_member(
                nested_members,
                "bank-full.csv",
            )

            if csv_member is None:
                raise FileNotFoundError(
                    "bank-full.csv was not found inside bank.zip."
                )

            copy_member_to_file(
                nested_archive,
                csv_member,
                CSV_PATH,
            )

    except zipfile.BadZipFile as error:
        raise RuntimeError(
            "The downloaded file is not a valid ZIP archive. "
            "Run the script again with --force."
        ) from error

    print(f"Dataset extracted to: {CSV_PATH}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download the official UCI Bank Marketing dataset."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Download the archive again even if it already exists.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the complete download and extraction process."""
    arguments = parse_arguments()

    try:
        download_archive(force=arguments.force)
        extract_bank_full_csv()

    except (RuntimeError, FileNotFoundError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    file_size_kb = CSV_PATH.stat().st_size / 1024

    print(f"CSV file size: {file_size_kb:,.1f} KB")
    print("Official dataset download completed successfully.")


if __name__ == "__main__":
    main()

