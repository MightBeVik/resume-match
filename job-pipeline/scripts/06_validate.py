"""Validate pipeline outputs and print row counts for demo/debugging."""
from __future__ import annotations

import sqlite3

import pandas as pd

from _common import DB_PATH, PROCESSED_DIR, RAW_DIR, log


def _require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required pipeline output is missing: {path}")


def _count_rows_csv(path):
    return len(pd.read_csv(path))


def main() -> None:
    raw_files = list(RAW_DIR.rglob("*.csv")) + list(RAW_DIR.rglob("*.xlsx")) + list(RAW_DIR.rglob("*.xls"))
    if not raw_files:
        raise FileNotFoundError("No raw source files were found. Run 01_ingest.py first.")
    log("validate", f"Raw files present: {len(raw_files)}")

    processed_files = {
        "raw_postings": PROCESSED_DIR / "raw_postings.csv",
        "noc_structure": PROCESSED_DIR / "noc_structure.csv",
        "oasis_competencies": PROCESSED_DIR / "oasis_competencies.csv",
        "enriched_jobs": PROCESSED_DIR / "enriched_jobs.csv",
        "synthesized_jobs": PROCESSED_DIR / "synthesized_jobs.csv",
    }
    for name, path in processed_files.items():
        _require_file(path)
        log("validate", f"{name}: {_count_rows_csv(path)} rows")

    _require_file(DB_PATH)
    with sqlite3.connect(DB_PATH) as connection:
        tables = ["raw_postings", "noc_structure", "oasis_competencies", "enriched_jobs"]
        for table in tables:
            count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            log("validate", f"{table} table rows: {count}")
        sample = connection.execute(
            "SELECT job_title, employer_name, city, province FROM enriched_jobs ORDER BY id LIMIT 3"
        ).fetchall()
        if not sample:
            raise RuntimeError("enriched_jobs is empty after load.")
        for index, row in enumerate(sample, start=1):
            log("validate", f"sample {index}: {row[0]} | {row[1]} | {row[2]}, {row[3]}")

    log("validate", "Pipeline validation succeeded")


if __name__ == "__main__":
    main()