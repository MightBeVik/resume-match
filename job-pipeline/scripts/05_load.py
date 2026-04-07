"""Load processed pipeline outputs into SQLite."""
from __future__ import annotations

import sqlite3

import pandas as pd

from _common import DB_PATH, PROCESSED_DIR, ensure_directories, log


def _read_csv(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing processed input {path}. Run the earlier pipeline steps first.")
    return pd.read_csv(path)


def main() -> None:
    ensure_directories()

    raw_postings = _read_csv("raw_postings.csv")
    noc_structure = _read_csv("noc_structure.csv")
    oasis_competencies = _read_csv("oasis_competencies.csv")
    enriched_jobs = _read_csv("synthesized_jobs.csv")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        connection.executescript(
            """
            DROP TABLE IF EXISTS raw_postings;
            DROP TABLE IF EXISTS noc_structure;
            DROP TABLE IF EXISTS oasis_competencies;
            DROP TABLE IF EXISTS enriched_jobs;

            CREATE TABLE raw_postings (
                id INTEGER PRIMARY KEY,
                source TEXT,
                noc_code TEXT,
                job_title TEXT,
                employer_name TEXT,
                city TEXT,
                province TEXT,
                salary TEXT,
                date_posted TEXT
            );

            CREATE TABLE noc_structure (
                noc_code TEXT PRIMARY KEY,
                noc_title TEXT,
                teer INTEGER,
                broad_category TEXT
            );

            CREATE TABLE oasis_competencies (
                noc_code TEXT PRIMARY KEY,
                top_skills TEXT,
                workplaces TEXT
            );

            CREATE TABLE enriched_jobs (
                id INTEGER PRIMARY KEY,
                source TEXT,
                noc_code TEXT,
                noc_title TEXT,
                teer INTEGER,
                broad_category TEXT,
                job_title TEXT,
                employer_name TEXT,
                city TEXT,
                province TEXT,
                salary TEXT,
                date_posted TEXT,
                lead_statement TEXT,
                main_duties TEXT,
                top_skills TEXT,
                workplaces TEXT,
                jd_text TEXT
            );
            """
        )

        raw_postings.to_sql("raw_postings", connection, if_exists="append", index=False)
        noc_structure.to_sql("noc_structure", connection, if_exists="append", index=False)
        oasis_competencies.to_sql("oasis_competencies", connection, if_exists="append", index=False)
        enriched_jobs.to_sql("enriched_jobs", connection, if_exists="append", index=False)

    log("load", f"SQLite database written to {DB_PATH}")


if __name__ == "__main__":
    main()