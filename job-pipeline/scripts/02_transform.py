"""Normalize raw source files into processed CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import PROCESSED_DIR, RAW_DIR, detect_column, ensure_directories, load_table, log, normalize_columns, normalize_noc_code


def transform_job_bank() -> pd.DataFrame:
    files = sorted((RAW_DIR / "job_bank").glob("*.csv"))
    if not files:
        raise FileNotFoundError("No Job Bank CSV files found under data/raw/job_bank. Run 01_ingest.py first.")

    frames: list[pd.DataFrame] = []
    for path in files:
        log("transform", f"Reading {path.name}")
        frame = normalize_columns(load_table(path))

        noc_column = detect_column(frame, ["noc", "noc_code", "national_occupational_classification", "noc_2021", "noc21_code", "noc_2016_code", "noc2016_code"])
        title_column = detect_column(frame, ["job_title", "title", "jobtitle", "title_en"])
        employer_column = detect_column(frame, ["employer_name", "company_name", "employer", "organization_name"])
        city_column = detect_column(frame, ["city", "municipality", "location_city"])
        province_column = detect_column(frame, ["province", "province_territory", "region"])
        salary_column = detect_column(frame, ["salary", "wage", "salary_range", "hourly_wage", "salary_condition_detail"])
        date_column = detect_column(frame, ["date_posted", "posted_date", "posting_date", "date", "first_posting_date"])

        postings = pd.DataFrame(
            {
                "source": "job_bank",
                "noc_code": frame[noc_column].map(normalize_noc_code) if noc_column else None,
                "job_title": frame[title_column] if title_column else None,
                "employer_name": frame[employer_column] if employer_column else None,
                "city": frame[city_column] if city_column else None,
                "province": frame[province_column] if province_column else None,
                "salary": frame[salary_column] if salary_column else None,
                "date_posted": frame[date_column] if date_column else None,
            }
        )
        postings = postings.dropna(subset=["noc_code"]).copy()
        postings["job_title"] = postings["job_title"].fillna("Unknown Title")
        frames.append(postings)

    result = pd.concat(frames, ignore_index=True).drop_duplicates()
    log("transform", f"Normalized {len(result)} Job Bank postings")
    return result


def transform_alberta() -> pd.DataFrame:
    files = sorted((RAW_DIR / "alberta").glob("*"))
    frames: list[pd.DataFrame] = []
    for path in files:
        if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            continue
        log("transform", f"Reading Alberta source {path.name}")
        frame = normalize_columns(load_table(path))
        frames.append(frame)
    if not frames:
        log("transform", "No Alberta tabular resources were available; skipping Alberta market transform")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def transform_noc_structure() -> pd.DataFrame:
    path = RAW_DIR / "reference" / "noc_structure.csv"
    if not path.exists():
        log("transform", "NOC structure file is unavailable; continuing with empty NOC reference")
        return pd.DataFrame(columns=["noc_code", "noc_title", "teer", "broad_category"])
    frame = normalize_columns(load_table(path))
    code_column = detect_column(frame, ["noc_code", "code", "unit_group"])
    title_column = detect_column(frame, ["title", "unit_group_title", "occupation_title"])
    teer_column = detect_column(frame, ["teer", "teer_category"])
    broad_column = detect_column(frame, ["broad_category", "broad_occupational_category", "broad_category_title"])

    result = pd.DataFrame(
        {
            "noc_code": frame[code_column].map(normalize_noc_code) if code_column else None,
            "noc_title": frame[title_column] if title_column else None,
            "teer": frame[teer_column] if teer_column else None,
            "broad_category": frame[broad_column] if broad_column else None,
        }
    ).dropna(subset=["noc_code"])
    return result.drop_duplicates(subset=["noc_code"]).reset_index(drop=True)


def transform_reference(filename: str) -> pd.DataFrame:
    path = RAW_DIR / "reference" / filename
    if filename == "oasis_skills.csv":
        return normalize_columns(pd.read_csv(path, sep=";"))
    return normalize_columns(load_table(path))


def main() -> None:
    ensure_directories()

    job_postings = transform_job_bank()
    job_postings.to_csv(PROCESSED_DIR / "raw_postings.csv", index=False)

    alberta_market = transform_alberta()
    if not alberta_market.empty:
        alberta_market.to_csv(PROCESSED_DIR / "alberta_market.csv", index=False)

    noc_structure = transform_noc_structure()
    noc_structure.to_csv(PROCESSED_DIR / "noc_structure.csv", index=False)

    transform_reference("oasis_lead_statements.csv").to_csv(PROCESSED_DIR / "oasis_lead_statements.csv", index=False)
    transform_reference("oasis_main_duties.csv").to_csv(PROCESSED_DIR / "oasis_main_duties.csv", index=False)
    transform_reference("oasis_workplaces.csv").to_csv(PROCESSED_DIR / "oasis_workplaces.csv", index=False)
    transform_reference("oasis_skills.csv").to_csv(PROCESSED_DIR / "oasis_skills.csv", index=False)

    log("transform", "Wrote processed CSV files to data/processed")


if __name__ == "__main__":
    main()