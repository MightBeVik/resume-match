"""Query enriched jobs from the pipeline SQLite database."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "job-pipeline" / "data" / "jobs.db"

PROVINCE_FILTERS = {
    "ab": ["AB", "Alberta"],
    "alberta": ["AB", "Alberta"],
    "bc": ["BC", "British Columbia"],
    "british columbia": ["BC", "British Columbia"],
    "mb": ["MB", "Manitoba"],
    "manitoba": ["MB", "Manitoba"],
    "nb": ["NB", "New Brunswick"],
    "new brunswick": ["NB", "New Brunswick"],
    "nl": ["NL", "Newfoundland and Labrador"],
    "newfoundland and labrador": ["NL", "Newfoundland and Labrador"],
    "ns": ["NS", "Nova Scotia"],
    "nova scotia": ["NS", "Nova Scotia"],
    "nt": ["NT", "Northwest Territories"],
    "northwest territories": ["NT", "Northwest Territories"],
    "nu": ["NU", "Nunavut"],
    "nunavut": ["NU", "Nunavut"],
    "on": ["ON", "Ontario"],
    "ontario": ["ON", "Ontario"],
    "pe": ["PE", "Prince Edward Island"],
    "pei": ["PE", "Prince Edward Island"],
    "prince edward island": ["PE", "Prince Edward Island"],
    "qc": ["QC", "Quebec", "Québec", "QuÃ©bec", "Qu�bec"],
    "quebec": ["QC", "Quebec", "Québec", "QuÃ©bec", "Qu�bec"],
    "québec": ["QC", "Quebec", "Québec", "QuÃ©bec", "Qu�bec"],
    "sk": ["SK", "Saskatchewan"],
    "saskatchewan": ["SK", "Saskatchewan"],
    "yt": ["YT", "Yukon"],
    "yukon": ["YT", "Yukon"],
}


def resolve_jobs_db_path() -> Path:
    configured = os.getenv("RESUME_MATCH_JOBS_DB_PATH")
    if configured:
        return Path(configured)
    return DEFAULT_DB_PATH


def _expand_province_filter(value: str) -> list[str]:
    cleaned = value.strip().lower()
    if not cleaned:
        return []
    return PROVINCE_FILTERS.get(cleaned, [value.strip()])


def fetch_rankable_jobs(filters: dict[str, str | int | None]) -> list[sqlite3.Row]:
    db_path = resolve_jobs_db_path()
    if not db_path.exists():
        raise FileNotFoundError(
            f"Jobs database not found at {db_path}. Build it with job-pipeline/scripts/run_all.py first."
        )

    clauses = ["jd_text IS NOT NULL", "TRIM(jd_text) <> ''"]
    parameters: list[object] = []

    if filters.get("province"):
        province_values = _expand_province_filter(str(filters["province"]))
        if province_values:
            placeholders = ", ".join("LOWER(?)" for _ in province_values)
            clauses.append(f"LOWER(province) IN ({placeholders})")
            parameters.extend(province_values)
    if filters.get("city"):
        clauses.append("LOWER(city) = LOWER(?)")
        parameters.append(filters["city"])
    if filters.get("source"):
        clauses.append("LOWER(source) = LOWER(?)")
        parameters.append(filters["source"])
    if filters.get("broad_category"):
        clauses.append("LOWER(broad_category) LIKE LOWER(?)")
        parameters.append(f"%{filters['broad_category']}%")
    if filters.get("job_title_query"):
        clauses.append("LOWER(job_title) LIKE LOWER(?)")
        parameters.append(f"%{filters['job_title_query']}%")
    if filters.get("employer_query"):
        clauses.append("LOWER(employer_name) LIKE LOWER(?)")
        parameters.append(f"%{filters['employer_query']}%")

    candidate_pool = int(filters.get("candidate_pool") or 100)
    candidate_pool = max(1, min(candidate_pool, 500))

    query = f"""
        SELECT
            id,
            source,
            noc_code,
            noc_title,
            teer,
            broad_category,
            job_title,
            employer_name,
            city,
            province,
            salary,
            date_posted,
            jd_text
        FROM enriched_jobs
        WHERE {' AND '.join(clauses)}
        ORDER BY COALESCE(date_posted, '') DESC, id DESC
        LIMIT ?
    """
    parameters.append(candidate_pool)

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(query, parameters).fetchall()
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                f"Could not query enriched_jobs from {db_path}. Ensure 05_load.py completed successfully."
            ) from exc
    return rows