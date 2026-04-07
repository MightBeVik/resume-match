"""Build synthetic job-description text for each enriched posting."""
from __future__ import annotations

import pandas as pd

from _common import PROCESSED_DIR, ensure_directories, log


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _format_skills(value: object) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    items = [item.strip() for item in text.split("|") if item.strip()]
    if not items:
        return ""
    return "Top skills: " + ", ".join(items)


def _format_context(row: pd.Series) -> str:
    parts = []
    employer = _clean_text(row.get("employer_name"))
    city = _clean_text(row.get("city"))
    province = _clean_text(row.get("province"))
    salary = _clean_text(row.get("salary"))
    if employer:
        parts.append(f"Employer: {employer}")
    location = ", ".join([part for part in [city, province] if part])
    if location:
        parts.append(f"Location: {location}")
    if salary:
        parts.append(f"Salary: {salary}")
    return "\n".join(parts)


def assemble_jd_text(row: pd.Series) -> str:
    sections = [
        _clean_text(row.get("lead_statement")),
        _clean_text(row.get("main_duties")),
        _format_skills(row.get("top_skills")),
        _clean_text(row.get("workplaces")),
        _format_context(row),
    ]
    return "\n\n".join(section for section in sections if section)


def main() -> None:
    ensure_directories()
    source_path = PROCESSED_DIR / "enriched_jobs.csv"
    if not source_path.exists():
        raise FileNotFoundError("Missing data/processed/enriched_jobs.csv. Run 03_enrich.py first.")

    frame = pd.read_csv(source_path)
    frame["jd_text"] = frame.apply(assemble_jd_text, axis=1)
    frame = frame[frame["jd_text"].str.strip().ne("")].copy()

    destination = PROCESSED_DIR / "synthesized_jobs.csv"
    frame.to_csv(destination, index=False)
    log("synthesize", f"Wrote {len(frame)} synthetic JD rows to {destination.name}")


if __name__ == "__main__":
    main()