"""Batch ranking against the enriched jobs database."""
from __future__ import annotations

from app.jobs.repository import fetch_rankable_jobs, fetch_jobs_search, fetch_jobs_by_ids
from app.nlp.matcher import analyze_match


def _collect_items(sections: dict[str, dict], key: str, limit: int = 5) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for section in sections.values():
        for item in section.get(key, []):
            if item not in seen:
                seen.add(item)
                ordered.append(item)
            if len(ordered) >= limit:
                return ordered
    return ordered


def rank_jobs_for_resume(request: dict) -> dict:
    rows = fetch_rankable_jobs(request)
    results = []
    for row in rows:
        match = analyze_match(request["resume_text"], row["jd_text"])
        results.append(
            {
                "id": row["id"],
                "source": row["source"],
                "noc_code": row["noc_code"],
                "noc_title": row["noc_title"],
                "teer": row["teer"],
                "broad_category": row["broad_category"],
                "job_title": row["job_title"],
                "employer_name": row["employer_name"],
                "city": row["city"],
                "province": row["province"],
                "salary": row["salary"],
                "date_posted": row["date_posted"],
                "overall_score": match["overall_score"],
                "verdict": match["verdict"],
                "summary": match["summary"],
                "top_matches": _collect_items(match["sections"], "matched"),
                "top_gaps": _collect_items(match["sections"], "missing"),
            }
        )

    results.sort(key=lambda item: (item["overall_score"], item.get("date_posted") or ""), reverse=True)

    limit = int(request.get("limit") or 20)
    limit = max(1, min(limit, 100))
    ranked = results[:limit]

    return {
        "total_jobs_considered": len(rows),
        "returned": len(ranked),
        "results": ranked,
    }


def bulk_match_resumes(request: dict) -> dict:
    """Match multiple resumes against a filtered job set.

    Returns a leaderboard matrix: rows=resumes, columns=jobs, cells=scores.
    """
    resumes: list[dict] = request["resumes"]
    job_ids: list[int] | None = request.get("job_ids")
    max_jobs = int(request.get("max_jobs") or 50)
    max_jobs = max(1, min(max_jobs, 200))

    # Fetch job rows
    if job_ids:
        job_rows = fetch_jobs_by_ids(job_ids)
    else:
        _, search_rows = fetch_jobs_search(
            keyword=request.get("keyword"),
            province=request.get("province"),
            city=request.get("city"),
            broad_category=request.get("broad_category"),
            page=1,
            page_size=max_jobs,
        )
        # search_rows only have jd_preview — re-fetch full rows for NLP access to jd_text
        ids_from_search = [r["id"] for r in search_rows]
        job_rows = fetch_jobs_by_ids(ids_from_search) if ids_from_search else []

    if not job_rows:
        return {"resumes": [], "jobs": [], "rows": []}

    # Build job cards list
    jobs_out = [
        {
            "id": row["id"],
            "job_title": row["job_title"],
            "employer_name": row["employer_name"],
            "city": row["city"],
            "province": row["province"],
            "salary": row["salary"],
            "broad_category": row["broad_category"],
            "date_posted": row["date_posted"],
            "jd_preview": (row["jd_text"] or "")[:200],
        }
        for row in job_rows
    ]

    # Run NLP for every resume × job combination
    rows_out = []
    for resume_entry in resumes:
        resume_text = resume_entry["resume_text"]
        resume_name = resume_entry["name"]
        cells: dict[int, dict] = {}
        best_score = -1
        best_job_title = None
        best_job_id = None

        for row in job_rows:
            jd_text = row["jd_text"] or ""
            if not jd_text.strip():
                continue
            match = analyze_match(resume_text, jd_text)
            score = match["overall_score"]
            cells[row["id"]] = {"score": score, "verdict": match["verdict"]}
            if score > best_score:
                best_score = score
                best_job_title = row["job_title"]
                best_job_id = row["id"]

        rows_out.append({
            "resume_name": resume_name,
            "best_score": best_score if best_score >= 0 else 0,
            "best_job_title": best_job_title,
            "best_job_id": best_job_id,
            "cells": cells,
        })

    # Sort rows by best score descending
    rows_out.sort(key=lambda r: r["best_score"], reverse=True)

    return {
        "resumes": [r["resume_name"] for r in rows_out],
        "jobs": jobs_out,
        "rows": rows_out,
    }