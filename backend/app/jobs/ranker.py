"""Batch ranking against the enriched jobs database."""
from __future__ import annotations

from app.jobs.repository import fetch_rankable_jobs
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