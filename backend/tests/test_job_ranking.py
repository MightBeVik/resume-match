"""Tests for SQLite-backed batch job ranking."""
from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


VALID_RESUME = """
Jane Candidate

Summary
Frontend engineer with 6 years of experience building React and TypeScript applications for SaaS products.

Experience
Senior Frontend Developer at ExampleCorp
- Built React and TypeScript interfaces for analytics workflows
- Integrated REST APIs with backend services
- Worked closely with product and platform teams

Skills
React, TypeScript, JavaScript, REST APIs, Git, Agile, Docker

Education
Bachelor of Science in Computer Science
"""


def _create_jobs_db(path: str) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
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
        connection.executemany(
            """
            INSERT INTO enriched_jobs (
                id, source, noc_code, noc_title, teer, broad_category, job_title,
                employer_name, city, province, salary, date_posted,
                lead_statement, main_duties, top_skills, workplaces, jd_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    1,
                    "job_bank",
                    "21234",
                    "Web developers and programmers",
                    1,
                    "Natural and applied sciences",
                    "Frontend Engineer",
                    "Toronto Apps",
                    "Toronto",
                    "ON",
                    "$110000",
                    "2026-04-01",
                    "Build frontend software products.",
                    "Build React and TypeScript interfaces\nCollaborate on REST API design",
                    "React|TypeScript|REST APIs|Git",
                    "Software companies",
                    "Build frontend software products.\n\nBuild React and TypeScript interfaces.\nCollaborate on REST API design.\n\nTop skills: React, TypeScript, REST APIs, Git",
                ),
                (
                    2,
                    "job_bank",
                    "63200",
                    "Cooks",
                    3,
                    "Sales and service",
                    "Line Cook",
                    "Downtown Bistro",
                    "Toronto",
                    "ON",
                    "$48000",
                    "2026-04-02",
                    "Prepare meals in a fast-paced kitchen.",
                    "Cook menu items\nMaintain kitchen sanitation",
                    "Food safety|Knife skills|Menu prep",
                    "Restaurants",
                    "Prepare meals in a fast-paced kitchen.\n\nCook menu items and maintain kitchen sanitation.\n\nTop skills: Food safety, Knife skills, Menu prep",
                ),
                (
                    3,
                    "job_bank",
                    "21234",
                    "Web developers and programmers",
                    1,
                    "Natural and applied sciences",
                    "Backend Engineer",
                    "Montreal Systems",
                    "Montreal",
                    "QC",
                    "$115000",
                    "2026-03-30",
                    "Build backend systems.",
                    "Design APIs\nMaintain distributed systems",
                    "Python|APIs|Databases",
                    "Technology companies",
                    "Build backend systems.\n\nDesign APIs and maintain distributed systems.\n\nTop skills: Python, APIs, Databases",
                ),
            ],
        )


def test_rank_jobs_returns_sorted_results(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs.db"
    _create_jobs_db(str(db_path))
    monkeypatch.setenv("RESUME_MATCH_JOBS_DB_PATH", str(db_path))

    response = client.post(
        "/api/rank-jobs",
        json={
            "resume_text": VALID_RESUME,
            "province": "ON",
            "limit": 2,
            "candidate_pool": 10,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_jobs_considered"] == 2
    assert data["returned"] == 2
    assert data["results"][0]["job_title"] == "Frontend Engineer"
    assert data["results"][0]["overall_score"] >= data["results"][1]["overall_score"]
    assert "React" in " ".join(data["results"][0]["top_matches"])


def test_rank_jobs_applies_city_filter(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs.db"
    _create_jobs_db(str(db_path))
    monkeypatch.setenv("RESUME_MATCH_JOBS_DB_PATH", str(db_path))

    response = client.post(
        "/api/rank-jobs",
        json={
            "resume_text": VALID_RESUME,
            "city": "Montreal",
            "limit": 5,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_jobs_considered"] == 1
    assert data["results"][0]["city"] == "Montreal"


def test_rank_jobs_missing_database_returns_503(monkeypatch, tmp_path):
    missing_path = tmp_path / "missing.db"
    monkeypatch.setenv("RESUME_MATCH_JOBS_DB_PATH", str(missing_path))

    response = client.post(
        "/api/rank-jobs",
        json={
            "resume_text": VALID_RESUME,
        },
    )

    assert response.status_code == 503
    assert "Jobs database not found" in response.json()["detail"]