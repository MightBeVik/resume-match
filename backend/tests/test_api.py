"""Integration tests for the /api/analyze endpoint."""
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


VALID_RESUME = """
John Doe - Software Engineer

Summary
Experienced software engineer with 6 years of frontend development building scalable web applications.

Experience
Senior Frontend Developer at Google (2019-2024)
- Led React and TypeScript migration for the core product serving 10M users
- Managed a team of 4 developers using Agile and Scrum methodologies
- Built REST API integrations and designed microservice architecture

Skills
React, TypeScript, JavaScript, Python, Docker, REST APIs, Git, Agile, Scrum, Node.js

Education
Bachelor of Science in Computer Science, Massachusetts Institute of Technology (2018)
"""

VALID_JD = """
Senior Frontend Engineer at TechCorp

Requirements
- 5+ years of experience in frontend development
- Proficiency in React and TypeScript
- Experience with REST APIs and microservices architecture
- Strong understanding of Git and version control
- Bachelor's degree in Computer Science or related field

Responsibilities
- Lead frontend architecture decisions for the platform
- Mentor junior developers and conduct code reviews
- Build scalable, performant web applications
- Collaborate with backend engineering team on API design

Preferred Qualifications
- Experience with AWS or cloud platforms
- Master's degree in Computer Science
- GraphQL experience
- Experience with CI/CD pipelines
"""


def test_analyze_success():
    response = client.post("/api/analyze", json={
        "resume_text": VALID_RESUME,
        "job_description": VALID_JD,
    })
    assert response.status_code == 200
    data = response.json()
    assert "overall_score" in data
    assert 0 <= data["overall_score"] <= 100
    assert data["verdict"] in ["Weak Match", "Moderate Match", "Strong Match"]
    assert "sections" in data
    assert "nlp_details" in data


def test_analyze_empty_resume_returns_422():
    response = client.post("/api/analyze", json={
        "resume_text": "too short",
        "job_description": VALID_JD,
    })
    assert response.status_code == 422


def test_analyze_empty_jd_returns_422():
    response = client.post("/api/analyze", json={
        "resume_text": VALID_RESUME,
        "job_description": "too short",
    })
    assert response.status_code == 422


def test_analyze_response_has_all_sections():
    response = client.post("/api/analyze", json={
        "resume_text": VALID_RESUME,
        "job_description": VALID_JD,
    })
    data = response.json()
    for key in ["skills", "experience", "education", "preferred"]:
        assert key in data["sections"]
        section = data["sections"][key]
        assert "score" in section
        assert "matched" in section
        assert "partial" in section
        assert "missing" in section


def test_analyze_nlp_details_structure():
    response = client.post("/api/analyze", json={
        "resume_text": VALID_RESUME,
        "job_description": VALID_JD,
    })
    details = response.json()["nlp_details"]
    assert "jd_sections_parsed" in details
    assert "resume_sections_parsed" in details
    assert "resume_entities" in details
    assert "tfidf_top_keywords" in details
    assert "similarity_scores" in details
    assert "tfidf_cosine" in details["similarity_scores"]
    assert "semantic" in details["similarity_scores"]


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
