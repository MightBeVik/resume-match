"""Tests for JD and resume section parsing."""
from app.nlp.section_parser import parse_job_description, parse_resume


STRUCTURED_JD = """
About the Company
TechCorp is a leading software company.

Requirements
- 5+ years of experience in frontend development
- Proficiency in React and TypeScript
- Experience with REST APIs
- Bachelor's degree in Computer Science

Responsibilities
- Lead frontend architecture decisions
- Mentor junior developers
- Build scalable web applications

Preferred Qualifications
- Experience with AWS or cloud platforms
- Master's degree preferred
- GraphQL experience
"""

UNSTRUCTURED_JD = """
We are looking for a Senior Frontend Engineer to join our team. You should have
5+ years of experience with React and TypeScript. You'll be leading frontend
architecture and mentoring junior developers. AWS experience is a plus.
Bachelor's degree required, Master's preferred.
"""

SAMPLE_RESUME = """
John Doe
Software Engineer

Summary
Experienced software engineer with 6 years of frontend development expertise.

Experience
Senior Frontend Developer at Google (2019-2024)
- Led React/TypeScript migration for core product
- Managed team of 4 developers
- Built REST API integrations

Skills
React, TypeScript, JavaScript, Python, Docker, REST APIs, Git, Agile

Education
Bachelor of Science in Computer Science, MIT (2018)
"""


def test_parse_structured_jd_finds_requirements():
    result = parse_job_description(STRUCTURED_JD)
    assert len(result["requirements"]) > 0
    assert any("react" in r.lower() or "typescript" in r.lower() for r in result["requirements"])


def test_parse_structured_jd_preserves_numeric_prefixes():
    result = parse_job_description(STRUCTURED_JD)
    assert any("5+ years" in requirement for requirement in result["requirements"])


def test_parse_structured_jd_finds_responsibilities():
    result = parse_job_description(STRUCTURED_JD)
    assert len(result["responsibilities"]) > 0


def test_parse_structured_jd_finds_preferred():
    result = parse_job_description(STRUCTURED_JD)
    assert len(result["preferred"]) > 0
    assert any("aws" in p.lower() or "cloud" in p.lower() for p in result["preferred"])


def test_parse_unstructured_jd_still_returns_all_keys():
    result = parse_job_description(UNSTRUCTURED_JD)
    assert "requirements" in result
    assert "responsibilities" in result
    assert "preferred" in result
    assert "other" in result
    total_items = sum(len(v) for v in result.values())
    assert total_items > 0


def test_parse_resume_extracts_sections():
    result = parse_resume(SAMPLE_RESUME)
    assert "skills" in result
    assert "experience" in result
    assert "education" in result
    assert "summary" in result


def test_parse_resume_skills_content():
    result = parse_resume(SAMPLE_RESUME)
    assert "react" in result["skills"].lower()


def test_parse_resume_education_content():
    result = parse_resume(SAMPLE_RESUME)
    assert "computer science" in result["education"].lower()
