"""Tests for the matcher orchestrator."""
from app.nlp.matcher import analyze_match


RESUME = """
John Doe
Software Engineer

Summary
Experienced software engineer with 6 years of frontend development.

Experience
Senior Frontend Developer at Google (2019-2024)
- Led React/TypeScript migration for core product
- Managed team of 4 developers using Agile/Scrum methodology
- Built REST API integrations and microservices

Skills
React, TypeScript, JavaScript, Python, Docker, REST APIs, Git, Agile, Scrum

Education
Bachelor of Science in Computer Science, MIT (2018)
"""

JOB_DESCRIPTION = """
Senior Frontend Engineer at TechCorp

Requirements
- 5+ years of experience in frontend development
- Proficiency in React and TypeScript
- Experience with REST APIs and microservices
- Strong understanding of Git and version control
- Bachelor's degree in Computer Science or related field

Responsibilities
- Lead frontend architecture decisions
- Mentor junior developers
- Build scalable, performant web applications
- Collaborate with backend team on API design

Preferred Qualifications
- Experience with AWS or cloud platforms
- Master's degree
- GraphQL experience
- Experience with CI/CD pipelines
"""


def test_analyze_match_returns_expected_structure():
    result = analyze_match(RESUME, JOB_DESCRIPTION)
    assert "overall_score" in result
    assert "verdict" in result
    assert "summary" in result
    assert "sections" in result
    assert "nlp_details" in result


def test_analyze_match_score_in_range():
    result = analyze_match(RESUME, JOB_DESCRIPTION)
    assert 0 <= result["overall_score"] <= 100


def test_analyze_match_verdict_is_valid():
    result = analyze_match(RESUME, JOB_DESCRIPTION)
    assert result["verdict"] in ["Weak Match", "Moderate Match", "Strong Match"]


def test_analyze_match_sections_have_scores():
    result = analyze_match(RESUME, JOB_DESCRIPTION)
    for key in ["skills", "experience", "education", "preferred"]:
        assert key in result["sections"]
        section = result["sections"][key]
        assert "score" in section
        assert "matched" in section
        assert "partial" in section
        assert "missing" in section
        assert 0 <= section["score"] <= 100


def test_analyze_match_nlp_details_present():
    result = analyze_match(RESUME, JOB_DESCRIPTION)
    details = result["nlp_details"]
    assert "jd_sections_parsed" in details
    assert "resume_sections_parsed" in details
    assert "resume_entities" in details
    assert "tfidf_top_keywords" in details
    assert "similarity_scores" in details


def test_analyze_match_good_resume_scores_well():
    result = analyze_match(RESUME, JOB_DESCRIPTION)
    assert result["overall_score"] >= 35


def test_analyze_match_poor_resume_scores_low():
    unrelated_resume = """
    Jane Smith
    Professional Chef

    Summary
    Award-winning chef with 15 years in fine dining.

    Experience
    Head Chef at Le Cordon Bleu Restaurant (2010-2024)
    - Created seasonal tasting menus
    - Managed kitchen staff of 20

    Skills
    French cuisine, Pastry, Wine pairing, Menu design, Food safety

    Education
    Culinary Arts Diploma, Le Cordon Bleu Paris (2009)
    """
    result = analyze_match(unrelated_resume, JOB_DESCRIPTION)
    assert result["overall_score"] < 40


def test_analyze_match_matches_bachelors_requirement():
    result = analyze_match(RESUME, JOB_DESCRIPTION)
    assert "Bachelor's degree in Computer Science or related field" in result["sections"]["education"]["matched"]
    assert "Bachelor's degree in Computer Science or related field" not in result["sections"]["skills"]["matched"]


def test_analyze_match_does_not_false_positive_preferred_items():
    result = analyze_match(RESUME, JOB_DESCRIPTION)
    preferred = result["sections"]["preferred"]
    assert "Master's degree" in preferred["missing"]
    assert "GraphQL experience" in preferred["missing"]
