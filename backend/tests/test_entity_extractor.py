"""Tests for entity extraction using spaCy NER and custom patterns."""
from app.nlp.entity_extractor import extract_entities, extract_skills, extract_education


SAMPLE_TEXT = """
Senior Software Engineer at Google with expertise in React, TypeScript,
Python, and AWS. Built microservices using Docker and Kubernetes.
Led a team of 5 engineers. Bachelor of Science in Computer Science from MIT.
"""


def test_extract_entities_finds_organizations():
    result = extract_entities(SAMPLE_TEXT)
    orgs = [e.lower() for e in result["organizations"]]
    assert "google" in orgs or "mit" in orgs


def test_extract_skills_finds_technical_skills():
    skills = extract_skills(SAMPLE_TEXT)
    skill_lower = [s.lower() for s in skills]
    assert "react" in skill_lower
    assert "python" in skill_lower


def test_extract_skills_from_comma_list():
    text = "Skills: React, TypeScript, Python, Docker, AWS, Kubernetes"
    skills = extract_skills(text)
    assert len(skills) >= 4


def test_extract_education_finds_degrees():
    edu = extract_education(SAMPLE_TEXT)
    edu_lower = [e.lower() for e in edu]
    assert any("bachelor" in e or "computer science" in e for e in edu_lower)


def test_extract_education_finds_multiple_degrees():
    text = "B.S. in Computer Science from MIT. M.S. in Data Science from Stanford. PhD in AI."
    edu = extract_education(text)
    assert len(edu) >= 2


def test_extract_entities_returns_all_keys():
    result = extract_entities(SAMPLE_TEXT)
    assert "skills" in result
    assert "organizations" in result
    assert "education" in result


def test_extract_skills_avoids_substring_false_positives():
    text = "Worked at Google building JavaScript apps for modern web teams."
    skills = [skill.lower() for skill in extract_skills(text)]
    assert "go" not in skills
    assert "java" not in skills
    assert "javascript" in skills


def test_extract_entities_filters_skill_noise_from_organizations():
    result = extract_entities("Skills: React, TypeScript, Git, Docker")
    orgs = [org.lower() for org in result["organizations"]]
    assert "typescript" not in orgs
    assert "git" not in orgs
