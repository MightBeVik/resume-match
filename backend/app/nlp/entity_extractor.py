"""Entity extraction using spaCy NER, POS tagging, and custom patterns."""
import re

import spacy

from app import models_state

# Common tech skills for matching
KNOWN_SKILLS = {
    "python", "java", "javascript", "typescript", "react", "angular", "vue",
    "node.js", "nodejs", "express", "django", "flask", "fastapi", "spring",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "redis", "elasticsearch",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins",
    "git", "github", "gitlab", "ci/cd", "devops", "linux", "bash",
    "html", "css", "sass", "tailwind", "bootstrap", "webpack", "vite",
    "rest", "rest apis", "graphql", "grpc", "microservices",
    "machine learning", "deep learning", "nlp", "data science",
    "agile", "scrum", "kanban", "jira", "confluence",
    "c++", "c#", "go", "rust", "ruby", "php", "swift", "kotlin",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "figma", "sketch", "adobe", "photoshop",
    "excel", "powerpoint", "tableau", "power bi",
    "salesforce", "sap", "oracle",
}

DEGREE_PATTERNS = re.compile(
    r"(?:bachelor(?:\'s)?|master(?:\'s)?|b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?|"
    r"ph\.?d\.?|mba|associate(?:\'s)?|doctorate)\s*(?:of\s+(?:science|arts|engineering|business))?"
    r"(?:\s+in\s+[\w\s&]+)?",
    re.IGNORECASE,
)


def _get_nlp():
    """Get spaCy model, preferring the preloaded one."""
    if models_state.nlp_model is not None:
        return models_state.nlp_model
    return spacy.load("en_core_web_sm")


def extract_skills(text: str) -> list[str]:
    """Extract technical skills using NLP and known skill matching.

    Uses: POS tagging, noun phrase chunking, pattern matching.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    found_skills = set()

    # Method 1: Match against known skills list
    text_lower = text.lower()
    for skill in KNOWN_SKILLS:
        if skill in text_lower:
            found_skills.add(skill.title() if len(skill) > 3 else skill.upper())

    # Method 2: Extract noun phrases (noun phrase chunking)
    for chunk in doc.noun_chunks:
        chunk_lower = chunk.text.lower().strip()
        if chunk_lower in KNOWN_SKILLS:
            found_skills.add(chunk.text.strip())

    # Method 3: Look for comma-separated lists after skill-related headers
    skill_list_pattern = re.compile(
        r"(?:skills|technologies|tech stack|tools)\s*:?\s*(.+?)(?:\n\n|\n[A-Z]|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in skill_list_pattern.finditer(text):
        items = re.split(r"[,\|;]", match.group(1))
        for item in items:
            cleaned = item.strip().strip("-•* ")
            if cleaned and 1 < len(cleaned) < 40:
                if cleaned.lower() in KNOWN_SKILLS:
                    found_skills.add(cleaned.strip())
                elif cleaned[0].isupper() and len(cleaned.split()) <= 3:
                    found_skills.add(cleaned.strip())

    return sorted(found_skills)


def extract_education(text: str) -> list[str]:
    """Extract education items using regex patterns and NER."""
    found = set()

    for match in DEGREE_PATTERNS.finditer(text):
        degree = match.group(0).strip()
        if len(degree) > 3:
            found.add(degree)

    return sorted(found)


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract all entity types from text.

    Returns dict with keys: skills, organizations, education.
    Uses: spaCy NER, POS tagging, noun phrase chunking, regex patterns.
    """
    nlp = _get_nlp()
    doc = nlp(text)

    organizations = set()
    for ent in doc.ents:
        if ent.label_ == "ORG":
            organizations.add(ent.text.strip())

    return {
        "skills": extract_skills(text),
        "organizations": sorted(organizations),
        "education": extract_education(text),
    }
