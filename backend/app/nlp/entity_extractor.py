"""Entity extraction using spaCy NER, POS tagging, and custom patterns."""
import re
from functools import lru_cache

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

DISPLAY_OVERRIDES = {
    "agile": "Agile",
    "aws": "AWS",
    "azure": "Azure",
    "bash": "Bash",
    "c#": "C#",
    "c++": "C++",
    "ci/cd": "CI/CD",
    "css": "CSS",
    "devops": "DevOps",
    "django": "Django",
    "docker": "Docker",
    "excel": "Excel",
    "fastapi": "FastAPI",
    "figma": "Figma",
    "gcp": "GCP",
    "git": "Git",
    "github": "GitHub",
    "gitlab": "GitLab",
    "go": "Go",
    "graphql": "GraphQL",
    "grpc": "gRPC",
    "html": "HTML",
    "java": "Java",
    "javascript": "JavaScript",
    "jenkins": "Jenkins",
    "jira": "Jira",
    "kubernetes": "Kubernetes",
    "linux": "Linux",
    "machine learning": "Machine Learning",
    "microservices": "Microservices",
    "mongodb": "MongoDB",
    "mysql": "MySQL",
    "nlp": "NLP",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "nosql": "NoSQL",
    "numpy": "NumPy",
    "oracle": "Oracle",
    "pandas": "Pandas",
    "php": "PHP",
    "postgresql": "PostgreSQL",
    "power bi": "Power BI",
    "powerpoint": "PowerPoint",
    "python": "Python",
    "pytorch": "PyTorch",
    "react": "React",
    "redis": "Redis",
    "rest": "REST",
    "rest apis": "REST APIs",
    "ruby": "Ruby",
    "rust": "Rust",
    "salesforce": "Salesforce",
    "sap": "SAP",
    "sass": "Sass",
    "scikit-learn": "scikit-learn",
    "scrum": "Scrum",
    "sketch": "Sketch",
    "sql": "SQL",
    "swift": "Swift",
    "tableau": "Tableau",
    "tailwind": "Tailwind",
    "tensorflow": "TensorFlow",
    "terraform": "Terraform",
    "typescript": "TypeScript",
    "vite": "Vite",
    "vue": "Vue",
    "webpack": "Webpack",
}


def _get_nlp():
    """Get spaCy model, preferring the preloaded one."""
    if models_state.nlp_model is not None:
        return models_state.nlp_model
    return spacy.load("en_core_web_sm")


def _format_skill(skill: str) -> str:
    normalized = skill.strip().lower()
    return DISPLAY_OVERRIDES.get(normalized, skill.strip())


@lru_cache(maxsize=None)
def _compile_skill_pattern(skill: str) -> re.Pattern[str]:
    escaped = re.escape(skill)
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


def _contains_skill(text: str, skill: str) -> bool:
    return bool(_compile_skill_pattern(skill).search(text))


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
        if _contains_skill(text_lower, skill):
            found_skills.add(_format_skill(skill))

    # Method 2: Extract noun phrases (noun phrase chunking)
    for chunk in doc.noun_chunks:
        chunk_lower = chunk.text.lower().strip()
        if chunk_lower in KNOWN_SKILLS:
            found_skills.add(_format_skill(chunk_lower))

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
                    found_skills.add(_format_skill(cleaned))
                elif cleaned[0].isupper() and len(cleaned.split()) <= 3:
                    found_skills.add(cleaned.strip())

    return sorted({_format_skill(skill) for skill in found_skills}, key=str.lower)


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
        cleaned = ent.text.strip()
        if ent.label_ == "ORG" and cleaned and "," not in cleaned and cleaned.lower() not in KNOWN_SKILLS:
            organizations.add(ent.text.strip())

    return {
        "skills": extract_skills(text),
        "organizations": sorted(organizations),
        "education": extract_education(text),
    }
