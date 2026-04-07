"""Parse job descriptions and resumes into structured sections."""
import re


# JD section header patterns
JD_SECTION_PATTERNS = {
    "requirements": re.compile(
        r"(?:^|\n)\s*(?:requirements|qualifications|required skills|what you\'ll need|"
        r"what we\'re looking for|must have|minimum qualifications|basic qualifications)\s*:?\s*\n",
        re.IGNORECASE,
    ),
    "responsibilities": re.compile(
        r"(?:^|\n)\s*(?:responsibilities|what you\'ll do|role description|"
        r"job duties|key responsibilities|the role|about the role)\s*:?\s*\n",
        re.IGNORECASE,
    ),
    "preferred": re.compile(
        r"(?:^|\n)\s*(?:preferred|nice to have|bonus|desired|preferred qualifications|"
        r"additional qualifications|it\'s a plus if)\s*:?\s*\n",
        re.IGNORECASE,
    ),
    "other": re.compile(
        r"(?:^|\n)\s*(?:about the company|about us|benefits|perks|compensation|"
        r"why join|our team|company overview)\s*:?\s*\n",
        re.IGNORECASE,
    ),
}

# Resume section header patterns
RESUME_SECTION_PATTERNS = {
    "summary": re.compile(
        r"(?:^|\n)\s*(?:summary|objective|profile|about me|professional summary)\s*:?\s*\n",
        re.IGNORECASE,
    ),
    "experience": re.compile(
        r"(?:^|\n)\s*(?:experience|work experience|employment|professional experience|work history)\s*:?\s*\n",
        re.IGNORECASE,
    ),
    "skills": re.compile(
        r"(?:^|\n)\s*(?:skills|technical skills|core competencies|technologies|tech stack)\s*:?\s*\n",
        re.IGNORECASE,
    ),
    "education": re.compile(
        r"(?:^|\n)\s*(?:education|academic|degrees|certifications|certifications & education)\s*:?\s*\n",
        re.IGNORECASE,
    ),
}


def _extract_items(text: str) -> list[str]:
    """Extract individual items from a text block (bullet points or lines)."""
    lines = text.strip().split("\n")
    items = []
    for line in lines:
        line = re.sub(r"^\s*(?:[-*•–—]+|\d+[.)])\s*", "", line).strip()
        if line and len(line) > 3:
            items.append(line)
    return items


def _find_sections(text: str, patterns: dict[str, re.Pattern]) -> dict[str, str]:
    """Find sections in text using regex patterns. Returns raw text per section."""
    matches = []
    for name, pattern in patterns.items():
        for match in pattern.finditer(text):
            matches.append((match.start(), match.end(), name))

    matches.sort(key=lambda x: x[0])

    sections = {}
    for i, (start, header_end, name) in enumerate(matches):
        if i + 1 < len(matches):
            section_text = text[header_end : matches[i + 1][0]]
        else:
            section_text = text[header_end:]
        sections[name] = section_text.strip()

    return sections


def parse_job_description(text: str) -> dict[str, list[str]]:
    """Parse a job description into structured sections.

    Returns dict with keys: requirements, responsibilities, preferred, other.
    Each value is a list of extracted items/sentences.
    """
    result = {
        "requirements": [],
        "responsibilities": [],
        "preferred": [],
        "other": [],
    }

    raw_sections = _find_sections(text, JD_SECTION_PATTERNS)

    if raw_sections:
        for key in result:
            if key in raw_sections:
                result[key] = _extract_items(raw_sections[key])

        # Any text before the first section header goes to "other"
        first_match_pos = len(text)
        for pattern in JD_SECTION_PATTERNS.values():
            m = pattern.search(text)
            if m and m.start() < first_match_pos:
                first_match_pos = m.start()

        preamble = text[:first_match_pos].strip()
        if preamble:
            result["other"].extend(_extract_items(preamble))
    else:
        # Unstructured JD — put everything in requirements as fallback
        items = _extract_items(text)
        result["requirements"] = items

    return result


def parse_resume(text: str) -> dict[str, str]:
    """Parse a resume into structured sections.

    Returns dict with keys: summary, experience, skills, education.
    Each value is the raw text of that section.
    """
    result = {
        "summary": "",
        "experience": "",
        "skills": "",
        "education": "",
    }

    raw_sections = _find_sections(text, RESUME_SECTION_PATTERNS)

    if raw_sections:
        for key in result:
            if key in raw_sections:
                result[key] = raw_sections[key]
    else:
        result["summary"] = text.strip()

    return result
