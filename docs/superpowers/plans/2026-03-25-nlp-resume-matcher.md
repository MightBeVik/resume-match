# NLP Resume Matcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resume-to-job-description matching tool with a Python NLP backend (FastAPI) and a redesigned Next.js frontend, showcasing 10+ NLP techniques for an NLP course.

**Architecture:** FastAPI backend in `backend/` with isolated venv handles all NLP processing via a single `POST /api/analyze` endpoint. Next.js frontend in `src/` provides a two-page flow (input → results). Models loaded once at startup.

**Tech Stack:** Python (FastAPI, spaCy, NLTK, scikit-learn, sentence-transformers), Next.js 16, React 19, Tailwind CSS 4, TypeScript

**Spec:** `docs/superpowers/specs/2026-03-25-nlp-resume-matcher-design.md`

---

## File Structure

### Backend (`backend/`)

| File | Responsibility |
|------|---------------|
| `backend/requirements.txt` | Python dependencies |
| `backend/run.py` | Entry point — starts uvicorn server |
| `backend/app/__init__.py` | Package init |
| `backend/app/main.py` | FastAPI app, CORS config, model preloading |
| `backend/app/api/__init__.py` | Package init |
| `backend/app/api/routes.py` | `POST /api/analyze` endpoint, validation, error handling |
| `backend/app/models/__init__.py` | Package init |
| `backend/app/models/schemas.py` | Pydantic request/response models |
| `backend/app/nlp/__init__.py` | Package init |
| `backend/app/nlp/preprocessor.py` | Tokenization, stopword removal, lemmatization, sentence segmentation |
| `backend/app/nlp/section_parser.py` | JD section splitting + resume section parsing |
| `backend/app/nlp/entity_extractor.py` | spaCy NER, POS tagging, noun phrase chunking, skill/education extraction |
| `backend/app/nlp/keyword_extractor.py` | TF-IDF vectorization, weighted keyword lists |
| `backend/app/nlp/similarity.py` | Cosine similarity (TF-IDF) + semantic similarity (sentence-transformers) |
| `backend/app/nlp/matcher.py` | Orchestrator — runs full pipeline, computes weighted scores, builds response |
| `backend/tests/__init__.py` | Package init |
| `backend/tests/test_preprocessor.py` | Tests for preprocessing |
| `backend/tests/test_section_parser.py` | Tests for section parsing |
| `backend/tests/test_entity_extractor.py` | Tests for entity extraction |
| `backend/tests/test_keyword_extractor.py` | Tests for TF-IDF keyword extraction |
| `backend/tests/test_similarity.py` | Tests for similarity scoring |
| `backend/tests/test_matcher.py` | Tests for the orchestrator |
| `backend/tests/test_api.py` | Integration tests for the API endpoint |

### Frontend (`src/`)

| File | Responsibility |
|------|---------------|
| `src/app/layout.tsx` | Root layout with fonts, metadata |
| `src/app/globals.css` | Tailwind imports, CSS variables, base styles |
| `src/app/page.tsx` | Input page — resume + JD text areas, analyze button |
| `src/app/results/page.tsx` | Results page — score gauge, accordions, NLP details |
| `src/components/ui/score-gauge.tsx` | Semi-circular SVG arc gauge |
| `src/components/ui/accordion.tsx` | Expandable accordion component |
| `src/components/ui/badge.tsx` | Color-coded match badges (green/yellow/red) |
| `src/components/ui/progress-bar.tsx` | Sub-score progress bars |
| `src/components/results/hero-score.tsx` | Hero section: gauge + verdict + summary + mini bars |
| `src/components/results/section-accordion.tsx` | Accordion row for a match section (Skills/Experience/etc.) |
| `src/components/results/nlp-details-panel.tsx` | Expandable NLP pipeline details panel |

---

## Task 1: Clean Up Existing Code & Set Up Backend Skeleton

**Files:**
- Delete: `src/components/matcher/`, `src/components/rewrite/`, `src/components/data-grid/`, `src/components/header.tsx`, `src/app/data-grid/`, `src/lib/utils.ts`
- Create: `backend/requirements.txt`, `backend/run.py`, `backend/app/__init__.py`, `backend/app/main.py`

- [ ] **Step 1: Delete existing frontend components**

```bash
rm -rf src/components/matcher src/components/rewrite src/components/data-grid src/components/header.tsx src/lib src/app/data-grid
```

- [ ] **Step 2: Clean up package.json — remove unused deps**

Remove `@dnd-kit/core`, `@dnd-kit/sortable`, `@dnd-kit/utilities`, `@tanstack/react-table`, `class-variance-authority` from `package.json` — they were for the old data-grid feature. Keep `clsx`, `tailwind-merge`, `lucide-react`, and all Next/React/Tailwind deps.

```bash
cd /Users/nateogunleye/Desktop/resume-matcher && npm uninstall @dnd-kit/core @dnd-kit/sortable @dnd-kit/utilities @tanstack/react-table class-variance-authority
```

- [ ] **Step 3: Create backend directory structure**

```bash
mkdir -p backend/app/api backend/app/models backend/app/nlp backend/tests
touch backend/app/__init__.py backend/app/api/__init__.py backend/app/models/__init__.py backend/app/nlp/__init__.py backend/tests/__init__.py
```

- [ ] **Step 4: Create `backend/requirements.txt`**

```
fastapi==0.115.12
uvicorn==0.34.2
pydantic==2.11.3
spacy==3.8.7
nltk==3.9.1
scikit-learn==1.6.1
sentence-transformers==4.1.0
pytest==8.3.5
httpx==0.28.1
```

- [ ] **Step 5: Create `backend/run.py`**

```python
"""Entry point for the FastAPI backend."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
```

- [ ] **Step 6: Create `backend/app/main.py`**

```python
"""FastAPI application with CORS and model preloading."""
from contextlib import asynccontextmanager

import nltk
import spacy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer


# Global model references — loaded once at startup
nlp_model = None
sentence_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load NLP models at startup."""
    global nlp_model, sentence_model

    # Download NLTK data
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    # Load spaCy model
    try:
        nlp_model = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp_model = spacy.load("en_core_web_sm")

    # Load sentence-transformers model
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    yield


app = FastAPI(title="ResumeMatch API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.routes import router  # noqa: E402
app.include_router(router, prefix="/api")
```

- [ ] **Step 7: Create placeholder `backend/app/api/routes.py`**

```python
"""API route definitions."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok"}
```

- [ ] **Step 8: Create virtual environment and install deps**

```bash
cd /Users/nateogunleye/Desktop/resume-matcher/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- [ ] **Step 9: Verify backend starts**

```bash
cd /Users/nateogunleye/Desktop/resume-matcher/backend
source venv/bin/activate
timeout 30 python run.py &
sleep 15
curl http://localhost:8000/api/health
kill %1
```

Expected: `{"status":"ok"}`

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "chore: clean up old code, set up backend skeleton with FastAPI"
```

---

## Task 2: Pydantic Schemas

**Files:**
- Create: `backend/app/models/schemas.py`

- [ ] **Step 1: Create Pydantic models**

```python
"""Request and response schemas for the analyze endpoint."""
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    resume_text: str = Field(..., min_length=50, max_length=15000)
    job_description: str = Field(..., min_length=50, max_length=10000)


class SectionResult(BaseModel):
    score: int = Field(..., ge=0, le=100)
    matched: list[str] = []
    partial: list[str] = []
    missing: list[str] = []


class TfidfKeyword(BaseModel):
    keyword: str
    weight: float


class SimilarityScores(BaseModel):
    tfidf_cosine: float
    semantic: float


class NlpDetails(BaseModel):
    jd_sections_parsed: dict[str, list[str]]
    resume_sections_parsed: dict[str, str]
    resume_entities: dict[str, list[str]]
    tfidf_top_keywords: dict[str, list[TfidfKeyword]]
    similarity_scores: SimilarityScores


class AnalyzeResponse(BaseModel):
    overall_score: int = Field(..., ge=0, le=100)
    verdict: str
    summary: str
    sections: dict[str, SectionResult]
    nlp_details: NlpDetails
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/models/schemas.py
git commit -m "feat: add Pydantic request/response schemas"
```

---

## Task 3: Preprocessor

**Files:**
- Create: `backend/app/nlp/preprocessor.py`, `backend/tests/test_preprocessor.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for text preprocessing."""
import pytest
from app.nlp.preprocessor import preprocess_text, tokenize, remove_stopwords, lemmatize, segment_sentences


def test_tokenize_splits_words():
    tokens = tokenize("Hello world, this is a test.")
    assert "Hello" in tokens
    assert "world" in tokens
    assert "." in tokens


def test_remove_stopwords_filters_common_words():
    tokens = ["this", "is", "a", "great", "test"]
    filtered = remove_stopwords(tokens)
    assert "great" in filtered
    assert "test" in filtered
    assert "this" not in filtered
    assert "is" not in filtered


def test_lemmatize_normalizes_words():
    tokens = ["running", "studies", "better"]
    lemmatized = lemmatize(tokens)
    assert "run" in lemmatized or "running" in lemmatized
    assert "study" in lemmatized


def test_segment_sentences():
    text = "First sentence. Second sentence. Third one here."
    sentences = segment_sentences(text)
    assert len(sentences) >= 3


def test_preprocess_text_full_pipeline():
    text = "The engineers are running complex distributed systems."
    result = preprocess_text(text)
    assert isinstance(result["tokens"], list)
    assert isinstance(result["lemmatized"], list)
    assert isinstance(result["sentences"], list)
    assert len(result["tokens"]) > 0
    # Stopwords should be removed
    assert "the" not in result["lemmatized"]
    assert "are" not in result["lemmatized"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/nateogunleye/Desktop/resume-matcher/backend
source venv/bin/activate
python -m pytest tests/test_preprocessor.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: Implement preprocessor**

```python
"""Text preprocessing: tokenization, stopword removal, lemmatization, sentence segmentation."""
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def tokenize(text: str) -> list[str]:
    """Split text into word tokens using NLTK word_tokenize."""
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove common English stopwords from token list."""
    stop_words = set(stopwords.words("english"))
    return [t for t in tokens if t.lower() not in stop_words]


def lemmatize(tokens: list[str]) -> list[str]:
    """Reduce words to base form using WordNet lemmatizer."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t.lower(), pos="v") for t in tokens]


def segment_sentences(text: str) -> list[str]:
    """Split text into sentences using spaCy."""
    import spacy
    from app.main import nlp_model

    if nlp_model is None:
        # Fallback for testing without app startup
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Final fallback: use NLTK sentence tokenizer
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
    else:
        nlp = nlp_model

    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def preprocess_text(text: str) -> dict:
    """Run full preprocessing pipeline on input text.

    Returns dict with keys: tokens, filtered, lemmatized, sentences
    """
    tokens = tokenize(text)
    filtered = remove_stopwords(tokens)
    lemmatized = lemmatize(filtered)
    sentences = segment_sentences(text)

    return {
        "tokens": tokens,
        "filtered": filtered,
        "lemmatized": lemmatized,
        "sentences": sentences,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/nateogunleye/Desktop/resume-matcher/backend
source venv/bin/activate
python -m pytest tests/test_preprocessor.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/nlp/preprocessor.py backend/tests/test_preprocessor.py
git commit -m "feat: add text preprocessor with tokenization, stopwords, lemmatization"
```

---

## Task 4: Section Parser

**Files:**
- Create: `backend/app/nlp/section_parser.py`, `backend/tests/test_section_parser.py`

- [ ] **Step 1: Write tests**

```python
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
    # At least some content should be extracted
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/nateogunleye/Desktop/resume-matcher/backend
source venv/bin/activate
python -m pytest tests/test_section_parser.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement section parser**

```python
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
        line = re.sub(r"^[\s\-\*\•\–\—\d.]+", "", line).strip()
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
        # Structured JD — extract items from each section
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
        # No clear sections found — use full text as summary
        result["summary"] = text.strip()

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/nateogunleye/Desktop/resume-matcher/backend
source venv/bin/activate
python -m pytest tests/test_section_parser.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/nlp/section_parser.py backend/tests/test_section_parser.py
git commit -m "feat: add JD and resume section parser with structured/unstructured support"
```

---

## Task 5: Entity Extractor

**Files:**
- Create: `backend/app/nlp/entity_extractor.py`, `backend/tests/test_entity_extractor.py`

- [ ] **Step 1: Write tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/nateogunleye/Desktop/resume-matcher/backend
source venv/bin/activate
python -m pytest tests/test_entity_extractor.py -v
```

- [ ] **Step 3: Implement entity extractor**

```python
"""Entity extraction using spaCy NER, POS tagging, and custom patterns."""
import re

import spacy

# Common tech skills for matching (expandable list)
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
    from app.main import nlp_model
    if nlp_model is not None:
        return nlp_model
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
                # Check if it's a known skill or a proper noun
                if cleaned.lower() in KNOWN_SKILLS:
                    found_skills.add(cleaned.strip())
                elif cleaned[0].isupper() and len(cleaned.split()) <= 3:
                    found_skills.add(cleaned.strip())

    return sorted(found_skills)


def extract_education(text: str) -> list[str]:
    """Extract education items using regex patterns and NER."""
    found = set()

    # Pattern matching for degree mentions
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

    # spaCy NER for organizations
    organizations = set()
    for ent in doc.ents:
        if ent.label_ == "ORG":
            organizations.add(ent.text.strip())

    return {
        "skills": extract_skills(text),
        "organizations": sorted(organizations),
        "education": extract_education(text),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/nateogunleye/Desktop/resume-matcher/backend
source venv/bin/activate
python -m pytest tests/test_entity_extractor.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/nlp/entity_extractor.py backend/tests/test_entity_extractor.py
git commit -m "feat: add entity extractor with spaCy NER, POS tagging, skill matching"
```

---

## Task 6: Keyword Extractor (TF-IDF)

**Files:**
- Create: `backend/app/nlp/keyword_extractor.py`, `backend/tests/test_keyword_extractor.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for TF-IDF keyword extraction."""
from app.nlp.keyword_extractor import extract_keywords, get_tfidf_vectors


def test_extract_keywords_returns_weighted_list():
    text = "React TypeScript frontend development web applications JavaScript"
    keywords = extract_keywords(text, top_n=5)
    assert len(keywords) > 0
    assert len(keywords) <= 5
    # Each keyword should be a dict with keyword and weight
    assert "keyword" in keywords[0]
    assert "weight" in keywords[0]


def test_extract_keywords_sorted_by_weight():
    text = "React React React TypeScript frontend development JavaScript Python"
    keywords = extract_keywords(text, top_n=5)
    weights = [k["weight"] for k in keywords]
    assert weights == sorted(weights, reverse=True)


def test_get_tfidf_vectors_returns_matrix_and_names():
    texts = ["React TypeScript frontend", "Python Django backend"]
    matrix, feature_names = get_tfidf_vectors(texts)
    assert matrix.shape[0] == 2
    assert len(feature_names) > 0


def test_extract_keywords_handles_empty_text():
    keywords = extract_keywords("", top_n=5)
    assert keywords == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/nateogunleye/Desktop/resume-matcher/backend
source venv/bin/activate
python -m pytest tests/test_keyword_extractor.py -v
```

- [ ] **Step 3: Implement keyword extractor**

```python
"""TF-IDF keyword extraction using scikit-learn."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf_vectors(texts: list[str]) -> tuple:
    """Compute TF-IDF vectors for a list of texts.

    Returns: (tfidf_matrix, feature_names)
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=500,
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return matrix, feature_names


def extract_keywords(text: str, top_n: int = 15) -> list[dict]:
    """Extract top-N keywords from text ranked by TF-IDF weight.

    Returns list of {"keyword": str, "weight": float} sorted by weight desc.
    """
    if not text.strip():
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=200,
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()

    scores = matrix.toarray()[0]
    top_indices = np.argsort(scores)[::-1][:top_n]

    return [
        {"keyword": feature_names[i], "weight": round(float(scores[i]), 4)}
        for i in top_indices
        if scores[i] > 0
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/nateogunleye/Desktop/resume-matcher/backend
source venv/bin/activate
python -m pytest tests/test_keyword_extractor.py -v
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/nlp/keyword_extractor.py backend/tests/test_keyword_extractor.py
git commit -m "feat: add TF-IDF keyword extractor"
```

---

## Task 7: Similarity Scorer

**Files:**
- Create: `backend/app/nlp/similarity.py`, `backend/tests/test_similarity.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for cosine and semantic similarity."""
from app.nlp.similarity import tfidf_cosine_similarity, semantic_similarity, item_semantic_similarity


def test_tfidf_cosine_identical_texts():
    score = tfidf_cosine_similarity("React TypeScript frontend", "React TypeScript frontend")
    assert score > 0.9


def test_tfidf_cosine_similar_texts():
    score = tfidf_cosine_similarity(
        "Experienced React frontend developer",
        "Looking for React frontend engineer"
    )
    assert score > 0.3


def test_tfidf_cosine_different_texts():
    score = tfidf_cosine_similarity(
        "React TypeScript frontend web development",
        "Cooking recipes Italian pasta baking"
    )
    assert score < 0.2


def test_semantic_similarity_similar_meaning():
    score = semantic_similarity(
        "Experienced software engineer with team leadership",
        "Looking for a developer who can lead engineering teams"
    )
    assert score > 0.4


def test_semantic_similarity_different_meaning():
    score = semantic_similarity(
        "Python machine learning data science",
        "Cooking Italian food recipes pasta"
    )
    assert score < 0.3


def test_item_semantic_similarity():
    score = item_semantic_similarity("Docker", "containerization")
    assert score > 0.3  # Related concepts


def test_item_semantic_similarity_identical():
    score = item_semantic_similarity("React", "React")
    assert score > 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/nateogunleye/Desktop/resume-matcher/backend
source venv/bin/activate
python -m pytest tests/test_similarity.py -v
```

- [ ] **Step 3: Implement similarity scorer**

```python
"""Similarity scoring using TF-IDF cosine similarity and sentence-transformers."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _get_sentence_model():
    """Get sentence-transformers model, preferring the preloaded one."""
    from app.main import sentence_model
    if sentence_model is not None:
        return sentence_model
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using TF-IDF vectors."""
    if not text_a.strip() or not text_b.strip():
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([text_a, text_b])
    sim = cosine_similarity(matrix[0:1], matrix[1:2])
    return float(sim[0][0])


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute semantic similarity using sentence-transformers embeddings."""
    if not text_a.strip() or not text_b.strip():
        return 0.0

    model = _get_sentence_model()
    embeddings = model.encode([text_a, text_b])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])
    return float(sim[0][0])


def item_semantic_similarity(item_a: str, item_b: str) -> float:
    """Compute semantic similarity between two short items (skills, titles, etc.)."""
    return semantic_similarity(item_a, item_b)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/nateogunleye/Desktop/resume-matcher/backend
source venv/bin/activate
python -m pytest tests/test_similarity.py -v
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/nlp/similarity.py backend/tests/test_similarity.py
git commit -m "feat: add TF-IDF cosine and semantic similarity scoring"
```

---

## Task 8: Matcher Orchestrator

**Files:**
- Create: `backend/app/nlp/matcher.py`, `backend/tests/test_matcher.py`

- [ ] **Step 1: Write tests**

```python
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
    """A resume closely matching the JD should score above 50."""
    result = analyze_match(RESUME, JOB_DESCRIPTION)
    assert result["overall_score"] >= 50


def test_analyze_match_poor_resume_scores_low():
    """A completely unrelated resume should score low."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/nateogunleye/Desktop/resume-matcher/backend
source venv/bin/activate
python -m pytest tests/test_matcher.py -v
```

- [ ] **Step 3: Implement matcher orchestrator**

```python
"""Matcher orchestrator — runs the full NLP pipeline and computes weighted scores."""
from app.nlp.preprocessor import preprocess_text
from app.nlp.section_parser import parse_job_description, parse_resume
from app.nlp.entity_extractor import extract_entities, extract_skills
from app.nlp.keyword_extractor import extract_keywords
from app.nlp.similarity import tfidf_cosine_similarity, semantic_similarity, item_semantic_similarity


# Score weights
WEIGHTS = {
    "skills": 0.40,
    "experience": 0.25,
    "education": 0.15,
    "preferred": 0.10,
    "semantic": 0.10,
}


def _get_verdict(score: int) -> str:
    """Map overall score to verdict string."""
    if score >= 70:
        return "Strong Match"
    elif score >= 40:
        return "Moderate Match"
    return "Weak Match"


def _generate_summary(score: int, sections: dict) -> str:
    """Generate a human-readable summary of the match."""
    verdict = _get_verdict(score)

    # Find strongest and weakest sections
    section_scores = {k: v["score"] for k, v in sections.items()}
    strongest = max(section_scores, key=section_scores.get)
    weakest = min(section_scores, key=section_scores.get)

    total_matched = sum(len(v["matched"]) for v in sections.values())
    total_missing = sum(len(v["missing"]) for v in sections.values())

    if score >= 70:
        summary = f"Your resume is a strong match for this role with {total_matched} matching qualifications. "
        summary += f"Your strongest area is {strongest} ({section_scores[strongest]}%). "
        if total_missing > 0:
            summary += f"Consider addressing {total_missing} missing qualification(s) in {weakest} to strengthen your application."
    elif score >= 40:
        summary = f"Your resume moderately matches this role with {total_matched} matching qualifications. "
        summary += f"Your strongest area is {strongest} ({section_scores[strongest]}%). "
        summary += f"Focus on improving {weakest} ({section_scores[weakest]}%) where you have {len(sections[weakest]['missing'])} gap(s)."
    else:
        summary = f"Your resume has limited alignment with this role. "
        summary += f"You matched {total_matched} qualification(s) but are missing {total_missing}. "
        summary += f"This role may require significant additional experience or skills."

    return summary


def _match_items(jd_items: list[str], resume_text: str, resume_skills: list[str]) -> dict:
    """Match JD requirement items against resume content.

    Returns: {"score": int, "matched": [...], "partial": [...], "missing": [...]}
    """
    if not jd_items:
        return {"score": 100, "matched": [], "partial": [], "missing": []}

    matched = []
    partial = []
    missing = []

    resume_lower = resume_text.lower()
    resume_skills_lower = [s.lower() for s in resume_skills]

    for item in jd_items:
        item_clean = item.strip()
        if not item_clean:
            continue

        item_lower = item_clean.lower()

        # Check 1: Direct text match (exact or substring)
        if item_lower in resume_lower:
            matched.append(item_clean)
            continue

        # Check 2: Word-level overlap check
        item_words = set(item_lower.split())
        # Remove very short/common words for matching
        item_words = {w for w in item_words if len(w) > 2}

        if item_words:
            overlap = sum(1 for w in item_words if w in resume_lower)
            overlap_ratio = overlap / len(item_words)
            if overlap_ratio > 0.6:
                matched.append(item_clean)
                continue

        # Check 3: Skill list match
        found_in_skills = any(
            skill_lower in item_lower or item_lower in skill_lower
            for skill_lower in resume_skills_lower
        )
        if found_in_skills:
            matched.append(item_clean)
            continue

        # Check 4: Semantic similarity
        sim = item_semantic_similarity(item_clean, resume_text[:500])
        if sim > 0.8:
            matched.append(item_clean)
        elif sim > 0.5:
            partial.append(item_clean)
        else:
            missing.append(item_clean)

    total = len(matched) + len(partial) + len(missing)
    if total == 0:
        score = 100
    else:
        score = int(((len(matched) + 0.5 * len(partial)) / total) * 100)

    return {
        "score": min(score, 100),
        "matched": matched,
        "partial": partial,
        "missing": missing,
    }


def analyze_match(resume_text: str, job_description: str) -> dict:
    """Run the full NLP analysis pipeline.

    Steps:
    1. Preprocess both texts
    2. Parse into sections
    3. Extract entities
    4. Extract TF-IDF keywords
    5. Compute per-section matches
    6. Compute similarities
    7. Build weighted final score

    Returns the full response dict matching the API schema.
    """
    # Step 1: Preprocess
    resume_processed = preprocess_text(resume_text)
    jd_processed = preprocess_text(job_description)

    # Step 2: Parse sections
    jd_sections = parse_job_description(job_description)
    resume_sections = parse_resume(resume_text)

    # Step 3: Extract entities
    resume_entities = extract_entities(resume_text)
    resume_skills = resume_entities["skills"]

    # Full resume text for matching
    full_resume = resume_text

    # Step 4: TF-IDF keywords
    jd_keywords = extract_keywords(job_description, top_n=15)
    resume_keywords = extract_keywords(resume_text, top_n=15)

    # Step 5: Per-section matching
    skills_result = _match_items(
        jd_sections["requirements"],
        full_resume,
        resume_skills,
    )

    experience_result = _match_items(
        jd_sections["responsibilities"],
        resume_sections.get("experience", "") + " " + resume_sections.get("summary", ""),
        resume_skills,
    )

    education_result = _match_items(
        [item for item in jd_sections["requirements"] if _is_education_item(item)]
        or _extract_education_requirements(job_description),
        resume_sections.get("education", "") + " " + full_resume,
        resume_skills,
    )

    preferred_result = _match_items(
        jd_sections["preferred"],
        full_resume,
        resume_skills,
    )

    # Step 6: Similarities
    tfidf_sim = tfidf_cosine_similarity(resume_text, job_description)
    sem_sim = semantic_similarity(resume_text, job_description)

    # Step 7: Weighted final score
    semantic_score = int(sem_sim * 100)
    overall_score = int(
        skills_result["score"] * WEIGHTS["skills"]
        + experience_result["score"] * WEIGHTS["experience"]
        + education_result["score"] * WEIGHTS["education"]
        + preferred_result["score"] * WEIGHTS["preferred"]
        + semantic_score * WEIGHTS["semantic"]
    )
    overall_score = max(0, min(100, overall_score))

    sections = {
        "skills": skills_result,
        "experience": experience_result,
        "education": education_result,
        "preferred": preferred_result,
    }

    verdict = _get_verdict(overall_score)
    summary = _generate_summary(overall_score, sections)

    return {
        "overall_score": overall_score,
        "verdict": verdict,
        "summary": summary,
        "sections": sections,
        "nlp_details": {
            "jd_sections_parsed": {k: v for k, v in jd_sections.items()},
            "resume_sections_parsed": {k: v for k, v in resume_sections.items()},
            "resume_entities": resume_entities,
            "tfidf_top_keywords": {
                "job_description": jd_keywords,
                "resume": resume_keywords,
            },
            "similarity_scores": {
                "tfidf_cosine": round(tfidf_sim, 4),
                "semantic": round(sem_sim, 4),
            },
        },
    }


def _is_education_item(item: str) -> bool:
    """Check if a JD requirement item is education-related."""
    edu_keywords = ["degree", "bachelor", "master", "phd", "b.s.", "m.s.", "mba",
                    "education", "university", "college", "diploma", "certified"]
    return any(kw in item.lower() for kw in edu_keywords)


def _extract_education_requirements(text: str) -> list[str]:
    """Extract education requirements from full JD text as fallback."""
    import re
    edu_patterns = re.compile(
        r"(?:bachelor|master|phd|b\.s\.|m\.s\.|mba|degree)[\w\s,'.]*(?:required|preferred|or equivalent)?",
        re.IGNORECASE,
    )
    matches = edu_patterns.findall(text)
    return [m.strip() for m in matches if len(m.strip()) > 5]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/nateogunleye/Desktop/resume-matcher/backend
source venv/bin/activate
python -m pytest tests/test_matcher.py -v
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/nlp/matcher.py backend/tests/test_matcher.py
git commit -m "feat: add matcher orchestrator with weighted scoring pipeline"
```

---

## Task 9: API Endpoint + Integration Tests

**Files:**
- Modify: `backend/app/api/routes.py`
- Create: `backend/tests/test_api.py`

- [ ] **Step 1: Write integration tests**

```python
"""Integration tests for the /api/analyze endpoint."""
import pytest
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/nateogunleye/Desktop/resume-matcher/backend
source venv/bin/activate
python -m pytest tests/test_api.py -v
```

- [ ] **Step 3: Implement the analyze endpoint**

Replace `backend/app/api/routes.py` with:

```python
"""API route definitions."""
from fastapi import APIRouter, HTTPException
from app.models.schemas import AnalyzeRequest, AnalyzeResponse
from app.nlp.matcher import analyze_match

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Analyze resume against job description using NLP pipeline."""
    try:
        result = analyze_match(request.resume_text, request.job_description)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "processing_error", "detail": f"NLP pipeline failed: {str(e)}"},
        )
```

- [ ] **Step 4: Run all backend tests**

```bash
cd /Users/nateogunleye/Desktop/resume-matcher/backend
source venv/bin/activate
python -m pytest tests/ -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/routes.py backend/tests/test_api.py
git commit -m "feat: add /api/analyze endpoint with full NLP pipeline integration"
```

---

## Task 10: Frontend — Layout, Globals, Shared UI Components

**Files:**
- Rewrite: `src/app/layout.tsx`, `src/app/globals.css`
- Create: `src/components/ui/score-gauge.tsx`, `src/components/ui/accordion.tsx`, `src/components/ui/badge.tsx`, `src/components/ui/progress-bar.tsx`

- [ ] **Step 1: Rewrite `src/app/globals.css`**

Fresh Tailwind setup with CSS variables for the clean academic design. Remove all old variables (dg-*, old matcher vars). Define the new color system.

```css
@import "tailwindcss";

:root {
  --bg-page: #fafaf9;
  --bg-card: #ffffff;
  --bg-hover: #f5f5f4;
  --border: #e7e5e4;
  --border-strong: #d6d3d1;
  --text-primary: #0f172a;
  --text-secondary: #475569;
  --text-muted: #94a3b8;
  --accent: #2563eb;
  --accent-light: #dbeafe;
  --success: #22c55e;
  --success-bg: #dcfce7;
  --success-text: #166534;
  --warning: #f59e0b;
  --warning-bg: #fef3c7;
  --warning-text: #92400e;
  --danger: #ef4444;
  --danger-bg: #fee2e2;
  --danger-text: #991b1b;
  --gauge-track: #e2e8f0;
  --radius: 10px;
}

@layer base {
  html, body {
    height: 100%;
    background: var(--bg-page);
    color: var(--text-primary);
  }
}
```

- [ ] **Step 2: Rewrite `src/app/layout.tsx`**

```tsx
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "ResumeMatch — NLP-Powered Resume Analysis",
  description: "Analyze how well your resume matches a job description using NLP",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className={`${inter.className} h-full antialiased`}>{children}</body>
    </html>
  );
}
```

- [ ] **Step 3: Create `src/components/ui/score-gauge.tsx`**

Semi-circular SVG arc gauge component. Takes a `score` (0-100) and renders an animated arc.

```tsx
"use client";

interface ScoreGaugeProps {
  score: number;
  size?: number;
}

export function ScoreGauge({ score, size = 200 }: ScoreGaugeProps) {
  const strokeWidth = 12;
  const radius = (size - strokeWidth) / 2;
  const cx = size / 2;
  const cy = size / 2;

  // Semi-circle: arc from 180° to 0° (left to right, bottom half)
  const startAngle = 180;
  const endAngle = 0;
  const totalArc = 180;
  const scoreAngle = startAngle - (score / 100) * totalArc;

  const toRad = (deg: number) => (deg * Math.PI) / 180;

  const arcPath = (startDeg: number, endDeg: number) => {
    const startX = cx + radius * Math.cos(toRad(startDeg));
    const startY = cy - radius * Math.sin(toRad(startDeg));
    const endX = cx + radius * Math.cos(toRad(endDeg));
    const endY = cy - radius * Math.sin(toRad(endDeg));
    const largeArc = Math.abs(endDeg - startDeg) > 180 ? 1 : 0;
    return `M ${startX} ${startY} A ${radius} ${radius} 0 ${largeArc} 1 ${endX} ${endY}`;
  };

  // Color based on score
  const scoreColor = score >= 70 ? "var(--success)" : score >= 40 ? "var(--warning)" : "var(--danger)";

  return (
    <div className="relative flex flex-col items-center" style={{ width: size, height: size * 0.6 }}>
      <svg width={size} height={size * 0.6} viewBox={`0 0 ${size} ${size * 0.6 + 10}`}>
        {/* Track */}
        <path
          d={arcPath(startAngle, endAngle)}
          fill="none"
          stroke="var(--gauge-track)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />
        {/* Score arc */}
        {score > 0 && (
          <path
            d={arcPath(startAngle, scoreAngle)}
            fill="none"
            stroke={scoreColor}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
        )}
      </svg>
      <div className="absolute bottom-0 flex flex-col items-center">
        <span className="text-5xl font-extrabold tracking-tight" style={{ color: "var(--text-primary)" }}>
          {score}
        </span>
        <span className="text-sm" style={{ color: "var(--text-muted)" }}>match score</span>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Create `src/components/ui/accordion.tsx`**

```tsx
"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";

interface AccordionProps {
  title: string;
  score?: number;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

export function Accordion({ title, score, children, defaultOpen = false }: AccordionProps) {
  const [open, setOpen] = useState(defaultOpen);

  const scoreColor =
    score === undefined
      ? "var(--text-muted)"
      : score >= 70
        ? "var(--success-text)"
        : score >= 40
          ? "var(--warning-text)"
          : "var(--danger-text)";

  const scoreBg =
    score === undefined
      ? "transparent"
      : score >= 70
        ? "var(--success-bg)"
        : score >= 40
          ? "var(--warning-bg)"
          : "var(--danger-bg)";

  return (
    <div className="border rounded-[var(--radius)] overflow-hidden" style={{ borderColor: "var(--border)" }}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-4 transition-colors hover:bg-[var(--bg-hover)]"
      >
        <div className="flex items-center gap-3">
          {score !== undefined && (
            <span
              className="inline-flex items-center justify-center w-10 h-7 text-xs font-bold rounded-md"
              style={{ background: scoreBg, color: scoreColor }}
            >
              {score}
            </span>
          )}
          <span className="font-semibold text-sm" style={{ color: "var(--text-primary)" }}>{title}</span>
        </div>
        <ChevronDown
          size={18}
          className="transition-transform"
          style={{
            color: "var(--text-muted)",
            transform: open ? "rotate(180deg)" : "rotate(0deg)",
          }}
        />
      </button>
      {open && (
        <div className="px-5 pb-4 pt-0 border-t" style={{ borderColor: "var(--border)" }}>
          {children}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Create `src/components/ui/badge.tsx`**

```tsx
interface BadgeProps {
  variant: "matched" | "partial" | "missing";
  children: React.ReactNode;
}

export function Badge({ variant, children }: BadgeProps) {
  const styles = {
    matched: { background: "var(--success-bg)", color: "var(--success-text)" },
    partial: { background: "var(--warning-bg)", color: "var(--warning-text)" },
    missing: { background: "var(--danger-bg)", color: "var(--danger-text)" },
  };

  return (
    <span
      className="inline-block px-3 py-1 text-xs font-medium rounded-md"
      style={styles[variant]}
    >
      {children}
    </span>
  );
}
```

- [ ] **Step 6: Create `src/components/ui/progress-bar.tsx`**

```tsx
interface ProgressBarProps {
  label: string;
  value: number;
}

export function ProgressBar({ label, value }: ProgressBarProps) {
  const barColor = value >= 70 ? "var(--success)" : value >= 40 ? "var(--accent)" : "var(--danger)";

  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span style={{ color: "var(--text-secondary)" }}>{label}</span>
        <span className="font-semibold" style={{ color: "var(--text-primary)" }}>{value}%</span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "var(--gauge-track)" }}>
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${value}%`, background: barColor }}
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 7: Create component directories**

```bash
mkdir -p src/components/ui src/components/results src/components/input
```

- [ ] **Step 8: Commit**

```bash
git add src/app/layout.tsx src/app/globals.css src/components/ui/
git commit -m "feat: add new layout, design system, and shared UI components"
```

---

## Task 11: Frontend — Input Page

**Files:**
- Rewrite: `src/app/page.tsx`

- [ ] **Step 1: Implement input page**

```tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowRight, FileSearch, Loader2 } from "lucide-react";

export default function InputPage() {
  const router = useRouter();
  const [resumeText, setResumeText] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = resumeText.length >= 50 && jobDescription.length >= 50 && !loading;

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          resume_text: resumeText,
          job_description: jobDescription,
        }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Analysis failed");
      }

      const data = await response.json();
      // Store results and navigate
      sessionStorage.setItem("matchResults", JSON.stringify(data));
      router.push("/results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
      setLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <header
        className="flex items-center gap-3 px-8 py-4 border-b bg-[var(--bg-card)]"
        style={{ borderColor: "var(--border)" }}
      >
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ background: "var(--text-primary)" }}
        >
          <FileSearch size={16} className="text-white" />
        </div>
        <span className="text-lg font-bold tracking-tight" style={{ color: "var(--text-primary)" }}>
          ResumeMatch
        </span>
        <span className="text-xs ml-1" style={{ color: "var(--text-muted)" }}>
          NLP-Powered Analysis
        </span>
      </header>

      {/* Main content */}
      <main className="flex-1 flex flex-col items-center justify-center px-8 py-12">
        <div className="w-full max-w-4xl">
          {/* Title */}
          <div className="text-center mb-10">
            <h1 className="text-3xl font-extrabold tracking-tight" style={{ color: "var(--text-primary)" }}>
              Analyze Your Resume Match
            </h1>
            <p className="mt-2 text-sm" style={{ color: "var(--text-secondary)" }}>
              Paste your resume and a job description to see how well they align
            </p>
          </div>

          {/* Input cards */}
          <div className="grid grid-cols-2 gap-6">
            {/* Resume */}
            <div
              className="rounded-[var(--radius)] border overflow-hidden"
              style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
            >
              <div
                className="px-5 py-3 border-b"
                style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
              >
                <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                  Your Resume
                </h2>
              </div>
              <div className="p-5">
                <textarea
                  value={resumeText}
                  onChange={(e) => setResumeText(e.target.value)}
                  className="w-full rounded-lg border p-3 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-[var(--accent)] transition-shadow"
                  style={{
                    borderColor: "var(--border)",
                    color: "var(--text-primary)",
                    background: "var(--bg-page)",
                  }}
                  rows={14}
                  placeholder="Paste your resume text here..."
                />
                <p className="mt-2 text-xs" style={{ color: "var(--text-muted)" }}>
                  {resumeText.length} characters {resumeText.length < 50 && resumeText.length > 0 && "· minimum 50"}
                </p>
              </div>
            </div>

            {/* Job Description */}
            <div
              className="rounded-[var(--radius)] border overflow-hidden"
              style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
            >
              <div
                className="px-5 py-3 border-b"
                style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
              >
                <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                  Job Description
                </h2>
              </div>
              <div className="p-5">
                <textarea
                  value={jobDescription}
                  onChange={(e) => setJobDescription(e.target.value)}
                  className="w-full rounded-lg border p-3 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-[var(--accent)] transition-shadow"
                  style={{
                    borderColor: "var(--border)",
                    color: "var(--text-primary)",
                    background: "var(--bg-page)",
                  }}
                  rows={14}
                  placeholder="Paste the job description here..."
                />
                <p className="mt-2 text-xs" style={{ color: "var(--text-muted)" }}>
                  {jobDescription.length} characters {jobDescription.length < 50 && jobDescription.length > 0 && "· minimum 50"}
                </p>
              </div>
            </div>
          </div>

          {/* Error message */}
          {error && (
            <div
              className="mt-4 p-3 rounded-lg text-sm"
              style={{ background: "var(--danger-bg)", color: "var(--danger-text)" }}
            >
              {error}
            </div>
          )}

          {/* Analyze button */}
          <div className="mt-8 flex justify-center">
            <button
              onClick={handleAnalyze}
              disabled={!canSubmit}
              className="flex items-center gap-2 px-8 py-3 rounded-lg text-sm font-semibold text-white transition-opacity disabled:opacity-40 disabled:cursor-not-allowed"
              style={{ background: "var(--text-primary)" }}
            >
              {loading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  Analyze Match
                  <ArrowRight size={16} />
                </>
              )}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add src/app/page.tsx
git commit -m "feat: add input page with resume and JD text areas"
```

---

## Task 12: Frontend — Results Page Components

**Files:**
- Create: `src/components/results/hero-score.tsx`, `src/components/results/section-accordion.tsx`, `src/components/results/nlp-details-panel.tsx`

- [ ] **Step 1: Create `src/components/results/hero-score.tsx`**

```tsx
import { ScoreGauge } from "@/components/ui/score-gauge";
import { ProgressBar } from "@/components/ui/progress-bar";

interface HeroScoreProps {
  overallScore: number;
  verdict: string;
  summary: string;
  sections: Record<string, { score: number }>;
}

export function HeroScore({ overallScore, verdict, summary, sections }: HeroScoreProps) {
  const verdictColor =
    overallScore >= 70 ? "var(--success-text)" : overallScore >= 40 ? "var(--warning-text)" : "var(--danger-text)";
  const verdictBg =
    overallScore >= 70 ? "var(--success-bg)" : overallScore >= 40 ? "var(--warning-bg)" : "var(--danger-bg)";

  const sectionLabels: Record<string, string> = {
    skills: "Skills",
    experience: "Experience",
    education: "Education",
    preferred: "Preferred",
  };

  return (
    <div
      className="rounded-[var(--radius)] border p-8 text-center"
      style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
    >
      <ScoreGauge score={overallScore} size={220} />

      <div className="mt-4">
        <span
          className="inline-block px-4 py-1.5 rounded-full text-sm font-bold"
          style={{ background: verdictBg, color: verdictColor }}
        >
          {verdict}
        </span>
      </div>

      <p className="mt-4 text-sm max-w-lg mx-auto leading-relaxed" style={{ color: "var(--text-secondary)" }}>
        {summary}
      </p>

      {/* Sub-score bars */}
      <div className="mt-8 grid grid-cols-2 gap-x-8 gap-y-3 max-w-md mx-auto">
        {Object.entries(sections).map(([key, section]) => (
          <ProgressBar key={key} label={sectionLabels[key] || key} value={section.score} />
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create `src/components/results/section-accordion.tsx`**

```tsx
import { Accordion } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";

interface SectionAccordionProps {
  title: string;
  score: number;
  matched: string[];
  partial: string[];
  missing: string[];
}

export function SectionAccordion({ title, score, matched, partial, missing }: SectionAccordionProps) {
  const total = matched.length + partial.length + missing.length;

  return (
    <Accordion title={title} score={score}>
      <div className="pt-3 space-y-4">
        {/* Stats line */}
        <p className="text-xs" style={{ color: "var(--text-muted)" }}>
          {matched.length} matched · {partial.length} partial · {missing.length} missing · {total} total
        </p>

        {/* Matched */}
        {matched.length > 0 && (
          <div>
            <p className="text-xs font-semibold mb-2 uppercase tracking-wide" style={{ color: "var(--success-text)" }}>
              Matched
            </p>
            <div className="flex flex-wrap gap-2">
              {matched.map((item) => (
                <Badge key={item} variant="matched">{item}</Badge>
              ))}
            </div>
          </div>
        )}

        {/* Partial */}
        {partial.length > 0 && (
          <div>
            <p className="text-xs font-semibold mb-2 uppercase tracking-wide" style={{ color: "var(--warning-text)" }}>
              Partial Match
            </p>
            <div className="flex flex-wrap gap-2">
              {partial.map((item) => (
                <Badge key={item} variant="partial">{item}</Badge>
              ))}
            </div>
          </div>
        )}

        {/* Missing */}
        {missing.length > 0 && (
          <div>
            <p className="text-xs font-semibold mb-2 uppercase tracking-wide" style={{ color: "var(--danger-text)" }}>
              Missing
            </p>
            <div className="flex flex-wrap gap-2">
              {missing.map((item) => (
                <Badge key={item} variant="missing">{item}</Badge>
              ))}
            </div>
          </div>
        )}
      </div>
    </Accordion>
  );
}
```

- [ ] **Step 3: Create `src/components/results/nlp-details-panel.tsx`**

```tsx
import { Accordion } from "@/components/ui/accordion";

interface NlpDetailsPanelProps {
  nlpDetails: {
    jd_sections_parsed: Record<string, string[]>;
    resume_sections_parsed: Record<string, string>;
    resume_entities: Record<string, string[]>;
    tfidf_top_keywords: Record<string, { keyword: string; weight: number }[]>;
    similarity_scores: { tfidf_cosine: number; semantic: number };
  };
}

export function NlpDetailsPanel({ nlpDetails }: NlpDetailsPanelProps) {
  const {
    jd_sections_parsed,
    resume_sections_parsed,
    resume_entities,
    tfidf_top_keywords,
    similarity_scores,
  } = nlpDetails;

  return (
    <div
      className="rounded-[var(--radius)] border overflow-hidden"
      style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
    >
      <div
        className="px-5 py-3 border-b"
        style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
      >
        <h3 className="text-sm font-bold" style={{ color: "var(--text-primary)" }}>
          NLP Pipeline Details
        </h3>
        <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
          Technical breakdown of how the analysis was computed
        </p>
      </div>

      <div className="p-4 space-y-3">
        {/* Similarity Scores */}
        <Accordion title="Similarity Scores" defaultOpen>
          <div className="pt-3 grid grid-cols-2 gap-4">
            <div className="p-3 rounded-lg" style={{ background: "var(--bg-page)" }}>
              <p className="text-xs font-semibold" style={{ color: "var(--text-muted)" }}>TF-IDF Cosine</p>
              <p className="text-2xl font-bold mt-1" style={{ color: "var(--text-primary)" }}>
                {(similarity_scores.tfidf_cosine * 100).toFixed(1)}%
              </p>
            </div>
            <div className="p-3 rounded-lg" style={{ background: "var(--bg-page)" }}>
              <p className="text-xs font-semibold" style={{ color: "var(--text-muted)" }}>Semantic</p>
              <p className="text-2xl font-bold mt-1" style={{ color: "var(--text-primary)" }}>
                {(similarity_scores.semantic * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </Accordion>

        {/* TF-IDF Keywords */}
        <Accordion title="TF-IDF Top Keywords">
          <div className="pt-3 grid grid-cols-2 gap-6">
            {Object.entries(tfidf_top_keywords).map(([source, keywords]) => (
              <div key={source}>
                <p className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "var(--text-muted)" }}>
                  {source === "job_description" ? "Job Description" : "Resume"}
                </p>
                <div className="space-y-1">
                  {keywords.slice(0, 10).map((kw) => (
                    <div key={kw.keyword} className="flex items-center gap-2">
                      <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: "var(--gauge-track)" }}>
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${kw.weight * 100}%`, background: "var(--accent)" }}
                        />
                      </div>
                      <span className="text-xs w-24 truncate" style={{ color: "var(--text-secondary)" }}>
                        {kw.keyword}
                      </span>
                      <span className="text-xs font-mono w-10 text-right" style={{ color: "var(--text-muted)" }}>
                        {kw.weight.toFixed(2)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Accordion>

        {/* Extracted Entities */}
        <Accordion title="Extracted Resume Entities">
          <div className="pt-3 space-y-3">
            {Object.entries(resume_entities).map(([type, entities]) => (
              <div key={type}>
                <p className="text-xs font-semibold uppercase tracking-wide mb-1" style={{ color: "var(--text-muted)" }}>
                  {type}
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {entities.map((entity) => (
                    <span
                      key={entity}
                      className="px-2 py-0.5 text-xs rounded"
                      style={{ background: "var(--accent-light)", color: "var(--accent)" }}
                    >
                      {entity}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Accordion>

        {/* JD Sections Parsed */}
        <Accordion title="Job Description Sections Parsed">
          <div className="pt-3 space-y-3">
            {Object.entries(jd_sections_parsed).map(([section, items]) => (
              <div key={section}>
                <p className="text-xs font-semibold uppercase tracking-wide mb-1" style={{ color: "var(--text-muted)" }}>
                  {section}
                </p>
                {items.length > 0 ? (
                  <ul className="text-xs space-y-1" style={{ color: "var(--text-secondary)" }}>
                    {items.map((item, i) => (
                      <li key={i} className="pl-3 border-l-2" style={{ borderColor: "var(--border)" }}>
                        {item}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-xs italic" style={{ color: "var(--text-muted)" }}>No items extracted</p>
                )}
              </div>
            ))}
          </div>
        </Accordion>

        {/* Resume Sections Parsed */}
        <Accordion title="Resume Sections Parsed">
          <div className="pt-3 space-y-3">
            {Object.entries(resume_sections_parsed).map(([section, text]) => (
              <div key={section}>
                <p className="text-xs font-semibold uppercase tracking-wide mb-1" style={{ color: "var(--text-muted)" }}>
                  {section}
                </p>
                <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                  {text || <span className="italic" style={{ color: "var(--text-muted)" }}>Not detected</span>}
                </p>
              </div>
            ))}
          </div>
        </Accordion>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Commit**

```bash
git add src/components/results/
git commit -m "feat: add results page components (hero score, section accordion, NLP details)"
```

---

## Task 13: Frontend — Results Page

**Files:**
- Create: `src/app/results/page.tsx`

- [ ] **Step 1: Implement results page**

```tsx
"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { FileSearch, ArrowLeft, FlaskConical } from "lucide-react";
import { HeroScore } from "@/components/results/hero-score";
import { SectionAccordion } from "@/components/results/section-accordion";
import { NlpDetailsPanel } from "@/components/results/nlp-details-panel";

interface MatchResults {
  overall_score: number;
  verdict: string;
  summary: string;
  sections: Record<string, {
    score: number;
    matched: string[];
    partial: string[];
    missing: string[];
  }>;
  nlp_details: {
    jd_sections_parsed: Record<string, string[]>;
    resume_sections_parsed: Record<string, string>;
    resume_entities: Record<string, string[]>;
    tfidf_top_keywords: Record<string, { keyword: string; weight: number }[]>;
    similarity_scores: { tfidf_cosine: number; semantic: number };
  };
}

const sectionLabels: Record<string, string> = {
  skills: "Skills Match",
  experience: "Experience Match",
  education: "Education Match",
  preferred: "Preferred Qualifications",
};

export default function ResultsPage() {
  const router = useRouter();
  const [results, setResults] = useState<MatchResults | null>(null);
  const [showNlpDetails, setShowNlpDetails] = useState(false);

  useEffect(() => {
    const stored = sessionStorage.getItem("matchResults");
    if (!stored) {
      router.push("/");
      return;
    }
    setResults(JSON.parse(stored));
  }, [router]);

  if (!results) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="animate-pulse text-sm" style={{ color: "var(--text-muted)" }}>Loading results...</div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <header
        className="flex items-center gap-3 px-8 py-4 border-b bg-[var(--bg-card)]"
        style={{ borderColor: "var(--border)" }}
      >
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ background: "var(--text-primary)" }}
        >
          <FileSearch size={16} className="text-white" />
        </div>
        <span className="text-lg font-bold tracking-tight" style={{ color: "var(--text-primary)" }}>
          ResumeMatch
        </span>
        <div className="ml-auto flex items-center gap-3">
          <button
            onClick={() => setShowNlpDetails(!showNlpDetails)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition-colors"
            style={{
              borderColor: showNlpDetails ? "var(--accent)" : "var(--border)",
              color: showNlpDetails ? "var(--accent)" : "var(--text-secondary)",
              background: showNlpDetails ? "var(--accent-light)" : "transparent",
            }}
          >
            <FlaskConical size={14} />
            NLP Details
          </button>
          <button
            onClick={() => router.push("/")}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-semibold text-white"
            style={{ background: "var(--text-primary)" }}
          >
            <ArrowLeft size={14} />
            New Analysis
          </button>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <div className="max-w-3xl mx-auto px-8 py-10 space-y-6">
          {/* Hero score */}
          <HeroScore
            overallScore={results.overall_score}
            verdict={results.verdict}
            summary={results.summary}
            sections={results.sections}
          />

          {/* Section accordions */}
          <div className="space-y-3">
            {Object.entries(results.sections).map(([key, section]) => (
              <SectionAccordion
                key={key}
                title={sectionLabels[key] || key}
                score={section.score}
                matched={section.matched}
                partial={section.partial}
                missing={section.missing}
              />
            ))}
          </div>

          {/* NLP Details panel */}
          {showNlpDetails && (
            <NlpDetailsPanel nlpDetails={results.nlp_details} />
          )}
        </div>
      </main>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add src/app/results/
git commit -m "feat: add results page with score gauge, accordions, and NLP details"
```

---

## Task 14: End-to-End Verification

- [ ] **Step 1: Add `backend/venv/` to `.gitignore`**

Append to `.gitignore`:
```
backend/venv/
__pycache__/
*.pyc
.next/
node_modules/
```

- [ ] **Step 2: Run all backend tests**

```bash
cd /Users/nateogunleye/Desktop/resume-matcher/backend
source venv/bin/activate
python -m pytest tests/ -v
```

Expected: All PASS

- [ ] **Step 3: Start backend and test API manually**

```bash
cd /Users/nateogunleye/Desktop/resume-matcher/backend
source venv/bin/activate
python run.py &
sleep 15
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"resume_text":"Experienced software engineer with 6 years of React and TypeScript development. Led frontend teams at Google. Built REST APIs. Bachelor of Science in Computer Science from MIT. Skills include React, TypeScript, Python, Docker, Git, Agile methodology.","job_description":"Senior Frontend Engineer. Requirements: 5+ years React and TypeScript experience. REST API experience. Git proficiency. BS in Computer Science. Responsibilities: Lead frontend architecture. Mentor developers. Preferred: AWS experience. Masters degree. GraphQL."}'
kill %1
```

Expected: JSON response with `overall_score`, `verdict`, `sections`, `nlp_details`

- [ ] **Step 4: Start frontend and verify it builds**

```bash
cd /Users/nateogunleye/Desktop/resume-matcher
npm run build
```

Expected: Build succeeds

- [ ] **Step 5: Final commit and push**

```bash
git add -A
git commit -m "chore: add gitignore entries, verify end-to-end pipeline"
git push origin main
```
