# NLP Resume Matcher — Design Spec

## Overview

A resume-to-job-description matching tool built for an NLP course. The app analyzes how well a resume matches a job description using 10+ NLP techniques, producing a weighted match score with per-section breakdowns. The UI is clean, academic, and data-focused with a two-page flow: input page → results page.

**This is a fresh build.** The existing codebase (single-page state machine, rewrite tab, data-grid page, matcher/rewrite components) will be deleted and replaced entirely.

## Architecture

**FastAPI (Python) backend + Next.js (React) frontend.**

```
resume-matcher/
├── backend/                      # Python — all NLP lives here (new directory)
│   ├── venv/                     # Virtual environment (isolated)
│   ├── app/
│   │   ├── main.py               # FastAPI app, CORS (allow localhost:3000)
│   │   ├── api/
│   │   │   └── routes.py         # POST /api/analyze
│   │   ├── nlp/
│   │   │   ├── preprocessor.py       # Tokenization, stopwords, lemmatization
│   │   │   ├── section_parser.py     # JD + resume section extraction
│   │   │   ├── keyword_extractor.py  # TF-IDF keyword extraction
│   │   │   ├── entity_extractor.py   # spaCy NER, POS tagging, noun phrases
│   │   │   ├── similarity.py         # Cosine similarity + semantic similarity
│   │   │   └── matcher.py            # Orchestrator — runs pipeline, computes scores
│   │   └── models/
│   │       └── schemas.py            # Pydantic request/response models
│   ├── requirements.txt
│   └── run.py                    # Entry point: python run.py
│
├── src/                          # Next.js frontend (rewritten from scratch)
│   ├── app/
│   │   ├── page.tsx              # Input page
│   │   ├── results/page.tsx      # Results page
│   │   └── layout.tsx
│   └── components/
│       ├── input/                # Text areas for resume + JD
│       ├── results/              # Score gauge, accordion sections, NLP details
│       └── ui/                   # Shared components (button, badge, progress bar)
│
└── package.json
```

- Backend runs on `:8000`, frontend on `:3000`
- CORS configured to allow `http://localhost:3000`
- All Python dependencies isolated in `backend/venv/`
- Single API endpoint: `POST /api/analyze`
- sentence-transformers model: `all-MiniLM-L6-v2` (~80MB, good balance of speed and quality), loaded once at app startup and reused across requests

## NLP Pipeline

### Step 1: Preprocessing (`preprocessor.py`)
- **Tokenization** — NLTK `word_tokenize`, split text into tokens
- **Stopword removal** — NLTK stopwords corpus
- **Lemmatization** — NLTK `WordNetLemmatizer`, normalize to base forms
- **Sentence segmentation** — spaCy, split into sentences for semantic comparison

### Step 2: Section Parsing (`section_parser.py`)
**Job Description** — regex + keyword heuristics to split into sections:
- Requirements / Qualifications
- Responsibilities
- Preferred / Nice-to-have
- Other (company info, benefits, etc.)

**Resume** — parsed into: Skills, Experience, Education, Summary.

Falls back gracefully for unstructured JDs — treats whole text as single block and attempts entity-level extraction across the full text.

### Step 3: Entity Extraction (`entity_extractor.py`)
- **spaCy NER** — extract named entities (ORG, PERSON, GPE, etc.)
- **Custom skill extraction** — POS tagging + noun phrase chunking for technical skills, tools, frameworks
- **Education extraction** — pattern matching for degrees (B.S., M.S., PhD) + NER for institutions

### Step 4: Keyword Extraction (`keyword_extractor.py`)
- **TF-IDF vectorization** — scikit-learn `TfidfVectorizer`
- Produces weighted keyword lists for each section

### Step 5: Matching & Similarity (`similarity.py` + `matcher.py`)

Each match category is computed by combining entity overlap and TF-IDF cosine similarity for the relevant content:

- **Skills match (40%)** — Entities extracted from JD requirements + responsibilities matched against resume skills. Uses entity overlap + TF-IDF cosine similarity on the skills-related text.
- **Experience match (25%)** — JD responsibilities + experience requirements matched against resume experience section. Entity overlap + cosine similarity.
- **Education match (15%)** — JD education requirements matched against resume education section. Pattern + entity matching.
- **Preferred match (10%)** — JD preferred/nice-to-have items matched against full resume. Softer matching — partial credit for related terms.
- **Semantic similarity (10%)** — Full-document sentence-transformers similarity between resume and JD overall. Captures meaning-level alignment beyond keyword matching.

**Weighted final score:**
```
final = (skills_match    * 0.40)
      + (experience_match * 0.25)
      + (education_match  * 0.15)
      + (preferred_match  * 0.10)
      + (semantic_sim     * 0.10)
```

**How "partial" matches are determined:**
- A skill/requirement is **matched** if exact or lemmatized match is found, OR semantic similarity > 0.8
- A skill/requirement is **partial** if semantic similarity is between 0.5 and 0.8 (e.g., "Docker" in resume, "containerization" in JD)
- A skill/requirement is **missing** if semantic similarity < 0.5 and no entity overlap found

**NLP techniques demonstrated (10+):** tokenization, stopword removal, lemmatization, sentence segmentation, named entity recognition, POS tagging, noun phrase chunking, TF-IDF vectorization, cosine similarity, semantic embeddings.

## API Contract

### `POST /api/analyze`

**Request:**
```json
{
  "resume_text": "string (required, min 50 chars, max 15000 chars)",
  "job_description": "string (required, min 50 chars, max 10000 chars)"
}
```

**Response (200 OK):**
```json
{
  "overall_score": 82,
  "verdict": "Strong Match",
  "summary": "Your resume aligns well with this role...",
  "sections": {
    "skills": {
      "score": 75,
      "matched": ["React", "TypeScript"],
      "partial": ["Docker"],
      "missing": ["AWS", "GraphQL"]
    },
    "experience": {
      "score": 88,
      "matched": ["5+ Years Frontend", "Team Lead"],
      "partial": [],
      "missing": ["Enterprise SaaS"]
    },
    "education": {
      "score": 90,
      "matched": ["B.S. Computer Science"],
      "partial": ["M.S. Preferred"],
      "missing": []
    },
    "preferred": {
      "score": 60,
      "matched": ["Agile methodology"],
      "partial": ["Cloud experience"],
      "missing": ["GraphQL expertise"]
    }
  },
  "nlp_details": {
    "jd_sections_parsed": {
      "requirements": ["5+ years frontend...", "React, TypeScript..."],
      "responsibilities": ["Lead frontend architecture..."],
      "preferred": ["AWS experience...", "M.S. degree"],
      "other": ["Competitive salary..."]
    },
    "resume_sections_parsed": {
      "skills": "React, TypeScript, Python...",
      "experience": "Software Engineer at Google...",
      "education": "B.S. Computer Science, MIT"
    },
    "resume_entities": {
      "skills": ["React", "TypeScript", "Python"],
      "organizations": ["Google", "MIT"],
      "education": ["B.S. Computer Science"]
    },
    "tfidf_top_keywords": {
      "job_description": [{"keyword": "react", "weight": 0.42}, {"keyword": "frontend", "weight": 0.38}],
      "resume": [{"keyword": "react", "weight": 0.45}, {"keyword": "python", "weight": 0.40}]
    },
    "similarity_scores": {
      "tfidf_cosine": 0.72,
      "semantic": 0.79
    }
  }
}
```

**Verdict mapping:**
| Score Range | Verdict |
|-------------|---------|
| 0–39 | Weak Match |
| 40–69 | Moderate Match |
| 70–100 | Strong Match |

**Error Response (422):**
```json
{
  "error": "validation_error",
  "detail": "Resume text must be at least 50 characters"
}
```

**Error Response (500):**
```json
{
  "error": "processing_error",
  "detail": "NLP pipeline failed: [specific error]"
}
```

## UI Design

### Design Direction
Clean, academic, professional, data-focused. Inspired by the Jobscan-style layout with semi-circular gauge and accordion sections.

### Page 1: Input
- Header with app name "ResumeMatch"
- Two input cards side by side: Resume (paste textarea) and Job Description (paste textarea)
- Centered "Analyze Match" button
- Stacked card aesthetic with rounded corners, subtle borders
- Text-only input — no file upload (keeps scope focused on NLP, not file parsing)

### Page 2: Results
- **Hero score section:** Semi-circular arc gauge with large score number, verdict badge, short summary text, mini sub-score bars for each category
- **Section accordion rows:** Skills, Experience, Education, Preferred — each row shows a color-coded score badge and expands to show matched (green), partial (yellow), missing (red) items
- **"View NLP Pipeline Details" button:** Expands a panel at the bottom showing extracted entities, TF-IDF keyword weights, similarity scores, parsed sections — the technical deep-dive
- **"New Analysis" button** to return to input page

### Color Coding
- Green (`#dcfce7` / `#166534`) — matched items
- Yellow (`#fef3c7` / `#92400e`) — partial matches
- Red (`#fee2e2` / `#991b1b`) — missing items
- Score badge border color scales with score value

### Loading State
Skeleton loading on the results page while NLP pipeline processes (animated pulse placeholders for gauge, accordion rows, etc.)

## Decisions & Constraints
- **Fresh build** — existing codebase (rewrite tab, matcher components, data-grid page) will be deleted and replaced
- **No rewrite feature** — skipped for now, may add later with local LLM
- **No file upload** — text paste only, avoids PDF parsing complexity; keeps focus on NLP
- **Python venv isolation** — all deps in `backend/venv/`, nothing system-wide
- **Graceful JD parsing** — handles both structured (with clear sections) and unstructured (prose) job descriptions
- **sentence-transformers** — uses `all-MiniLM-L6-v2` (~80MB), loaded once at startup; downloads on first run
- **Model preloading** — spaCy model and sentence-transformers model loaded at FastAPI startup, reused across requests
