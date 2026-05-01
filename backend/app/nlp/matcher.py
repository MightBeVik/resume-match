"""Matcher orchestrator — runs the full NLP pipeline and computes weighted scores."""
import hashlib
import re

from app.nlp.preprocessor import preprocess_text
from app.nlp.section_parser import parse_job_description, parse_resume
from app.nlp.entity_extractor import extract_entities, extract_skills
from app.nlp.keyword_extractor import extract_keywords
from app.nlp.similarity import tfidf_cosine_similarity, semantic_similarity, item_semantic_similarity
from app.nlp.word2vec_expander import check_w2v_partial, generate_rewrite_suggestions


# Score weights
WEIGHTS = {
    "skills": 0.40,
    "experience": 0.25,
    "education": 0.15,
    "preferred": 0.10,
    "semantic": 0.10,
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "build", "for", "in", "of", "on", "or",
    "the", "to", "with", "using", "strong", "understanding", "experience", "proficiency",
    "required", "preferred", "qualification", "qualifications", "years", "year", "plus",
    "role", "related", "field",
}

DEGREE_LEVELS = {
    "associate": {"associate"},
    "bachelor": {"bachelor", "bachelors", "b.s", "bs", "b.a", "ba"},
    "master": {"master", "masters", "m.s", "ms", "m.a", "ma", "mba"},
    "phd": {"phd", "doctorate"},
}

# ---------------------------------------------------------------------------
# Simple hash-based response cache
# Keyed on (sha256(resume_text), sha256(jd_text)) so identical pairs are
# served instantly — critical for bulk_match which calls analyze_match in an
# O(resumes × jobs) loop.
# ---------------------------------------------------------------------------
_MATCH_CACHE: dict[tuple[str, str], dict] = {}
_CACHE_MAX = 512  # evict oldest when full


def _cache_key(resume_text: str, jd_text: str) -> tuple[str, str]:
    return (
        hashlib.sha256(resume_text.encode()).hexdigest(),
        hashlib.sha256(jd_text.encode()).hexdigest(),
    )


# ---------------------------------------------------------------------------
# Years-of-experience helpers
# ---------------------------------------------------------------------------

# Matches "3+ years", "2-5 years", "at least 3 years", "minimum 5 years", etc.
_YEARS_REQ_PATTERN = re.compile(
    r"(?:at\s+least\s+|minimum\s+|(?:a\s+)?minimum\s+of\s+)?(\d+)(?:\s*[-–]\s*\d+)?\+?\s+years?",
    re.IGNORECASE,
)

# Matches "20XX – 20XX" or "20XX - present" style date ranges in resumes
_DATE_RANGE_PATTERN = re.compile(
    r"(20\d{2}|19\d{2})\s*[-–]\s*(20\d{2}|19\d{2}|present|current|now)",
    re.IGNORECASE,
)


def _extract_years_requirement(item: str) -> int | None:
    """Return the minimum years required from a JD item, or None if not stated."""
    match = _YEARS_REQ_PATTERN.search(item)
    if match:
        return int(match.group(1))
    return None


def _estimate_resume_experience_years(resume_text: str) -> float:
    """Estimate total years of professional experience from date ranges in the resume.

    Sums all non-overlapping year-span pairs found.  Returns 0 if no dates detected.
    """
    import datetime
    current_year = datetime.datetime.now().year
    total = 0.0
    for match in _DATE_RANGE_PATTERN.finditer(resume_text):
        start_year = int(match.group(1))
        end_raw = match.group(2).lower()
        end_year = current_year if end_raw in {"present", "current", "now"} else int(end_raw)
        span = max(0, end_year - start_year)
        if 0 < span <= 50:  # sanity check
            total += span
    # Cap at 40 to avoid education date ranges inflating the total unrealistically
    return min(total, 40.0)


def _get_verdict(score: int) -> str:
    """Map overall score to verdict string."""
    if score >= 70:
        return "Strong Match"
    elif score >= 40:
        return "Moderate Match"
    return "Weak Match"


def _generate_summary(score: int, sections: dict) -> str:
    """Generate a human-readable summary of the match."""
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
        summary += "This role may require significant additional experience or skills."

    return summary


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("'", "'")
    text = re.sub(r"[^a-z0-9+#./\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize_meaningful(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9+#/.]+", _normalize_text(text)))
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def _split_resume_segments(text: str) -> list[str]:
    parts = re.split(r"\n+|(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if len(part.strip()) > 10]


def _get_degree_level(text: str) -> str | None:
    normalized = _normalize_text(text)
    for level, markers in DEGREE_LEVELS.items():
        if any(marker in normalized for marker in markers):
            return level
    return None


def _get_field_tokens(text: str) -> set[str]:
    normalized = _normalize_text(text)
    if "computer science" in normalized:
        return {"computer", "science"}
    if "data science" in normalized:
        return {"data", "science"}
    match = re.search(r"\bin\s+([a-z\s&]+)", normalized)
    if not match:
        return set()
    return {token for token in re.findall(r"[a-z]+", match.group(1)) if token not in STOPWORDS}


def _match_education_item(item: str, resume_text: str, resume_education: list[str]) -> str:
    item_level = _get_degree_level(item)
    resume_levels = {_get_degree_level(entry) for entry in resume_education}
    resume_levels.discard(None)

    item_fields = _get_field_tokens(item)
    resume_field_text = " ".join(resume_education) or resume_text
    resume_fields = _get_field_tokens(resume_field_text)
    normalized_resume = _normalize_text(resume_text)

    level_matches = item_level is None or item_level in resume_levels
    field_matches = not item_fields or bool(item_fields & resume_fields) or all(field in normalized_resume for field in item_fields)

    if level_matches and field_matches:
        return "matched"
    if field_matches and item_level is None:
        return "partial"
    return "missing"


def _match_items(
    jd_items: list[str],
    resume_text: str,
    resume_skills: list[str],
    resume_education: list[str],
    resume_years: float = 0.0,
) -> dict:
    """Match JD requirement items against resume content.

    Returns: {"score": int, "matched": [...], "partial": [...], "missing": [...]}
    """
    if not jd_items:
        return {"score": 100, "matched": [], "partial": [], "missing": []}

    matched = []
    partial = []
    missing = []

    resume_lower = resume_text.lower()
    normalized_resume = _normalize_text(resume_text)
    resume_skills_lower = [s.lower() for s in resume_skills]
    resume_segments = _split_resume_segments(resume_text)

    for item in jd_items:
        item_clean = item.strip()
        if not item_clean:
            continue

        item_lower = item_clean.lower()
        normalized_item = _normalize_text(item_clean)

        # --- Years-of-experience check ---
        required_years = _extract_years_requirement(item_clean)
        if required_years is not None:
            if resume_years >= required_years:
                matched.append(item_clean)
            elif resume_years >= required_years * 0.6:
                # Candidate has a significant portion of required years
                partial.append(item_clean)
            else:
                # Fall through to regular matching — skill may still be found
                # even if years are unclear
                required_years = None  # don't skip, let text matching decide

        if required_years is not None:
            continue  # already classified by years logic

        if _is_education_item(item_clean):
            education_match = _match_education_item(item_clean, resume_text, resume_education)
            if education_match == "matched":
                matched.append(item_clean)
            elif education_match == "partial":
                partial.append(item_clean)
            else:
                missing.append(item_clean)
            continue

        item_skills = {skill.lower() for skill in extract_skills(item_clean)}
        if item_skills and item_skills.issubset(set(resume_skills_lower)):
            matched.append(item_clean)
            continue

        # Check 1: Direct text match (exact or substring)
        if normalized_item and normalized_item in normalized_resume:
            matched.append(item_clean)
            continue

        # Check 2: Word-level overlap check
        item_words = _tokenize_meaningful(item_clean)

        if item_words:
            overlap = sum(1 for w in item_words if w in normalized_resume)
            overlap_ratio = overlap / len(item_words)
            if overlap_ratio >= 0.6 or (len(item_words) <= 3 and overlap_ratio >= 0.5):
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
        best_segment = ""
        best_overlap = 0
        for segment in resume_segments:
            segment_overlap = len(item_words & _tokenize_meaningful(segment)) if item_words else 0
            if segment_overlap > best_overlap:
                best_segment = segment
                best_overlap = segment_overlap

        if best_segment and best_overlap > 0:
            sim = item_semantic_similarity(item_clean, best_segment)
        else:
            sim = 0.0

        if sim >= 0.88:
            matched.append(item_clean)
        elif sim >= 0.72:
            partial.append(item_clean)
        else:
            # Step 5 (W2V): check if a semantic synonym exists in the resume
            w2v_hit, _ = check_w2v_partial(item_clean, resume_text)
            if w2v_hit:
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


def _is_education_item(item: str) -> bool:
    """Check if a JD requirement item is education-related."""
    edu_keywords = ["degree", "bachelor", "master", "phd", "b.s.", "m.s.", "mba",
                    "education", "university", "college", "diploma", "certified"]
    return any(kw in item.lower() for kw in edu_keywords)


def _extract_education_requirements(text: str) -> list[str]:
    """Extract education requirements from full JD text as fallback."""
    edu_patterns = re.compile(
        r"(?:bachelor|master|phd|b\.s\.|m\.s\.|mba|degree)[\w\s,'.]*(?:required|preferred|or equivalent)?",
        re.IGNORECASE,
    )
    matches = edu_patterns.findall(text)
    return [m.strip() for m in matches if len(m.strip()) > 5]


def analyze_match(resume_text: str, job_description: str) -> dict:
    """Run the full NLP analysis pipeline.

    Results are cached by (resume_hash, jd_hash) so repeated calls for the
    same pair — e.g. during bulk_match — are served instantly.

    Steps:
    1. Check cache
    2. Preprocess both texts
    3. Parse into sections
    4. Extract entities (skills, certifications, education)
    5. Extract years of experience from resume
    6. Extract TF-IDF keywords
    7. Compute per-section matches
    8. Compute similarities
    9. Build weighted final score
    10. Cache and return

    Returns the full response dict matching the API schema.
    """
    # Step 1: Cache lookup
    key = _cache_key(resume_text, job_description)
    if key in _MATCH_CACHE:
        return _MATCH_CACHE[key]

    # Step 2: Preprocess
    preprocess_text(resume_text)
    preprocess_text(job_description)

    # Step 3: Parse sections
    jd_sections = parse_job_description(job_description)
    resume_sections = parse_resume(resume_text)

    # Step 4: Extract entities
    resume_entities = extract_entities(resume_text)
    resume_skills = resume_entities["skills"]
    resume_education = resume_entities["education"]

    full_resume = resume_text

    # Step 5: Estimate years of experience from date ranges in the resume
    resume_years = _estimate_resume_experience_years(resume_text)

    # Step 6: TF-IDF keywords
    jd_keywords = extract_keywords(job_description, top_n=15)
    resume_keywords = extract_keywords(resume_text, top_n=15)

    # Step 7: Per-section matching
    skill_items = [item for item in jd_sections["requirements"] if not _is_education_item(item)]

    skills_result = _match_items(
        skill_items,
        full_resume,
        resume_skills,
        resume_education,
        resume_years,
    )

    experience_result = _match_items(
        jd_sections["responsibilities"],
        resume_sections.get("experience", "") + " " + resume_sections.get("summary", ""),
        resume_skills,
        resume_education,
        resume_years,
    )

    # Education: filter education items from requirements, or extract from full JD
    edu_items = [item for item in jd_sections["requirements"] if _is_education_item(item)]
    if not edu_items:
        edu_items = _extract_education_requirements(job_description)
    education_result = _match_items(
        edu_items,
        resume_sections.get("education", "") + " " + full_resume,
        resume_skills,
        resume_education,
        resume_years,
    )

    preferred_result = _match_items(
        jd_sections["preferred"],
        full_resume,
        resume_skills,
        resume_education,
        resume_years,
    )

    # Step 8: Similarities
    tfidf_sim = tfidf_cosine_similarity(resume_text, job_description)
    sem_sim = semantic_similarity(resume_text, job_description)

    # Step 9: Weighted final score
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

    # Generate W2V-powered rewrite suggestions for all missing items
    rewrite_suggestions = generate_rewrite_suggestions(sections, resume_text)

    result = {
        "overall_score": overall_score,
        "verdict": verdict,
        "summary": summary,
        "sections": sections,
        "rewrite_suggestions": rewrite_suggestions,
        "nlp_details": {
            "jd_sections_parsed": {k: v for k, v in jd_sections.items()},
            "resume_sections_parsed": {k: v for k, v in resume_sections.items()},
            "resume_entities": resume_entities,
            "resume_years_estimated": round(resume_years, 1),
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

    # Step 10: Store in cache (evict oldest entry when at capacity)
    if len(_MATCH_CACHE) >= _CACHE_MAX:
        oldest_key = next(iter(_MATCH_CACHE))
        del _MATCH_CACHE[oldest_key]
    _MATCH_CACHE[key] = result
    return result
