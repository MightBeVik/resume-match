"""Matcher orchestrator — runs the full NLP pipeline and computes weighted scores."""
import re

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
        item_words = {w for w in item_words if len(w) > 2}

        if item_words:
            overlap = sum(1 for w in item_words if w in resume_lower)
            overlap_ratio = overlap / len(item_words)
            if overlap_ratio > 0.4:
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
    preprocess_text(resume_text)
    preprocess_text(job_description)

    # Step 2: Parse sections
    jd_sections = parse_job_description(job_description)
    resume_sections = parse_resume(resume_text)

    # Step 3: Extract entities
    resume_entities = extract_entities(resume_text)
    resume_skills = resume_entities["skills"]

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

    # Education: filter education items from requirements, or extract from full JD
    edu_items = [item for item in jd_sections["requirements"] if _is_education_item(item)]
    if not edu_items:
        edu_items = _extract_education_requirements(job_description)
    education_result = _match_items(
        edu_items,
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
