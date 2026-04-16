"""Word2Vec-powered skill expansion for resume matching.

When a JD requirement item is "missing" from a resume, this module uses the
trained Word2Vec model to check whether a semantically close synonym *is*
present in the resume — upgrading the item from "missing" to "partial".

It also generates human-readable rewrite suggestions for each missing item,
explaining what the user could add to their resume to address the gap.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app import models_state

if TYPE_CHECKING:
    from gensim.models import Word2Vec

# Similarity threshold: W2V cosine score above which we treat a synonym as
# a "partial" match (i.e. related concept found in resume).
W2V_PARTIAL_THRESHOLD = 0.60

# How many nearest neighbours to consider when looking for synonyms
TOPN = 8

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "have", "he", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "they", "this", "to", "was", "were", "will", "with",
    "you", "your", "experience", "years", "knowledge", "ability", "strong",
    "required", "preferred", "working", "related",
}


def _get_model() -> "Word2Vec | None":
    return models_state.word2vec_model


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./\-\s]", " ", text)
    return [t for t in text.split() if len(t) >= 2 and t not in _STOPWORDS]


def _resume_token_set(resume_text: str) -> set[str]:
    return set(_tokenize(resume_text))


def _synonyms_for_token(token: str, model: "Word2Vec", topn: int = TOPN) -> list[tuple[str, float]]:
    """Return (word, similarity) pairs for a single token."""
    if token not in model.wv:
        return []
    return model.wv.most_similar(token, topn=topn)


def check_w2v_partial(item: str, resume_text: str) -> tuple[bool, list[str]]:
    """Check if any W2V synonym of `item` tokens appears in `resume_text`.

    Returns:
        (is_partial, synonym_hints)
        is_partial  — True if a synonym was found above threshold
        synonym_hints — list of matching synonym words found
    """
    model = _get_model()
    if model is None:
        return False, []

    item_tokens = _tokenize(item)
    resume_tokens = _resume_token_set(resume_text)
    found_synonyms: list[str] = []

    for token in item_tokens:
        if token in resume_tokens:
            # Token already directly present — not a W2V hit, caller handles
            continue
        for synonym, score in _synonyms_for_token(token, model):
            if score >= W2V_PARTIAL_THRESHOLD and synonym in resume_tokens:
                found_synonyms.append(synonym)

    return len(found_synonyms) > 0, found_synonyms


def generate_rewrite_suggestions(
    sections: dict[str, dict],
    resume_text: str,
) -> list[dict]:
    """Generate resume rewrite suggestions for all missing items.

    Uses W2V neighbours to detect *what the user already has* that's related,
    then crafts a targeted suggestion for each missing item.

    Returns a list of suggestion dicts:
        {
          "section": str,          e.g. "skills"
          "missing_item": str,     the original JD requirement
          "suggestion": str,       human-readable rewrite tip
          "related_in_resume": list[str],  W2V synonyms found in resume
          "w2v_expanded": bool,    True if W2V found a related concept
        }
    """
    model = _get_model()
    suggestions: list[dict] = []

    section_action_verbs = {
        "skills": "Add to your Skills section",
        "experience": "Add to your Experience bullets",
        "education": "Add to your Education section",
        "preferred": "Consider adding to strengthen your application",
    }

    for section_name, section_data in sections.items():
        action = section_action_verbs.get(section_name, "Add to your resume")
        for missing_item in section_data.get("missing", []):
            # Find W2V-related terms already in resume
            related: list[str] = []
            if model is not None:
                item_tokens = _tokenize(missing_item)
                resume_tokens = _resume_token_set(resume_text)
                for token in item_tokens:
                    for synonym, score in _synonyms_for_token(token, model):
                        if score >= W2V_PARTIAL_THRESHOLD and synonym in resume_tokens:
                            if synonym not in related:
                                related.append(synonym)

            w2v_expanded = len(related) > 0

            # Build suggestion text
            if w2v_expanded:
                related_str = ", ".join(f'"{r}"' for r in related[:3])
                suggestion = (
                    f'{action}: "{missing_item}". '
                    f"Your resume mentions {related_str} — explicitly calling out "
                    f'"{missing_item}" will improve your match score.'
                )
            else:
                suggestion = (
                    f'{action}: "{missing_item}". '
                    f"This requirement was not found in your resume. "
                    f"If you have this experience, add it directly."
                )

            suggestions.append({
                "section": section_name,
                "missing_item": missing_item,
                "suggestion": suggestion,
                "related_in_resume": related,
                "w2v_expanded": w2v_expanded,
            })

    return suggestions
