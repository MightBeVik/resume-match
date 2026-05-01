"""Similarity scoring using TF-IDF cosine similarity and sentence-transformers."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app import models_state


def _get_tfidf_vectorizer(text_a: str, text_b: str) -> TfidfVectorizer:
    """Return the corpus-fitted vectorizer if available, else create a 2-doc one.

    The corpus-fitted vectorizer (loaded at startup from jobs.db) produces
    meaningful IDF weights because it has seen thousands of documents.
    Without it, IDF is computed on only 2 documents — every non-shared term
    gets the same maximum weight, which makes the cosine score noisy.
    """
    if models_state.tfidf_vectorizer is not None:
        return models_state.tfidf_vectorizer

    # Fallback: fit on the two documents themselves with improved parameters
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    vectorizer.fit([text_a, text_b])
    return vectorizer


def _get_sentence_model():
    """Get sentence-transformers model, preferring the preloaded one."""
    if models_state.sentence_model is not None:
        return models_state.sentence_model
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using TF-IDF vectors.

    Uses the corpus-fitted vectorizer when available so that IDF weights
    reflect term frequency across thousands of job descriptions rather than
    just the two documents being compared.
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0

    vectorizer = _get_tfidf_vectorizer(text_a, text_b)

    try:
        if models_state.tfidf_vectorizer is not None:
            # Transform only — vectorizer already fitted on corpus
            matrix = vectorizer.transform([text_a, text_b])
        else:
            # Fallback: fit_transform on the two docs
            matrix = vectorizer.fit_transform([text_a, text_b])
    except Exception:
        return 0.0

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
