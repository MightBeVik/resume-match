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
