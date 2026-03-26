"""Tests for TF-IDF keyword extraction."""
from app.nlp.keyword_extractor import extract_keywords, get_tfidf_vectors


def test_extract_keywords_returns_weighted_list():
    text = "React TypeScript frontend development web applications JavaScript"
    keywords = extract_keywords(text, top_n=5)
    assert len(keywords) > 0
    assert len(keywords) <= 5
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
