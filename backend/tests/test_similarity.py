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
    assert score > 0.2


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
    assert score > 0.3


def test_item_semantic_similarity_identical():
    score = item_semantic_similarity("React", "React")
    assert score > 0.9
