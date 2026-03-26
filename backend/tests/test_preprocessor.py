"""Tests for text preprocessing."""
from app.nlp.preprocessor import preprocess_text, tokenize, remove_stopwords, lemmatize, segment_sentences


def test_tokenize_splits_words():
    tokens = tokenize("Hello world, this is a test.")
    assert "Hello" in tokens
    assert "world" in tokens


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
    assert "run" in lemmatized
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
    assert "the" not in result["lemmatized"]
    assert "are" not in result["lemmatized"]
