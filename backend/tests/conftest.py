"""Test configuration — ensure NLP models and data are available."""
import nltk
import spacy
import pytest
from app import models_state


@pytest.fixture(scope="session", autouse=True)
def setup_nlp_models():
    """Load NLP models once for the entire test session."""
    # NLTK data
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    # spaCy
    try:
        models_state.nlp_model = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        models_state.nlp_model = spacy.load("en_core_web_sm")

    # sentence-transformers
    from sentence_transformers import SentenceTransformer
    models_state.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
