"""Text preprocessing: tokenization, stopword removal, lemmatization, sentence segmentation."""
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from app import models_state


def tokenize(text: str) -> list[str]:
    """Split text into word tokens using NLTK word_tokenize."""
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove common English stopwords from token list."""
    stop_words = set(stopwords.words("english"))
    return [t for t in tokens if t.lower() not in stop_words]


def lemmatize(tokens: list[str]) -> list[str]:
    """Reduce words to base form using WordNet lemmatizer."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t.lower(), pos="v") for t in tokens]


def segment_sentences(text: str) -> list[str]:
    """Split text into sentences using spaCy."""
    nlp = models_state.nlp_model
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)

    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def preprocess_text(text: str) -> dict:
    """Run full preprocessing pipeline on input text.

    Returns dict with keys: tokens, filtered, lemmatized, sentences
    """
    tokens = tokenize(text)
    filtered = remove_stopwords(tokens)
    lemmatized = lemmatize(filtered)
    sentences = segment_sentences(text)

    return {
        "tokens": tokens,
        "filtered": filtered,
        "lemmatized": lemmatized,
        "sentences": sentences,
    }
