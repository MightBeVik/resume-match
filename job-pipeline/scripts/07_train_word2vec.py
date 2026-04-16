"""Train a Word2Vec model on the jobs corpus from jobs.db.

Reads jd_text from the enriched_jobs table, tokenises each document,
trains a Skip-gram Word2Vec model with gensim, then saves the model to
  data/word2vec_jobs.model

The saved model is loaded by the backend at startup and used to expand
skill matching: when a JD requirement is "missing" from a resume, the
expander checks whether any semantically-close synonym *is* present.
"""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

from gensim.models import Word2Vec

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "jobs.db"
MODEL_PATH = ROOT / "data" / "word2vec_jobs.model"

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "have", "he", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "they", "this", "to", "was", "were", "will", "with",
    "you", "your",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords, keep tokens ≥ 2 chars."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./\-\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]


def load_sentences(db_path: Path) -> list[list[str]]:
    if not db_path.exists():
        raise FileNotFoundError(f"jobs.db not found at {db_path}. Run job-pipeline first.")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT jd_text FROM enriched_jobs WHERE jd_text IS NOT NULL AND TRIM(jd_text) != ''")
    rows = cur.fetchall()
    conn.close()

    print(f"[word2vec] Loaded {len(rows):,} job descriptions from corpus")

    sentences: list[list[str]] = []
    for (jd_text,) in rows:
        tokens = _tokenize(jd_text)
        if tokens:
            sentences.append(tokens)

    print(f"[word2vec] Tokenised into {len(sentences):,} sentences")
    return sentences


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(sentences: list[list[str]], model_path: Path) -> Word2Vec:
    print("[word2vec] Training Word2Vec (Skip-gram, vector_size=100, window=5, min_count=3)...")
    t0 = time.time()

    model = Word2Vec(
        sentences=sentences,
        vector_size=100,   # embedding dimensions
        window=5,          # context window
        min_count=3,       # ignore tokens seen < 3 times
        workers=4,         # parallel threads
        sg=1,              # 1 = Skip-gram (better for rare words / domain vocab)
        epochs=10,
        seed=42,
    )

    elapsed = time.time() - t0
    print(f"[word2vec] Training complete in {elapsed:.1f}s")
    print(f"[word2vec] Vocabulary size: {len(model.wv):,} unique tokens")

    model.save(str(model_path))
    print(f"[word2vec] Model saved to {model_path}")
    return model


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(model: Word2Vec) -> None:
    test_pairs = [
        ("python", 5),
        ("kubernetes", 5),
        ("nursing", 5),
        ("accounting", 5),
        ("machine", 5),
    ]
    print("\n[word2vec] === Smoke test — nearest neighbours ===")
    for word, topn in test_pairs:
        if word in model.wv:
            neighbours = model.wv.most_similar(word, topn=topn)
            formatted = ", ".join(f"{w}({s:.2f})" for w, s in neighbours)
            print(f"  {word:20s} → {formatted}")
        else:
            print(f"  {word:20s} → (not in vocabulary)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sentences = load_sentences(DB_PATH)
    model = train(sentences, MODEL_PATH)
    smoke_test(model)
    print("\n[word2vec] Done. Model ready for backend use.")
