"""Train a Word2Vec model on the combined jobs + resume corpus.

Corpus sources:
  1. jd_text from enriched_jobs table in jobs.db  (job description language)
  2. data/raw/resumes/UpdatedResumeDataSet.csv     (resume language — optional)

Combining both corpora teaches the model to bridge job-posting vocabulary
("proficient in Python") with resume vocabulary ("developed Python scripts"),
which improves synonym matching when a JD requirement is compared against
a user's resume.

The saved model is written to data/word2vec_jobs.model and loaded by the
backend at startup for skill-expansion in the NLP matching pipeline.
"""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

import pandas as pd

from gensim.models import Word2Vec

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "jobs.db"
MODEL_PATH = ROOT / "data" / "word2vec_jobs.model"
RESUME_CSV_PATH = ROOT / "data" / "raw" / "resumes" / "resume_data.csv"

# Columns in resume_data.csv that contain useful text for training.
# These are assembled into one document per resume row.
RESUME_TEXT_COLUMNS = [
    "career_objective",
    "skills",
    "responsibilities",
    "positions",
    "related_skils_in_job",
    "certification_skills",
    "major_field_of_studies",
]

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


def load_jd_sentences(db_path: Path) -> list[list[str]]:
    """Load tokenised job-description sentences from jobs.db."""
    if not db_path.exists():
        raise FileNotFoundError(f"jobs.db not found at {db_path}. Run job-pipeline first.")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT jd_text FROM enriched_jobs WHERE jd_text IS NOT NULL AND TRIM(jd_text) != ''")
    rows = cur.fetchall()
    conn.close()

    print(f"[word2vec] Loaded {len(rows):,} job descriptions from jobs.db")

    sentences: list[list[str]] = []
    for (jd_text,) in rows:
        tokens = _tokenize(jd_text)
        if tokens:
            sentences.append(tokens)

    print(f"[word2vec] Tokenised {len(sentences):,} JD sentences")
    return sentences


def load_resume_sentences(csv_path: Path) -> list[list[str]]:
    """Load tokenised resume sentences from the structured resume_data.csv.

    The file has one resume per row split across many columns.  We assemble
    the most informative text columns into a single document per row, then
    tokenise that document — no manual preprocessing required.

    If the file does not exist the function returns an empty list so that
    training can continue with the JD corpus only.
    """
    if not csv_path.exists():
        print(
            f"[word2vec] Resume CSV not found at {csv_path} — "
            "training with job descriptions only.\n"
            f"         → Place resume_data.csv at: {csv_path}"
        )
        return []

    try:
        # utf-8-sig strips the BOM character present in this file
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001
        print(f"[word2vec] Could not read resume CSV ({exc}) — skipping resume corpus.")
        return []

    # Normalise column names (strip BOM / whitespace that may survive)
    df.columns = [col.strip().lstrip("\ufeff") for col in df.columns]

    # Keep only the columns that exist in this file
    usable_cols = [col for col in RESUME_TEXT_COLUMNS if col in df.columns]
    if not usable_cols:
        print(
            f"[word2vec] None of the expected columns found in {csv_path.name}. "
            f"Available: {list(df.columns)[:10]} — skipping resume corpus."
        )
        return []

    print(f"[word2vec] Assembling resume text from columns: {usable_cols}")

    sentences: list[list[str]] = []
    for _, row in df.iterrows():
        parts: list[str] = []
        for col in usable_cols:
            val = row.get(col)
            if pd.isna(val) or not str(val).strip():
                continue
            # Strip Python-list literals like "['Python', 'Java']"
            text = re.sub(r"[\[\]'\"]", " ", str(val))
            parts.append(text.strip())
        combined = " ".join(parts)
        tokens = _tokenize(combined)
        if tokens:
            sentences.append(tokens)

    print(f"[word2vec] Tokenised {len(sentences):,} resume sentences from {csv_path.name}")
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
    jd_sentences = load_jd_sentences(DB_PATH)
    resume_sentences = load_resume_sentences(RESUME_CSV_PATH)

    all_sentences = jd_sentences + resume_sentences
    print(
        f"[word2vec] Combined corpus: {len(all_sentences):,} sentences "
        f"({len(jd_sentences):,} JD + {len(resume_sentences):,} resume)"
    )

    if not all_sentences:
        raise RuntimeError("No training sentences available. Ensure jobs.db is populated.")

    model = train(all_sentences, MODEL_PATH)
    smoke_test(model)
    print("\n[word2vec] Done. Model ready for backend use.")
