"""FastAPI application with CORS and model preloading."""
from contextlib import asynccontextmanager
from pathlib import Path

import nltk
import spacy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from app import models_state

W2V_MODEL_PATH = Path(__file__).resolve().parents[2] / "job-pipeline" / "data" / "word2vec_jobs.model"
DB_PATH = Path(__file__).resolve().parents[2] / "job-pipeline" / "data" / "jobs.db"

# Number of job descriptions to sample when fitting the TF-IDF corpus vectorizer.
# More = better IDF weights, slower startup. 5000 is a good balance.
TFIDF_CORPUS_SAMPLE = 5_000


def _build_tfidf_vectorizer(db_path: Path):
    """Fit a TfidfVectorizer on a sample of job descriptions from jobs.db.

    By fitting on a real corpus, each term's IDF weight reflects how
    common or rare it is across many documents — far more meaningful than
    the degenerate IDF you get from a 2-document fit.
    """
    import sqlite3
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not db_path.exists():
        print(f"[startup] jobs.db not found at {db_path} — TF-IDF will use 2-doc fit")
        return None

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT jd_text FROM enriched_jobs "
            "WHERE jd_text IS NOT NULL AND TRIM(jd_text) != '' "
            f"LIMIT {TFIDF_CORPUS_SAMPLE}"
        )
        rows = cur.fetchall()
        conn.close()
    except Exception as exc:
        print(f"[startup] Could not query jobs.db for TF-IDF corpus: {exc}")
        return None

    corpus = [row[0] for row in rows if row[0]]
    if not corpus:
        return None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,      # log(1+tf) dampens very frequent terms
        max_features=30_000,    # cap vocabulary for memory efficiency
        min_df=3,               # ignore tokens in fewer than 3 documents
    )
    vectorizer.fit(corpus)
    print(f"[startup] TF-IDF vectorizer fitted on {len(corpus):,} job descriptions "
          f"— vocab size: {len(vectorizer.vocabulary_):,}")
    return vectorizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load NLP models at startup."""
    # Download NLTK data
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    # Load spaCy model
    try:
        models_state.nlp_model = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        models_state.nlp_model = spacy.load("en_core_web_sm")

    # Load sentence-transformers model
    models_state.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load Word2Vec model trained on combined job + resume corpus
    if W2V_MODEL_PATH.exists():
        from gensim.models import Word2Vec
        models_state.word2vec_model = Word2Vec.load(str(W2V_MODEL_PATH))
        print(f"[startup] Word2Vec loaded — vocab size: {len(models_state.word2vec_model.wv):,}")
    else:
        print(f"[startup] Word2Vec model not found at {W2V_MODEL_PATH} — skill expansion disabled")

    # Fit TF-IDF vectorizer on job corpus for meaningful IDF weights
    models_state.tfidf_vectorizer = _build_tfidf_vectorizer(DB_PATH)

    yield


app = FastAPI(title="ResumeMatch API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.routes import router  # noqa: E402

app.include_router(router, prefix="/api")
