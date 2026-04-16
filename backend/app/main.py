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

    # Load Word2Vec model trained on Canadian job corpus
    if W2V_MODEL_PATH.exists():
        from gensim.models import Word2Vec
        models_state.word2vec_model = Word2Vec.load(str(W2V_MODEL_PATH))
        print(f"[startup] Word2Vec loaded — vocab size: {len(models_state.word2vec_model.wv):,}")
    else:
        print(f"[startup] Word2Vec model not found at {W2V_MODEL_PATH} — skill expansion disabled")

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
