# Project Overview

## Vision
ResumeMatch aims to democratize the recruitment process by providing candidates and recruiters with a high-fidelity, context-aware matching engine. By leveraging Canadian open labour-market data, the platform provides objective scores that go beyond simple keyword matching.

## Key Value Propositions
- **For Candidates:** Understand exactly how your resume aligns with real-world requirements and get actionable suggestions on how to rephrase your experience to match recruiter expectations.
- **For Recruiters:** Instantly rank a pool of candidates against a set of target jobs, identifying the best fit based on skills, experience tenure, and education.
- **Data-Driven:** Unlike black-box AI models, ResumeMatch uses a multi-tiered approach combining dictionary lookups, heuristic parsing, and semantic embeddings for explainable results.

## Technology Stack
- **Frontend:** Next.js, React 19, Tailwind CSS.
- **Backend:** FastAPI (Python), SQLite.
- **NLP Engine:** spaCy (Entity Extraction), Sentence-Transformers (Embeddings), Gensim (Word2Vec), scikit-learn (TF-IDF).
- **Data Pipeline:** pandas-driven ETL pipeline ingesting Job Bank, NOC, and OaSIS datasets.

## Unique Features
- **Heuristic Sectioning:** Automatically understands unstructured job descriptions.
- **Tenure Validation:** Compares "Years of Experience" requirements against resume dates.
- **Cross-Domain Word2Vec:** Bridges the gap between resume language and job description language.
