# Technical Documentation

## Key Features

ResumeMatch contains an array of sophisticated features designed to bridge the gap between traditional applicant tracking systems and modern context-aware AI text parsing.

1. **Multi-Dimensional NLP Matching Engine**
   Instead of relying on simple keyword overlap, the engine calculates a composite match score by parsing job descriptions into requirements, responsibilities, and education. It extracts skills securely using `spaCy` and computes alignment over multiple independent dimensions.
   
2. **Bulk Matrix Leaderboard (Candidate vs. Job Pool)**
   Users can upload multiple resumes simultaneously (up to 20 files processed via `pdfplumber`) and rank them against up to 50 jobs. The application generates an intuitive, interactive UI matrix table highlighting the best candidate per job grouping.

3. **Word2Vec-Powered Rewrite Suggestions**
   If a candidate is missing specific skills listed in a job description, the system queries a custom-built Gensim Word2Vec model trained on Canadian Labour market data. It calculates contextual pathways to suggest precise action-verb rephrases utilizing the candidate's existing strengths to match the missing gap.

4. **Semantic SBERT Fallback**
   To resolve vocabulary discrepancies (e.g., "Developer" vs "Software Engineer"), the backend integrates HuggingFace's `sentence-transformers` (`all-MiniLM-L6-v2`) to capture and inject an overarching cosine-similarity text distribution into the final score.

5. **Locally Sourced Job Corpus Integration**
   The Next.js interactive job browser fetches paginated data drawn directly from real Canadian CKAN Job Bank CSV instances, pre-enriched with OaSIS occupational competencies and NOC identifiers via a robust SQLite pipeline.

---

## Model Performance & Accuracy

Due to the absence of a standard binary ground-truth dataset in subjective recruitment screening, model performance and accuracy in ResumeMatch are optimized through constrained NLP thresholds rather than raw loss-function ML loops. 

### 1. Scoring Integrity & Weighted Distribution
The system enforces accuracy by compartmentalizing the extraction algorithm. The composite (0-100) score is strictly weighted:
- **Skills (40%)**: Enforces hard overlap mixed with TF-IDF thresholding.
- **Experience (25%)**: Maps explicit responsibilities parsed from the JD.
- **Education (15%)**: Prevents false negatives via fuzzy degree-level detection bounds (e.g. associating "M.S" to "Master").
- **Preferred Qualifications (10%)**
- **SBERT Semantic Alignment (10%)**

### 2. Tiered Thresholding for Precision & Recall
To balance strict keyword hunting (high precision, low recall) with semantic embeddings (high recall, poor structural alignment), ResumeMatch uses Tiered NLP:
* **Check 1 (Direct Text / spaCy Entity Match):** 100% accuracy logic over distinct tokens.
* **Check 2 (SBERT Cosine Match >= 0.88):** High threshold semantic pairing guarantees matching synonyms without cross-domain collision.
* **Check 3 (Word2Vec Expansion):** For items undetected by direct phrasing, Word2Vec verifies geometric dataset clustering to flag items as "Partially Matched" avoiding unfair penalization for identical unlinked jargon.

### 3. Isolated Test Coverage Baseline
The `test_matcher.py` test suite sets a ground-truth quantitative constraint testing pipeline verifying correct semantic filtering behaviors:
- Validates a completely disparate resume (e.g. a "Chef" applying to a "Frontend Engineer" role) safely categorizes as a **Weak Match** dropping below a 40% threshold safely.
- Asserts strict parsing avoids false positives (i.e. appropriately identifying missing preferred qualifications like missing a Master's degree).
