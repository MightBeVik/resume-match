# Technical Documentation

## Key Features

ResumeMatch contains an array of sophisticated features designed to bridge the gap between traditional applicant tracking systems and modern context-aware AI text parsing.

1. **Multi-Dimensional NLP Matching Engine**
   Instead of relying on simple keyword overlap, the engine calculates a composite match score by parsing job descriptions into requirements, responsibilities, and education. It extracts skills using an expanded dictionary of 150+ modern technologies and professional certifications.

2. **Heuristic Section Parsing**
   For unstructured job descriptions (common in copy-pasted posts), the system utilizes a heuristic line-classifier. By identifying action verbs and requirement signal phrases, it correctly categorizes text into responsibilities and requirements even in the absence of explicit headers.

3. **Years-of-Experience Validation**
   The engine parses numerical experience requirements (e.g., "5+ years of Python") from the job description and compares them against a tenure estimation calculated from date ranges (e.g., "2018 - 2022") found within the resume.

4. **Word2Vec-Powered Rewrite Suggestions**
   The system queries a custom-built Gensim Word2Vec model trained on a combined corpus of Canadian Labour market data and professional resumes. This allows the system to bridge the gap between "recruiter jargon" and "candidate phrasing," suggesting precise rephrasing for missing skills.

5. **Corpus-Fitted TF-IDF Similarity**
   To resolve vocabulary discrepancies, the backend integrates HuggingFace's `all-MiniLM-L6-v2`. This is augmented by a TF-IDF vectorizer pre-fitted on 5,000+ real job descriptions, ensuring that IDF weights reflect the true rarity of technical terms across the industry.

6. **Performance Optimization (Caching)**
   To handle heavy operations like the Bulk Matrix Leaderboard (O(N*M) complexity), the system implements SHA-256 hash-based response caching. Identical Resume/JD pairs are resolved in constant time after the initial analysis.

---

## Model Performance & Accuracy

### 1. Scoring Integrity & Weighted Distribution
The system enforces accuracy by compartmentalizing the extraction algorithm. The composite (0-100) score is strictly weighted:
- **Skills (40%)**: Enforces hard overlap mixed with TF-IDF thresholding.
- **Experience (25%)**: Maps explicit responsibilities and validates tenure.
- **Education (15%)**: Prevents false negatives via fuzzy degree-level detection bounds.
- **Preferred Qualifications (10%)**
- **Semantic Alignment (10%)**: Captured via SentenceBERT with corpus-fitted weights.

### 2. Tiered Thresholding for Precision & Recall
* **Tier 1 (Direct Entity Match):** 100% accuracy logic over distinct tokens (Skills & Certs).
* **Tier 2 (SBERT Cosine Match >= 0.88):** High threshold semantic pairing for synonyms.
* **Tier 3 (Word2Vec Expansion):** Geometric dataset clustering to flag "Partially Matched" items, avoiding penalization for synonymous jargon.

### 3. File Processing
- **PDF Extraction:** Handled via `pdfplumber` for high-fidelity text recovery.
- **DOCX Extraction:** Handled via `python-docx`, including recursive extraction from nested tables which often house resume content.

### 4. Training Pipeline
The Word2Vec model is trained using Skip-gram architecture on a synthesized corpus. By including resume data, the model learns the semantic proximity between requirement phrasing ("Experience in...") and achievement phrasing ("Developed and led...").
