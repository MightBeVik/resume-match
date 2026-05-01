# System Workflow

The ResumeMatch system operates through a three-stage lifecycle: **Data Pipeline**, **Model Training**, and **Runtime Analysis**.

## 1. Data Pipeline (The "Brain" Building)
1. **Ingest:** Download raw CSV/XLSX job postings from the Canadian Job Bank API.
2. **Transform:** Normalize column names, date formats, and NOC (National Occupational Classification) codes.
3. **Enrich:** Join postings with OaSIS competency data to attach standardized skill requirements.
4. **Synthesize:** Assemble fragmented data into a cohesive "Synthetic Job Description" text.
5. **Load:** Write the final enriched dataset into a SQLite `jobs.db`.

## 2. Model Training
- **Word2Vec:** The system tokenizes all synthesized job descriptions and (optionally) a corpus of resumes. It trains a Skip-gram Word2Vec model that maps the relationship between professional terms.
- **TF-IDF Fitting:** At server startup, the backend fits a TF-IDF vectorizer on thousands of job descriptions to establish "term importance" (IDF weights).

## 3. Runtime Analysis (The "Match" Logic)
When a user uploads a resume and selects a job:
1. **Extraction:**
   - PDF/DOCX text is extracted and normalized.
   - spaCy extracts Skills, Certifications, and Education.
   - Heuristics estimate total years of experience from resume date ranges.
2. **Parsing:**
   - The Job Description is split into Requirements, Responsibilities, and Preferred sections.
3. **Scoring:**
   - **Skills:** Matches extracted entities + SBERT synonyms + W2V suggestions.
   - **Experience:** Matches responsibility phrases and validates tenure (years).
   - **Education:** Validates degree levels (Bachelors, Masters, etc.) and fields.
   - **Semantic:** Computes overarching document similarity using corpus-fitted TF-IDF.
4. **Caching:** The final JSON result is cached via a SHA-256 hash of the inputs for instant retrieval in future requests.
