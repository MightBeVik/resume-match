# Canadian Job Pipeline

This project ingests Canadian open job data, enriches it with NOC and OaSIS occupation metadata, synthesizes job-description text, and writes a SQLite database for resume ranking workflows.

## Output

The final database is written to `data/jobs.db` with these primary tables:

- `raw_postings`
- `noc_structure`
- `oasis_competencies`
- `enriched_jobs`

The `enriched_jobs` table extends the originally proposed schema with `job_title`, because ranking results are not usable without the posting title.

## Pipeline

Run the full pipeline:

```powershell
python scripts/run_all.py
```

Run individual steps:

```powershell
python scripts/01_ingest.py
python scripts/02_transform.py
python scripts/03_enrich.py
python scripts/04_synthesize.py
python scripts/05_load.py
python scripts/06_validate.py
```

## Data Flow

1. `01_ingest.py` downloads raw CKAN and direct CSV/XLSX sources into `data/raw/`
2. `02_transform.py` normalizes Job Bank postings plus NOC/OaSIS reference files into `data/processed/`
3. `03_enrich.py` joins postings to NOC/OaSIS competencies and writes `enriched_jobs.csv`
4. `04_synthesize.py` assembles synthetic JD text per posting
5. `05_load.py` writes the curated outputs into SQLite
6. `06_validate.py` verifies row counts, required outputs, and a few sample jobs

## Notes

- Job Bank postings are the primary ranking corpus.
- Alberta vacancy data is downloaded for future market-signal features. The current live dataset appears to publish PDF reports rather than posting-level CSV/XLSX resources, so it is not currently loaded into `enriched_jobs` as if it were a posting-level corpus.
- NOC codes are normalized to zero-padded 5-digit strings before joins.
- By default the pipeline downloads the two most recent English Job Bank CSV resources. Override with `JOB_BANK_RESOURCE_LIMIT` if you want more history.

## What You Still Need To Provide

- Which provinces or cities should be treated as the default ranking filters.
- Whether Alberta vacancy data should remain analytics-only or influence ranking.
- Whether you want the frontend updated next to show ranked jobs instead of only single-job analysis.