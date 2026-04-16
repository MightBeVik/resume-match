"""Request and response schemas for the analyze endpoint."""
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    resume_text: str = Field(..., min_length=50, max_length=15000)
    job_description: str = Field(..., min_length=50, max_length=10000)


class SectionResult(BaseModel):
    score: int = Field(..., ge=0, le=100)
    matched: list[str] = []
    partial: list[str] = []
    missing: list[str] = []


class TfidfKeyword(BaseModel):
    keyword: str
    weight: float


class SimilarityScores(BaseModel):
    tfidf_cosine: float
    semantic: float


class NlpDetails(BaseModel):
    jd_sections_parsed: dict[str, list[str]]
    resume_sections_parsed: dict[str, str]
    resume_entities: dict[str, list[str]]
    tfidf_top_keywords: dict[str, list[TfidfKeyword]]
    similarity_scores: SimilarityScores


class RewriteSuggestion(BaseModel):
    section: str
    missing_item: str
    suggestion: str
    related_in_resume: list[str] = []
    w2v_expanded: bool = False


class AnalyzeResponse(BaseModel):
    overall_score: int = Field(..., ge=0, le=100)
    verdict: str
    summary: str
    sections: dict[str, SectionResult]
    rewrite_suggestions: list[RewriteSuggestion] = []
    nlp_details: NlpDetails


class RankJobsRequest(BaseModel):
    resume_text: str = Field(..., min_length=50, max_length=15000)
    province: str | None = None
    city: str | None = None
    source: str | None = None
    broad_category: str | None = None
    job_title_query: str | None = None
    employer_query: str | None = None
    limit: int = Field(default=20, ge=1, le=100)
    candidate_pool: int = Field(default=100, ge=1, le=500)


class RankedJobResult(BaseModel):
    id: int
    source: str | None = None
    noc_code: str | None = None
    noc_title: str | None = None
    teer: int | None = None
    broad_category: str | None = None
    job_title: str | None = None
    employer_name: str | None = None
    city: str | None = None
    province: str | None = None
    salary: str | None = None
    date_posted: str | None = None
    overall_score: int = Field(..., ge=0, le=100)
    verdict: str
    summary: str
    top_matches: list[str]
    top_gaps: list[str]


class RankJobsResponse(BaseModel):
    total_jobs_considered: int = Field(..., ge=0)
    returned: int = Field(..., ge=0)
    results: list[RankedJobResult]


# ---------------------------------------------------------------------------
# Job search schemas
# ---------------------------------------------------------------------------

class JobSearchRequest(BaseModel):
    keyword: str | None = None
    province: str | None = None
    city: str | None = None
    broad_category: str | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=30, ge=1, le=100)


class JobCard(BaseModel):
    id: int
    job_title: str | None = None
    employer_name: str | None = None
    city: str | None = None
    province: str | None = None
    salary: str | None = None
    broad_category: str | None = None
    date_posted: str | None = None
    jd_preview: str | None = None


class JobSearchResponse(BaseModel):
    total: int
    page: int
    page_size: int
    results: list[JobCard]


# ---------------------------------------------------------------------------
# Bulk match schemas
# ---------------------------------------------------------------------------

class BulkResumeEntry(BaseModel):
    name: str                 # display label (e.g. filename)
    resume_text: str = Field(..., min_length=50, max_length=15000)


class BulkMatchRequest(BaseModel):
    resumes: list[BulkResumeEntry] = Field(..., min_length=1, max_length=20)
    job_ids: list[int] | None = None          # specific job IDs to match against
    # OR filter params (used if job_ids is None)
    province: str | None = None
    city: str | None = None
    keyword: str | None = None
    broad_category: str | None = None
    max_jobs: int = Field(default=50, ge=1, le=200)


class BulkMatchCell(BaseModel):
    score: int
    verdict: str


class BulkMatchRow(BaseModel):
    resume_name: str
    best_score: int
    best_job_title: str | None
    best_job_id: int | None
    cells: dict[int, BulkMatchCell]   # job_id → score cell


class BulkMatchResponse(BaseModel):
    resumes: list[str]                # ordered resume names
    jobs: list[JobCard]               # ordered job cards
    rows: list[BulkMatchRow]          # one per resume
