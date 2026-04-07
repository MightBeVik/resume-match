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


class AnalyzeResponse(BaseModel):
    overall_score: int = Field(..., ge=0, le=100)
    verdict: str
    summary: str
    sections: dict[str, SectionResult]
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
