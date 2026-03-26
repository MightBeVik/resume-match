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
