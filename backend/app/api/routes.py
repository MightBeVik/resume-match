"""API route definitions."""
from fastapi import APIRouter, HTTPException
from app.models.schemas import AnalyzeRequest, AnalyzeResponse
from app.nlp.matcher import analyze_match

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Analyze resume against job description using NLP pipeline."""
    try:
        result = analyze_match(request.resume_text, request.job_description)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"NLP pipeline failed: {str(e)}",
        )
