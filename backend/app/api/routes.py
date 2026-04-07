"""API route definitions."""
import io

import pdfplumber
from fastapi import APIRouter, HTTPException, UploadFile, File

from app.jobs.ranker import rank_jobs_for_resume
from app.models.schemas import AnalyzeRequest, AnalyzeResponse, RankJobsRequest, RankJobsResponse
from app.nlp.matcher import analyze_match

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """Extract text from an uploaded PDF resume."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are supported")

    try:
        contents = await file.read()
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )

        text = text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="Could not extract text from PDF. The file may be image-based.")

        return {"text": text, "filename": file.filename, "pages": len(pdf.pages)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


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


@router.post("/rank-jobs", response_model=RankJobsResponse)
async def rank_jobs(request: RankJobsRequest):
    """Rank stored jobs against a resume using the NLP matcher."""
    try:
        return rank_jobs_for_resume(request.model_dump())
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Job ranking failed: {str(exc)}") from exc
