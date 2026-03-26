"""API route definitions."""
import io

import pdfplumber
from fastapi import APIRouter, HTTPException, UploadFile, File

from app.models.schemas import AnalyzeRequest, AnalyzeResponse
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
