import io

from fastapi import APIRouter, HTTPException, UploadFile, File

from backend.api.schemas import AnalyzeRequest, AnalyzeResponse
from backend.api.services.analyze import run_analyze

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest):
    if not body.text or len(body.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="계약서 내용이 너무 짧습니다.")
    return await run_analyze(body.text)


@router.post("/api/analyze-pdf", response_model=AnalyzeResponse)
async def analyze_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 지원합니다.")
    try:
        from pypdf import PdfReader
    except ImportError:
        raise HTTPException(status_code=500, detail="pypdf 미설치")

    data = await file.read()
    reader = PdfReader(io.BytesIO(data))
    text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    if len(text) < 20:
        raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다.")
    return await run_analyze(text)
