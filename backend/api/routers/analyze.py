# backend/api/routers/analyze.py
import io

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, File

from backend.api.auth import require_api_key
from backend.api.rate_limit import enforce_rate_limit
from backend.api.schemas import AnalyzeRequest, AnalyzeResponse
from backend.api.services.analyze import run_analyze
from backend.db.connection import get_conn

router = APIRouter()
_protected = [Depends(require_api_key), Depends(enforce_rate_limit)]

# frontend/lib/config.ts의 MAX_UPLOAD_SIZE_BYTES와 동일한 상한.
# 클라이언트 쪽 체크는 우회 가능하므로 서버가 최종 방어선.
MAX_PDF_BYTES = 10 * 1024 * 1024


@router.get("/health")
def health(response: Response) -> dict[str, str]:
    """DB까지 확인하는 readiness 체크.

    이전엔 무조건 {"status": "ok"}만 리턴해서 Postgres가 죽어있어도 health check가
    통과했다 — 오케스트레이터/모니터링이 실제 장애를 못 잡았다. SELECT 1로 DB
    연결까지 확인하고, 실패하면 503으로 내려 로드밸런서가 바로 감지하게 한다.
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
    except Exception:
        response.status_code = 503
        return {"status": "degraded", "db": "unreachable"}
    return {"status": "ok", "db": "ok"}


@router.post("/api/analyze", response_model=AnalyzeResponse, dependencies=_protected)
async def analyze(body: AnalyzeRequest):
    if not body.text or len(body.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="계약서 내용이 너무 짧습니다.")
    return await run_analyze(body.text)


@router.post("/api/analyze-pdf", response_model=AnalyzeResponse, dependencies=_protected)
async def analyze_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 지원합니다.")
    try:
        from pypdf import PdfReader
    except ImportError:
        raise HTTPException(status_code=500, detail="pypdf 미설치")

    # Content-Length 헤더는 신뢰할 수 없으므로(스푸핑 가능) 청크 단위로 읽으며
    # 직접 누적 크기를 세어 초과 시 즉시 중단한다 (메모리 고갈 방지).
    data = bytearray()
    while chunk := await file.read(1024 * 1024):
        data.extend(chunk)
        if len(data) > MAX_PDF_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"파일 크기는 {MAX_PDF_BYTES // (1024 * 1024)}MB를 초과할 수 없습니다.",
            )

    try:
        reader = PdfReader(io.BytesIO(bytes(data)))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="PDF 파일을 읽을 수 없습니다. 손상되었거나 암호화된 파일일 수 있습니다.",
        )

    if len(text) < 20:
        raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다.")
    return await run_analyze(text)
