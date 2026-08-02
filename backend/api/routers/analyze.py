# backend/api/routers/analyze.py
import io
import json

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, File
from fastapi.responses import StreamingResponse

from backend.api.auth import require_api_key
from backend.api.rate_limit import enforce_rate_limit
from backend.api.schemas import AnalyzeRequest, AnalyzeResponse
from backend.api.services.analyze import run_analyze, run_analyze_stream
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


@router.post("/api/analyze/stream", dependencies=_protected)
async def analyze_stream(body: AnalyzeRequest):
    if not body.text or len(body.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="계약서 내용이 너무 짧습니다.")

    # 검증·세마포어 획득은 여기서 끝낸다 — 실패 시 여기서 HTTPException이
    # 터져야 정상적인 4xx/5xx 응답이 나간다. StreamingResponse가 일단
    # 시작되면 상태 코드를 더 이상 바꿀 수 없다.
    events = await run_analyze_stream(body.text)

    async def sse():
        # 클라이언트가 스트리밍 도중 연결을 끊으면 Starlette은 이 바깥
        # 제너레이터의 aclose()만 호출한다 — async for가 GeneratorExit로 그냥
        # 중단되면 안쪽 제너레이터(events)의 aclose()는 자동으로 안 불려서,
        # 세마포어를 반납하는 events의 finally가 GC 타이밍까지 안 돈다
        # (직접 재현: aclose() 직후엔 반납 안 됨). 명시적으로 닫아서 세마포어가
        # 그 자리에서 바로 반납되게 한다.
        try:
            async for event in events:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            await events.aclose()

    return StreamingResponse(sse(), media_type="text/event-stream")


async def _extract_pdf_text(file: UploadFile) -> str:
    """PDF 업로드에서 텍스트를 뽑아낸다. /api/analyze-pdf와 스트리밍 버전이 공유한다."""
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
    return text


@router.post("/api/analyze-pdf", response_model=AnalyzeResponse, dependencies=_protected)
async def analyze_pdf(file: UploadFile = File(...)):
    text = await _extract_pdf_text(file)
    return await run_analyze(text)


@router.post("/api/analyze-pdf/stream", dependencies=_protected)
async def analyze_pdf_stream(file: UploadFile = File(...)):
    text = await _extract_pdf_text(file)

    # 검증(확장자·크기·파싱)과 세마포어 획득은 여기서 끝낸다 — StreamingResponse가
    # 시작되면 상태 코드를 더 이상 바꿀 수 없으므로 실패 경로는 첫 바이트 전에 처리한다.
    events = await run_analyze_stream(text)

    async def sse():
        # analyze_stream과 동일한 이유 — 연결 끊김 시 세마포어를 즉시 반납하려면
        # 안쪽 제너레이터를 명시적으로 닫아야 한다.
        try:
            async for event in events:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            await events.aclose()

    return StreamingResponse(sse(), media_type="text/event-stream")
