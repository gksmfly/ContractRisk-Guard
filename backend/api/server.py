# backend/api/server.py
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routers.analyze import router
from backend.db.connection import close_pool, init_pool
from backend.utils import load_logger

load_dotenv()

logger = load_logger("server.log")

# 서버 런타임에 실제로 필요한 값만 검사한다(FORWARD_MODEL/VERIFY_MODEL/HF_TOKEN은
# fb_check/training 같은 오프라인 파이프라인 전용이라 여기선 필요 없음).
_REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "DATABASE_URL"]


def _validate_required_env() -> None:
    """빠진 환경변수가 있으면 기동 시점에 바로 실패시킨다.

    이 검사가 없으면 서버는 정상 기동된 것처럼 보이다가, 첫 실제 요청에서야
    (예: OpenAI 호출 시점) 원인 파악하기 어려운 에러로 실패한다.
    """
    missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        raise RuntimeError(f".env에 다음 환경변수가 없습니다: {', '.join(missing)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # KoE5 임베딩 모델 로딩(HF 다운로드+GPU 적재)과 DB 커넥션 풀 생성을 기동 시
    # 미리 해둬서, 첫 요청 사용자가 콜드스타트 지연을 떠안지 않게 한다.
    from backend.api.services.retrieval import _get_cached_embedder

    _validate_required_env()
    init_pool()
    _get_cached_embedder()
    logger.info("서버 기동: DB 풀 초기화 + 임베더 워밍업 완료")
    yield
    close_pool()


app = FastAPI(title="ContractRiskGuard API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("CORS_ORIGIN", "http://localhost:3000")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # 예상 못 한 예외의 스택트레이스가 그대로 클라이언트에 노출되지 않도록 막고,
    # 원인 추적은 서버 로그로만 남긴다.
    logger.error(f"처리되지 않은 예외: {request.method} {request.url.path} — {exc!r}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "서버 내부 오류가 발생했습니다."})


app.include_router(router)
