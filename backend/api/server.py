# backend/api/server.py
import os
from collections.abc import AsyncIterator
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

# 서버 런타임에 실제로 필요한 값.
#
# **`FORWARD_MODEL`이 여기 있는 이유** — 예전 주석은 "FORWARD_MODEL/VERIFY_MODEL/HF_TOKEN은
# 오프라인 파이프라인 전용이라 여기선 필요 없음"이라고 적혀 있었다. **틀렸다.**
# 서빙 import 체인이 오프라인 라벨링 모듈을 지나간다:
#
#     server → routers.analyze → services.analyze → agents.graph
#            → analysis_agent → fb_check.forward_labeling
#
# Analysis 단계가 `run_forward()`를 그대로 재사용하기 때문이다(논문 설계상 의도된 재사용 —
# FB-Check와 같은 프롬프트를 써야 라벨과 서빙 판단의 기준이 일치한다). 그래서 오프라인
# 모듈의 제약이 서빙으로 샌다. 2026-09-01에 `forward_labeling`이 최상단에서
# `os.environ["FORWARD_MODEL"]`을 읽고 있었고, 그 변수가 없으면 이 검사가 **돌기도 전에**
# import가 raw KeyError로 죽었다. `docker compose up`은 `env_file`에 값이 있어 떴기 때문에
# 아무도 몰랐고, README는 여전히 두 개만 안내하고 있었다.
#
# `VERIFY_MODEL`/`HF_TOKEN`은 **지금은** 체인 밖이라 넣지 않는다. 다만 그건 우연이고
# 체인은 바뀐다(2026-08-31에 게이트를 GPT에서 모델로 옮기며 한 번 바뀌었다) —
# 그래서 `tests/api/test_env_contract.py`가 "최상단 os.environ[...] 금지"를 부류로 막는다.
#
# 이 목록은 `.env.example`과 일치해야 한다(같은 테스트가 강제).
_REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "DATABASE_URL", "FORWARD_MODEL"]


def _validate_required_env() -> None:
    """빠진 환경변수가 있으면 기동 시점에 바로 실패시킨다.

    이 검사가 없으면 서버는 정상 기동된 것처럼 보이다가, 첫 실제 요청에서야
    (예: OpenAI 호출 시점) 원인 파악하기 어려운 에러로 실패한다.
    """
    missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        raise RuntimeError(f".env에 다음 환경변수가 없습니다: {', '.join(missing)}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # 첫 요청 사용자가 콜드스타트 지연을 떠안지 않도록 무거운 자원을 기동 시 미리 올린다.
    #
    # **EXAONE 워밍업은 제거했다.** 2026-08-21에 법령 라우팅을 상수(약관규제법 제6~14조
    # 고정)로 바꾸면서 `route_law_names()`의 운영 호출부가 0곳이 됐는데 워밍업만 남아 있었다
    # — 서버가 뜰 때마다 7.8B를 GPU에 15.6GB 올려놓고 아무도 쓰지 않았다. 예전 경고 문구가
    # 말하던 "조문 적중률 33%→8%"도 지금은 틀렸다(상수 스코핑으로 81%).
    # 라우터 모듈 자체는 재검토 여지가 있어 남겨두되, 기동 경로에서는 뺀다.
    #
    # 대신 **KoELECTRA를 워밍업에 추가한다.** 판단 모델인데 빠져 있어서
    # `judgment_agent.predict_articles()`가 첫 호출에 lazy 로드됐고, 지연 벤치마크에서
    # 1조항 20.4초 vs 이후 조항당 ~11초로 약 10초가 첫 사용자에게 전가되고 있었다.
    from backend.agents.judgment_agent import _get_electra
    from backend.api.services.retrieval import _get_cached_embedder

    _validate_required_env()
    init_pool()
    _get_cached_embedder()

    # 워밍업 실패가 서버 기동을 막지 않게 한다 — 실패해도 첫 요청에서 다시 시도된다.
    try:
        _get_electra()
    except Exception as e:
        logger.warning(f"KoELECTRA 워밍업 실패 — 첫 분석 요청에서 재시도됨: {e}")

    logger.info("서버 기동: DB 풀 초기화 + KoE5/KoELECTRA 워밍업 완료")
    yield
    close_pool()


app = FastAPI(title="Verilex API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("CORS_ORIGIN", "http://localhost:3000")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    # 예상 못 한 예외의 스택트레이스가 그대로 클라이언트에 노출되지 않도록 막고,
    # 원인 추적은 서버 로그로만 남긴다.
    logger.error(f"처리되지 않은 예외: {request.method} {request.url.path} — {exc!r}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "서버 내부 오류가 발생했습니다."})


app.include_router(router)
