# backend/api/rate_limit.py
"""단일 프로세스 인메모리 rate limiter.

API 키가 유출되거나 프론트가 실수로 반복 호출해도 조항마다 OpenAI 호출이 도는
분석 엔드포인트의 비용이 무한정 새지 않도록, 클라이언트(API 키 우선, 없으면 IP)별
분당 요청 수를 제한한다.

여러 인스턴스로 수평 확장하면 프로세스마다 카운터가 따로 놀아 제한이 느슨해진다
— 그 시점엔 Redis 등 공유 저장소 기반으로 바꿔야 한다. 지금은 단일 인스턴스라
그런 인프라 없이 메모리 카운터로 충분하다.
"""

import os
import time
from collections import defaultdict

from fastapi import HTTPException, Request

_WINDOW_SECONDS = 60
_MAX_REQUESTS_PER_WINDOW = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "20"))

_hits: dict[str, list[float]] = defaultdict(list)


def _get_client_key(request: Request) -> str:
    api_key = request.headers.get("x-api-key")
    if api_key:
        return f"key:{api_key}"
    client = request.client
    return f"ip:{client.host}" if client else "ip:unknown"


async def enforce_rate_limit(request: Request) -> None:
    key = _get_client_key(request)
    now = time.monotonic()
    window_start = now - _WINDOW_SECONDS

    hits = _hits[key]
    while hits and hits[0] < window_start:
        hits.pop(0)

    if len(hits) >= _MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(status_code=429, detail="요청이 너무 많습니다. 잠시 후 다시 시도해주세요.")

    hits.append(now)
