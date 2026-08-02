# tests/test_rate_limit.py
"""backend/api/rate_limit.py의 인메모리 rate limiter 단위 테스트.

실제 ASGI 서버 없이 최소한의 scope로 Request 객체를 직접 구성해서 검증한다.
"""

import pytest
from fastapi import HTTPException, Request

from backend.api import rate_limit


def _make_request(headers: dict[str, str] | None = None, client_host: str = "1.2.3.4") -> Request:
    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": (client_host, 12345),
    }
    return Request(scope)


class TestEnforceRateLimit:
    @pytest.fixture(autouse=True)
    def _clear_hits(self):
        rate_limit._hits.clear()
        yield
        rate_limit._hits.clear()

    async def test_allows_requests_under_limit(self, monkeypatch):
        monkeypatch.setattr(rate_limit, "_MAX_REQUESTS_PER_WINDOW", 3)
        req = _make_request()
        for _ in range(3):
            await rate_limit.enforce_rate_limit(req)  # 예외 없어야 함

    async def test_blocks_requests_over_limit(self, monkeypatch):
        monkeypatch.setattr(rate_limit, "_MAX_REQUESTS_PER_WINDOW", 2)
        req = _make_request()
        await rate_limit.enforce_rate_limit(req)
        await rate_limit.enforce_rate_limit(req)
        with pytest.raises(HTTPException) as exc_info:
            await rate_limit.enforce_rate_limit(req)
        assert exc_info.value.status_code == 429

    async def test_different_api_keys_tracked_separately(self, monkeypatch):
        monkeypatch.setattr(rate_limit, "_MAX_REQUESTS_PER_WINDOW", 1)
        req_a = _make_request(headers={"x-api-key": "key-a"})
        req_b = _make_request(headers={"x-api-key": "key-b"})
        await rate_limit.enforce_rate_limit(req_a)
        await rate_limit.enforce_rate_limit(req_b)  # 다른 키라 안 막혀야 함
        with pytest.raises(HTTPException):
            await rate_limit.enforce_rate_limit(req_a)

    async def test_falls_back_to_ip_without_api_key(self, monkeypatch):
        monkeypatch.setattr(rate_limit, "_MAX_REQUESTS_PER_WINDOW", 1)
        req_a = _make_request(client_host="1.1.1.1")
        req_b = _make_request(client_host="2.2.2.2")
        await rate_limit.enforce_rate_limit(req_a)
        await rate_limit.enforce_rate_limit(req_b)  # 다른 IP라 안 막혀야 함
        with pytest.raises(HTTPException):
            await rate_limit.enforce_rate_limit(req_a)
