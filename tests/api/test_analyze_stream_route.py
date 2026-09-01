# tests/api/test_analyze_stream_route.py
"""backend/api/routers/analyze.py의 SSE 엔드포인트(/api/analyze/stream) 테스트.

정상 완료 시 SSE 이벤트 모양을 확인하고, 클라이언트가 스트림을 끝까지 안 읽고
연결을 끊었을 때(TestClient의 stream 컨텍스트를 일찍 빠져나가는 것으로 흉내)도
세마포어가 즉시 반납되는지 — 즉 sse()의 try/finally: await events.aclose() 수정이
라우터 레벨에서도 실제로 동작하는지 — 확인한다.
"""

import asyncio
import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import auth
from backend.api.routers import analyze as analyze_router
from backend.api.services import analyze as analyze_service

_TEXT = "제1조(목적) 이 약관은 서비스 이용에 관한 사항을 규정한다."


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(auth, "API_KEY", "")
    monkeypatch.setattr(analyze_service, "_get_openai", lambda: object())
    monkeypatch.setattr(analyze_service, "_process_clause", lambda client, clause, index: None)
    app = FastAPI()
    app.include_router(analyze_router.router)
    return TestClient(app)


class TestAnalyzeStreamRoute:
    def test_stream_yields_progress_then_done(self, client: Any) -> None:
        with client.stream("POST", "/api/analyze/stream", json={"text": _TEXT}) as resp:
            assert resp.status_code == 200
            lines = [line for line in resp.iter_lines() if line.startswith("data: ")]

        events = [json.loads(line[len("data: "):]) for line in lines]
        assert events[-1]["type"] == "done"
        assert any(e["type"] == "progress" for e in events)

    def test_early_disconnect_releases_semaphore(self, client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(analyze_service, "_analyze_semaphore", asyncio.Semaphore(1))
        monkeypatch.setattr(analyze_service, "_QUEUE_TIMEOUT_SECONDS", 0.5)

        with client.stream("POST", "/api/analyze/stream", json={"text": _TEXT}) as resp:
            next(resp.iter_lines())  # 첫 줄만 받고 with 블록을 빠져나가 연결을 끊는다

        # 세마포어가 안 풀렸으면 여기서 503(큐 타임아웃)이 난다.
        resp2 = client.post("/api/analyze/stream", json={"text": _TEXT})
        assert resp2.status_code == 200
