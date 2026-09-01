# tests/api/test_pdf_upload.py
"""backend/api/routers/analyze.py의 PDF 업로드 방어 로직(확장자/크기/파싱 예외) 테스트.

인증·rate limit은 각각 test_auth.py/test_rate_limit.py가 다루므로 여기서는
API_KEY를 비워 우회하고 PDF 처리 경로만 검증한다. run_analyze()(OpenAI+GPU+DB)는
호출되지 않는 실패 케이스만 다루므로 별도 모킹이 필요 없다.
"""

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import auth
from backend.api.routers import analyze as analyze_router


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(auth, "API_KEY", "")
    app = FastAPI()
    app.include_router(analyze_router.router)
    return TestClient(app)


class TestAnalyzePdf:
    def test_rejects_non_pdf_extension(self, client: Any) -> None:
        resp = client.post(
            "/api/analyze-pdf",
            files={"file": ("contract.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 400

    def test_rejects_oversized_file(self, client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(analyze_router, "MAX_PDF_BYTES", 10)
        resp = client.post(
            "/api/analyze-pdf",
            files={"file": ("contract.pdf", b"x" * 100, "application/pdf")},
        )
        assert resp.status_code == 413

    def test_rejects_unparseable_pdf(self, client: Any) -> None:
        resp = client.post(
            "/api/analyze-pdf",
            files={"file": ("contract.pdf", b"not a real pdf", "application/pdf")},
        )
        assert resp.status_code == 400
        assert "PDF" in resp.json()["detail"]
