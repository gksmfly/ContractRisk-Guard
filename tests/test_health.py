# tests/test_health.py
"""backend/api/routers/analyze.py의 /health readiness 체크 단위 테스트.

실제 Postgres 없이, get_conn()을 가짜 컨텍스트 매니저로 바꿔치기해서
DB 연결 성공/실패 두 경로 모두 검증한다.
"""

from contextlib import contextmanager

from fastapi import Response

from backend.api.routers import analyze as analyze_router


class _FakeCursor:
    def execute(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _FakeConn:
    def cursor(self):
        return _FakeCursor()


@contextmanager
def _fake_get_conn_ok():
    yield _FakeConn()


@contextmanager
def _fake_get_conn_fail():
    raise ConnectionError("db down")
    yield  # pragma: no cover


class TestHealth:
    def test_db_reachable_returns_ok(self, monkeypatch):
        monkeypatch.setattr(analyze_router, "get_conn", _fake_get_conn_ok)
        response = Response()
        result = analyze_router.health(response)
        assert result == {"status": "ok", "db": "ok"}
        assert response.status_code == 200

    def test_db_unreachable_returns_503(self, monkeypatch):
        monkeypatch.setattr(analyze_router, "get_conn", _fake_get_conn_fail)
        response = Response()
        result = analyze_router.health(response)
        assert result == {"status": "degraded", "db": "unreachable"}
        assert response.status_code == 503
