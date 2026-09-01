# tests/test_db_connection.py
"""backend/db/connection.py의 get_conn() 컨텍스트 매니저 단위 테스트.

실제 Postgres 없이, getconn/putconn을 흉내 내는 가짜 풀로 "에러 시 롤백 후
반납"이 실제로 지켜지는지만 검증한다. 이게 없으면 aborted transaction 상태의
커넥션이 풀에 반납되어, 다음 요청이 그 커넥션을 빌렸을 때 원인불명 에러를 받는다.
"""

import pytest

from backend.db import connection


class _FakeConn:
    def __init__(self):
        self.rolled_back = False

    def rollback(self):
        self.rolled_back = True


class _FakePool:
    def __init__(self, conn):
        self._conn = conn
        self.putconn_called_with = None

    def getconn(self):
        return self._conn

    def putconn(self, conn):
        self.putconn_called_with = conn


class TestGetConn:
    def test_returns_connection_and_puts_back_on_success(self, monkeypatch):
        conn = _FakeConn()
        pool = _FakePool(conn)
        monkeypatch.setattr(connection, "_pool", pool)

        with connection.get_conn() as c:
            assert c is conn

        assert pool.putconn_called_with is conn
        assert conn.rolled_back is False

    def test_rolls_back_and_puts_back_on_exception(self, monkeypatch):
        conn = _FakeConn()
        pool = _FakePool(conn)
        monkeypatch.setattr(connection, "_pool", pool)

        with pytest.raises(ValueError):
            with connection.get_conn():        # 예외 경로 검증이라 커넥션 객체는 안 쓴다
                raise ValueError("boom")

        assert conn.rolled_back is True
        assert pool.putconn_called_with is conn
