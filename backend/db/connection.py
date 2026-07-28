# backend/db/connection.py
"""API 요청 경로에서 공용으로 쓰는 psycopg2 커넥션 풀.

backend.api.services.retrieval은 요청(조항)마다 검색을 수행하므로, 매번
psycopg2.connect()로 새 연결을 맺으면 TCP+인증 왕복 비용이 그대로 지연시간에
더해지고 동시 요청이 몰리면 Postgres max_connections를 고갈시킬 수 있다.
이 모듈은 프로세스 수명 동안 재사용하는 풀을 제공한다.

배치 스크립트(backend/db/loader.py)는 단발성 장시간 연결 하나만 쓰면 되므로
풀을 쓰지 않고 이 모듈의 DATABASE_URL만 공유한다.
"""

import os
from contextlib import contextmanager
from typing import Iterator

from dotenv import load_dotenv

load_dotenv()

# docker/docker-compose.yml의 기본값과 일치시켜, .env 없이 실행해도
# docker compose up 직후 바로 연결되도록 한다.
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://contractrisk-guard-db:1234@localhost:5432/contractrisk-guard-db",
)

_pool = None


def init_pool(minconn: int = 1, maxconn: int = 10) -> None:
    global _pool
    if _pool is not None:
        return
    from psycopg2.pool import ThreadedConnectionPool
    _pool = ThreadedConnectionPool(minconn, maxconn, DATABASE_URL)


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


@contextmanager
def get_conn() -> Iterator[object]:
    """풀에서 연결을 빌려 쓰고 반납한다. init_pool()이 아직 안 불렸으면 지연 초기화한다.

    커서 실행 중 예외가 나면 트랜잭션이 aborted 상태로 남는데, 그대로 풀에
    반납하면 다음 대여자가 첫 쿼리부터 "current transaction is aborted" 에러를
    받는다. 반납 전에 롤백해 커넥션을 깨끗한 상태로 되돌린다.
    """
    if _pool is None:
        init_pool()
    conn = _pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)
