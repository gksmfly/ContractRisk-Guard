# backend/db/loader.py
"""
DB 적재 스크립트 — PostgreSQL + pgvector

data/processed/chunks/*.jsonl (법령·판례·해석례 청크) 와
data/labels/seed_labeled.jsonl (Seed 라벨 데이터) 를
PostgreSQL에 임베딩과 함께 적재한다.

임베딩 모델: text-embedding-3-large (dim=3072)

사용법:
    python -m backend.db.loader              # 전체 적재
    python -m backend.db.loader --source chunks   # 청크만
    python -m backend.db.loader --source seed     # Seed만
    python -m backend.db.loader --source clean    # FB-Check CLEAN 데이터

환경변수 (.env):
    DATABASE_URL  postgresql://user:pass@host:5432/dbname
    OPENAI_API_KEY
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from backend.utils import load_logger, load_jsonl, PROJECT_ROOT

logger = load_logger("db_load.log")

DATABASE_URL   = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/crg")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBED_MODEL    = "text-embedding-3-large"
EMBED_DIM      = 1536  # HNSW 인덱스 최대 2000차원 제한, large 모델 dimensions 파라미터로 축소
BATCH_SIZE     = 100   # OpenAI API 배치 크기 (precedents 평균 221 토큰 × 100 = ~22,000, TPM 40K 이내)
UPSERT_BATCH   = 200   # DB insert 배치 크기

PROCESSED_DIR  = Path(os.environ.get("PROCESSED_DIR", str(PROJECT_ROOT / "data/processed")))
SEED_DIR       = Path(os.environ.get("SEED_DIR",      str(PROJECT_ROOT / "data/labels")))
FB_CHECK_DIR   = Path(os.environ.get("FB_CHECK_DIR",  str(PROJECT_ROOT / "data/fb_check")))

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT        PRIMARY KEY,
    source      VARCHAR(20) NOT NULL,
    doc_id      VARCHAR(200),
    rec_index   INTEGER,
    chunk_index INTEGER,
    text        TEXT        NOT NULL,
    metadata    JSONB,
    embedding   vector({dim})
);
CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks (source);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m=16, ef_construction=64);

CREATE TABLE IF NOT EXISTS seed_clauses (
    chunk_id        TEXT        PRIMARY KEY,
    source          VARCHAR(30) NOT NULL,
    doc_id          VARCHAR(200),
    text            TEXT        NOT NULL,
    domain          VARCHAR(20),
    risk_level      VARCHAR(10),
    risk_basis      TEXT,
    patterns_matched JSONB,
    metadata        JSONB,
    embedding       vector({dim})
);
CREATE INDEX IF NOT EXISTS seed_domain_idx ON seed_clauses (domain);
CREATE INDEX IF NOT EXISTS seed_risk_idx   ON seed_clauses (risk_level);
CREATE INDEX IF NOT EXISTS seed_embedding_idx ON seed_clauses
    USING hnsw (embedding vector_cosine_ops)
    WITH (m=16, ef_construction=64);

CREATE TABLE IF NOT EXISTS clean_clauses (
    chunk_id         TEXT        PRIMARY KEY,
    source           VARCHAR(30),
    doc_id           VARCHAR(200),
    text             TEXT        NOT NULL,
    domain           VARCHAR(20),
    risk_level       VARCHAR(10),
    forward_label    VARCHAR(10),
    backward_label   VARCHAR(10),
    evidence_span    TEXT,
    reasoning        TEXT,
    metadata         JSONB,
    embedding        vector({dim})
);
CREATE INDEX IF NOT EXISTS clean_domain_idx ON clean_clauses (domain);
CREATE INDEX IF NOT EXISTS clean_embedding_idx ON clean_clauses
    USING hnsw (embedding vector_cosine_ops)
    WITH (m=16, ef_construction=64);
""".format(dim=EMBED_DIM)


def _get_conn():
    try:
        import psycopg2
    except ImportError:
        logger.error("psycopg2 미설치: pip install psycopg2-binary")
        sys.exit(1)
    try:
        conn = psycopg2.connect(DATABASE_URL)
    except Exception as e:
        logger.error(f"DB 연결 실패: {e}")
        sys.exit(1)
    return conn


def _get_openai():
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai 미설치: pip install openai")
        sys.exit(1)
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY 환경변수 없음")
        sys.exit(1)
    return OpenAI(api_key=OPENAI_API_KEY)


def init_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()
    logger.info("  스키마 초기화 완료")


_RATE_LIMIT_RE = re.compile(
    r"(?:retry after|Please try again in)\s*(\d+)\s*s", re.IGNORECASE
)
# precedents 평균 221 토큰 × 100개 = 22,100 토큰 → 40K TPM 한도에서 배치당 최소 34s 필요
_BATCH_INTERVAL = 35  # 초; 마지막 배치엔 대기 불필요


def _parse_retry_after(err_msg: str) -> int | None:
    m = _RATE_LIMIT_RE.search(err_msg)
    return int(m.group(1)) if m else None


def embed_texts(client, texts: list[str]) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    total = len(texts)
    for i in range(0, total, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        t_start = time.time()
        for attempt in range(3):
            try:
                resp = client.embeddings.create(
                    model=EMBED_MODEL, input=batch, dimensions=EMBED_DIM
                )
                all_embeddings.extend([d.embedding for d in resp.data])
                break
            except Exception as e:
                err = str(e)
                if "Request too large" in err or "maximum context length" in err.lower():
                    logger.error(
                        f"  배치 크기 초과 (idx={i}, size={len(batch)}): BATCH_SIZE를 줄이세요"
                    )
                    raise
                wait = _parse_retry_after(err) or (10 * 2 ** attempt)
                logger.warning(f"  임베딩 재시도 {attempt+1}/3 ({wait}s 대기): {err[:120]}")
                time.sleep(wait)
        # 마지막 배치가 아니면 TPM 초과 방지를 위해 인터벌 보장
        if i + BATCH_SIZE < total:
            elapsed = time.time() - t_start
            sleep_for = max(0.0, _BATCH_INTERVAL - elapsed)
            if sleep_for:
                time.sleep(sleep_for)
        logger.info(f"  임베딩 진행: {min(i + BATCH_SIZE, total)}/{total}")
    return all_embeddings


def _existing_ids(conn, table: str) -> set[str]:
    """테이블에 이미 존재하는 chunk_id 집합을 반환한다."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT chunk_id FROM {table}")
        return {row[0] for row in cur.fetchall()}


def _upsert_chunks(conn, table: str, cols: list[str], rows: list[tuple]) -> None:
    from psycopg2.extras import execute_values, Json
    rows = [
        tuple(Json(v) if isinstance(v, (dict, list)) else v for v in row)
        for row in rows
    ]
    col_str = ", ".join(cols)
    sql = (
        f"INSERT INTO {table} ({col_str}) VALUES %s "
        f"ON CONFLICT (chunk_id) DO UPDATE SET "
        + ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c != "chunk_id")
    )
    with conn.cursor() as cur:
        for i in range(0, len(rows), UPSERT_BATCH):
            execute_values(cur, sql, rows[i : i + UPSERT_BATCH])
    conn.commit()


def load_chunks(conn, client) -> dict[str, int]:
    """data/processed/chunks/*.jsonl → chunks 테이블."""
    sources = ["laws", "precedents", "interpretations"]
    total_inserted = 0

    existing = _existing_ids(conn, "chunks")
    if existing:
        logger.info(f"  [chunks] 이미 적재된 chunk_id {len(existing)}개 — 건너뜀")

    for src in sources:
        path = PROCESSED_DIR / f"{src}.jsonl"
        if not path.exists():
            logger.warning(f"  {path} 없음 — 건너뜀")
            continue

        all_records = load_jsonl(path)
        records = [r for r in all_records if r["chunk_id"] not in existing]
        skipped = len(all_records) - len(records)
        if skipped:
            logger.info(f"  [chunks/{src}] {skipped}건 이미 적재 — 건너뜀")
        if not records:
            continue

        logger.info(f"  [chunks/{src}] {len(records)}개 청크 임베딩 시작")
        texts = [r["text"] for r in records]
        embeddings = embed_texts(client, texts)

        rows = [
            (
                r["chunk_id"], r["source"], r.get("doc_id"),
                r.get("rec_index"), r.get("chunk_index"),
                r["text"], r.get("metadata"), emb,
            )
            for r, emb in zip(records, embeddings)
        ]
        cols = ["chunk_id", "source", "doc_id", "rec_index", "chunk_index", "text", "metadata", "embedding"]
        _upsert_chunks(conn, "chunks", cols, rows)
        logger.info(f"  [chunks/{src}] {len(rows)}건 적재 완료")
        total_inserted += len(rows)

    return {"inserted": total_inserted}


def load_seed(conn, client) -> dict[str, int]:
    """data/labels/seed_labeled.jsonl → seed_clauses 테이블."""
    path = SEED_DIR / "seed_labeled.jsonl"
    if not path.exists():
        logger.error(f"  {path} 없음 — seed_label.py 먼저 실행하세요")
        return {"inserted": 0}

    all_records = load_jsonl(path)
    existing = _existing_ids(conn, "seed_clauses")
    records = [r for r in all_records if r["chunk_id"] not in existing]
    skipped = len(all_records) - len(records)
    if skipped:
        logger.info(f"  [seed] {skipped}건 이미 적재 — 건너뜀")
    if not records:
        logger.info("  [seed] 모두 적재 완료 상태")
        return {"inserted": 0}
    logger.info(f"  [seed] {len(records)}건 임베딩 시작")

    texts = [r["text"] for r in records]
    embeddings = embed_texts(client, texts)

    rows = [
        (
            r["chunk_id"], r["source"], r.get("doc_id"), r["text"],
            r["domain"], r["risk_level"], r["risk_basis"],
            r.get("patterns_matched", []), r.get("metadata"), emb,
        )
        for r, emb in zip(records, embeddings)
    ]
    cols = ["chunk_id", "source", "doc_id", "text", "domain", "risk_level",
            "risk_basis", "patterns_matched", "metadata", "embedding"]
    _upsert_chunks(conn, "seed_clauses", cols, rows)
    logger.info(f"  [seed] {len(rows)}건 적재 완료")
    return {"inserted": len(rows)}


def load_clean(conn, client) -> dict[str, int]:
    """data/fb_check/clean.jsonl → clean_clauses 테이블."""
    path = FB_CHECK_DIR / "clean.jsonl"
    if not path.exists():
        logger.warning(f"  {path} 없음 — fb_check.py 먼저 실행하세요")
        return {"inserted": 0}

    all_records = load_jsonl(path)
    existing = _existing_ids(conn, "clean_clauses")
    records = [r for r in all_records if r["chunk_id"] not in existing]
    skipped = len(all_records) - len(records)
    if skipped:
        logger.info(f"  [clean] {skipped}건 이미 적재 — 건너뜀")
    if not records:
        logger.info("  [clean] 모두 적재 완료 상태")
        return {"inserted": 0}
    logger.info(f"  [clean] {len(records)}건 임베딩 시작")

    # 8192 토큰 한도 방지: 약 6000자 이상은 잘라냄 (한국어 ~2토큰/자 기준)
    MAX_CHARS = 6000
    texts = [r["text"][:MAX_CHARS] for r in records]
    embeddings = embed_texts(client, texts)

    rows = [
        (
            r["chunk_id"], r.get("source"), r.get("doc_id"), r["text"][:MAX_CHARS],
            r.get("forward_domain"), r.get("forward_label"), r.get("forward_label"),
            r.get("backward_risk"), r.get("evidence_span"), r.get("reasoning"),
            None, emb,
        )
        for r, emb in zip(records, embeddings)
    ]
    cols = ["chunk_id", "source", "doc_id", "text", "domain", "risk_level",
            "forward_label", "backward_label", "evidence_span", "reasoning",
            "metadata", "embedding"]
    _upsert_chunks(conn, "clean_clauses", cols, rows)
    logger.info(f"  [clean] {len(rows)}건 적재 완료")
    return {"inserted": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="PostgreSQL+pgvector DB 적재")
    parser.add_argument("--source", default="all",
                        choices=["all", "chunks", "seed", "clean"],
                        help="적재할 소스 (기본: all)")
    args = parser.parse_args()

    logger.info("========== DB 적재 시작 ==========")

    conn   = _get_conn()
    client = _get_openai()
    init_schema(conn)

    report: dict[str, Any] = {}

    if args.source in ("all", "chunks"):
        report["chunks"] = load_chunks(conn, client)

    if args.source in ("all", "seed"):
        report["seed"] = load_seed(conn, client)

    if args.source in ("all", "clean"):
        report["clean"] = load_clean(conn, client)

    conn.close()

    total = sum(v.get("inserted", 0) for v in report.values())
    logger.info(f"  결과: {report}")
    logger.info(f"========== DB 적재 완료 (총 {total}건) ==========")


if __name__ == "__main__":
    main()
