# backend/db/loader.py
"""
DB 적재 스크립트 — PostgreSQL + pgvector

data/processed/chunks/*.jsonl (법령·판례·해석례 청크) 와
data/labels/seed_labeled.jsonl (Seed 라벨 데이터) 를
PostgreSQL에 임베딩과 함께 적재한다.

임베딩 모델: nlpai-lab/KoE5 (dim=1024, local GPU)
  backend/db/embedding_benchmark.py 비교 결과, 기존 openai/text-embedding-3-large
  대비 관련/비관련 문서 분리도가 더 높고(0.1041 vs 0.0463) API 비용도 없어 채택.
  E5 계열 컨벤션에 따라 저장 문서는 "passage: " 프리픽스를 붙여 인코딩한다
  (검색 쿼리 임베딩 시에는 "query: " 프리픽스를 붙여야 함).

사용법:
    python -m backend.db.loader              # 전체 적재
    python -m backend.db.loader --source chunks   # 청크만
    python -m backend.db.loader --source seed     # Seed만
    python -m backend.db.loader --source clean    # FB-Check CLEAN 데이터
    python -m backend.db.loader --source noise    # FB-Check NOISE 데이터

환경변수 (.env):
    DATABASE_URL  postgresql://user:pass@host:5432/dbname
    EMBED_DEVICE  cuda:1 (기본값) / cuda:0 / cpu
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from backend.db.connection import DATABASE_URL
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger

logger = load_logger("db_load.log")

EMBED_MODEL  = "nlpai-lab/KoE5"
EMBED_DIM    = 1024
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cuda:1")
BATCH_SIZE   = 64    # KoE5 로컬 GPU 인코딩 배치 크기
UPSERT_BATCH = 200   # DB insert 배치 크기

PROCESSED_DIR  = Path(os.environ.get("PROCESSED_DIR", str(PROJECT_ROOT / "data/processed")))
SEED_DIR       = Path(os.environ.get("SEED_DIR",      str(PROJECT_ROOT / "data/labels")))
FB_CHECK_DIR   = Path(os.environ.get("FB_CHECK_DIR",  str(PROJECT_ROOT / "data/fb_check")))

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

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
-- Retrieval Strategy Agent의 Sparse(어휘) 검색용 — 형태소 분석 없이 문자 3-gram
-- 중복도로 유사도를 매기는 방식이라 한국어에도 별도 토크나이저 없이 바로 쓸 수 있다.
CREATE INDEX IF NOT EXISTS chunks_text_trgm_idx ON chunks
    USING gin (text gin_trgm_ops);

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

CREATE TABLE IF NOT EXISTS noise_clauses (
    chunk_id         TEXT        PRIMARY KEY,
    source           VARCHAR(30),
    doc_id           VARCHAR(200),
    text             TEXT        NOT NULL,
    domain           VARCHAR(20),
    risk_level       VARCHAR(10),
    forward_label    VARCHAR(10),
    verify_label     VARCHAR(10),
    noise_reason     TEXT,
    evidence_span    TEXT,
    reasoning        TEXT,
    metadata         JSONB,
    embedding        vector({dim})
);
CREATE INDEX IF NOT EXISTS noise_domain_idx ON noise_clauses (domain);
CREATE INDEX IF NOT EXISTS noise_reason_idx ON noise_clauses (noise_reason);
CREATE INDEX IF NOT EXISTS noise_embedding_idx ON noise_clauses
    USING hnsw (embedding vector_cosine_ops)
    WITH (m=16, ef_construction=64);
""".format(dim=EMBED_DIM)


def _get_conn() -> Any:
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


def get_embedder() -> Any:
    """KoE5 SentenceTransformer를 로드한다.

    backend.api.services.retrieval처럼 다른 모듈에서 검색 쿼리를 임베딩할 때도
    이 함수를 재사용해, DB 적재 시점과 검색 시점의 임베딩 모델이 어긋나지 않도록 한다.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers 미설치: pip install sentence-transformers")
        sys.exit(1)
    return SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)


def init_schema(conn) -> None:
    """빈 DB에 바로 붙여 쓰는 부트스트랩용 — 새 스키마 변경은 이제 alembic
    revision으로 관리한다(backend/db/migrations/, alembic.ini 참고). 이 함수는
    CREATE ... IF NOT EXISTS라 alembic이 이미 관리 중인 DB에 실행해도 안전하다.
    """
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()
    logger.info("  스키마 초기화 완료")


def embed_texts(model, texts: list[str], on_batch=None, prefix: str = "passage: ") -> list[list[float]]:
    """텍스트를 배치 단위로 임베딩한다 (KoE5, E5 컨벤션에 따른 프리픽스 부여).

    저장용 문서는 prefix="passage: "(기본값), 검색 쿼리는 prefix="query: "를 넘긴다.
    on_batch(start_idx, batch_embeddings)가 주어지면 배치가 끝날 때마다 즉시 호출된다.
    호출자는 이 콜백에서 해당 배치만 DB에 upsert하여, 중간에 프로세스가 끊겨도
    이미 처리된 배치는 보존되도록 한다. KoE5의 max_seq_length(512 토큰)를 넘는
    텍스트는 SentenceTransformer가 자동으로 잘라낸다.
    """
    all_embeddings: list[list[float]] = []
    total = len(texts)
    for i in range(0, total, BATCH_SIZE):
        batch = [f"{prefix}{t}" for t in texts[i : i + BATCH_SIZE]]
        batch_embeddings = model.encode(batch, convert_to_numpy=True).tolist()

        all_embeddings.extend(batch_embeddings)
        if on_batch:
            on_batch(i, batch_embeddings)
        logger.info(f"  임베딩 진행: {min(i + BATCH_SIZE, total)}/{total}")
    return all_embeddings


def _existing_ids(conn, table: str) -> set[str]:
    """테이블에 이미 존재하는 chunk_id 집합을 반환한다."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT chunk_id FROM {table}")
        return {row[0] for row in cur.fetchall()}


def _strip_nul(v: Any) -> Any:
    """Postgres text 컬럼은 NUL(0x00) 문자를 저장할 수 없다 (PDF 추출 과정에서 종종 섞여 들어옴)."""
    if isinstance(v, str):
        return v.replace("\x00", "")
    if isinstance(v, dict):
        return {k: _strip_nul(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_strip_nul(x) for x in v]
    return v


def _upsert_chunks(conn, table: str, cols: list[str], rows: list[tuple]) -> None:
    from psycopg2.extras import Json, execute_values
    rows = [
        tuple(
            Json(_strip_nul(v)) if isinstance(v, (dict, list)) else _strip_nul(v)
            for v in row
        )
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


def _load_incremental(
    conn, embedder, table: str, cols: list[str], records: list[dict], row_fn, label: str,
) -> int:
    """레코드를 배치 단위로 임베딩 + upsert한다.

    embed_texts의 on_batch 콜백을 이용해 배치가 성공할 때마다 즉시 DB에 반영하므로,
    중간에 프로세스가 끊겨도 이미 처리된 배치는 보존된다 (재실행 시 스킵됨).
    """
    texts = [r["text"] for r in records]

    def _on_batch(start_idx: int, batch_embeddings: list[list[float]]) -> None:
        batch_records = records[start_idx : start_idx + len(batch_embeddings)]
        rows = [row_fn(r, emb) for r, emb in zip(batch_records, batch_embeddings)]
        _upsert_chunks(conn, table, cols, rows)

    embed_texts(embedder, texts, on_batch=_on_batch)
    logger.info(f"  [{label}] {len(records)}건 적재 완료")
    return len(records)


def load_chunks(conn, embedder, refresh: tuple[str, ...] = ()) -> dict[str, int]:
    """data/processed/chunks/*.jsonl → chunks 테이블.

    기본은 증분 적재 — 이미 있는 `chunk_id`는 건너뛴다. 그래서 **텍스트가 바뀌어도
    반영되지 않는다**. 전처리 규칙을 고쳐 같은 chunk_id의 내용이 달라졌다면
    `refresh`에 그 소스를 넣어야 다시 임베딩·upsert된다
    (예: 법령 조문에 항·호·목을 추가한 변경 — `--refresh laws`).

    upsert 자체는 `ON CONFLICT (chunk_id) DO UPDATE`라 덮어쓰기가 되고, 없어진
    chunk_id는 남으므로 필요하면 `--purge`로 먼저 지운다.
    """
    sources = ["laws", "precedents", "interpretations"]
    total_inserted = 0
    cols = ["chunk_id", "source", "doc_id", "rec_index", "chunk_index", "text", "metadata", "embedding"]

    existing = _existing_ids(conn, "chunks")
    if existing:
        logger.info(f"  [chunks] 이미 적재된 chunk_id {len(existing)}개 — 건너뜀")
    if refresh:
        logger.info(f"  [chunks] 재적재 대상 소스: {', '.join(refresh)} (기존 chunk_id도 다시 임베딩)")

    for src in sources:
        path = PROCESSED_DIR / f"{src}.jsonl"
        if not path.exists():
            logger.warning(f"  {path} 없음 — 건너뜀")
            continue

        all_records = load_jsonl(path)
        skip_ids = set() if src in refresh else existing
        records = [r for r in all_records if r["chunk_id"] not in skip_ids]
        skipped = len(all_records) - len(records)
        if skipped:
            logger.info(f"  [chunks/{src}] {skipped}건 이미 적재 — 건너뜀")
        if not records:
            continue

        logger.info(f"  [chunks/{src}] {len(records)}개 청크 임베딩 시작")
        row_fn = lambda r, emb: (
            r["chunk_id"], r["source"], r.get("doc_id"),
            r.get("rec_index"), r.get("chunk_index"),
            r["text"], r.get("metadata"), emb,
        )
        n = _load_incremental(conn, embedder, "chunks", cols, records, row_fn, f"chunks/{src}")
        total_inserted += n

    return {"inserted": total_inserted}


def load_seed(conn, embedder) -> dict[str, int]:
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

    row_fn = lambda r, emb: (
        r["chunk_id"], r["source"], r.get("doc_id"), r["text"],
        r["domain"], r["risk_level"], r["risk_basis"],
        r.get("patterns_matched", []), r.get("metadata"), emb,
    )
    cols = ["chunk_id", "source", "doc_id", "text", "domain", "risk_level",
            "risk_basis", "patterns_matched", "metadata", "embedding"]
    n = _load_incremental(conn, embedder, "seed_clauses", cols, records, row_fn, "seed")
    return {"inserted": n}


def load_clean(conn, embedder) -> dict[str, int]:
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

    row_fn = lambda r, emb: (
        r["chunk_id"], r.get("source"), r.get("doc_id"), r["text"],
        r.get("forward_domain"), r.get("forward_label"), r.get("forward_label"),
        r.get("backward_risk"), r.get("evidence_span"), r.get("reasoning"),
        None, emb,
    )
    cols = ["chunk_id", "source", "doc_id", "text", "domain", "risk_level",
            "forward_label", "backward_label", "evidence_span", "reasoning",
            "metadata", "embedding"]
    n = _load_incremental(conn, embedder, "clean_clauses", cols, records, row_fn, "clean")
    return {"inserted": n}


def load_noise(conn, embedder) -> dict[str, int]:
    """data/fb_check/noise.jsonl → noise_clauses 테이블."""
    path = FB_CHECK_DIR / "noise.jsonl"
    if not path.exists():
        logger.warning(f"  {path} 없음 — fb_check.py 먼저 실행하세요")
        return {"inserted": 0}

    all_records = load_jsonl(path)
    existing = _existing_ids(conn, "noise_clauses")
    records = [r for r in all_records if r["chunk_id"] not in existing]
    skipped = len(all_records) - len(records)
    if skipped:
        logger.info(f"  [noise] {skipped}건 이미 적재 — 건너뜀")
    if not records:
        logger.info("  [noise] 모두 적재 완료 상태")
        return {"inserted": 0}
    logger.info(f"  [noise] {len(records)}건 임베딩 시작")

    row_fn = lambda r, emb: (
        r["chunk_id"], r.get("source"), r.get("doc_id"), r["text"],
        r.get("forward_domain"), r.get("forward_label"), r.get("forward_label"),
        r.get("verify_label"), r.get("noise_reason"), r.get("evidence_span"), r.get("reasoning"),
        None, emb,
    )
    cols = ["chunk_id", "source", "doc_id", "text", "domain", "risk_level",
            "forward_label", "verify_label", "noise_reason", "evidence_span", "reasoning",
            "metadata", "embedding"]
    n = _load_incremental(conn, embedder, "noise_clauses", cols, records, row_fn, "noise")
    return {"inserted": n}


def main() -> None:
    parser = argparse.ArgumentParser(description="PostgreSQL+pgvector DB 적재")
    parser.add_argument("--source", default="all",
                        choices=["all", "chunks", "seed", "clean", "noise"],
                        help="적재할 소스 (기본: all)")
    parser.add_argument("--refresh", nargs="*", default=[],
                        choices=["laws", "precedents", "interpretations"],
                        help="이미 적재된 chunk_id도 다시 임베딩할 소스 "
                             "(전처리 규칙이 바뀌어 같은 id의 텍스트가 달라졌을 때)")
    parser.add_argument("--purge", nargs="*", default=[],
                        choices=["law", "precedent", "interpretation"],
                        help="적재 전에 해당 source의 기존 청크를 삭제 "
                             "(청크 분할 방식이 바뀌어 없어진 chunk_id가 남을 때)")
    args = parser.parse_args()

    logger.info("========== DB 적재 시작 ==========")

    conn     = _get_conn()
    embedder = get_embedder()
    init_schema(conn)

    report: dict[str, Any] = {}

    if args.purge:
        with conn.cursor() as cur:
            for s in args.purge:
                cur.execute("DELETE FROM chunks WHERE source = %s", (s,))
                logger.info(f"  [purge] chunks source={s} {cur.rowcount}건 삭제")
        conn.commit()

    if args.source in ("all", "chunks"):
        report["chunks"] = load_chunks(conn, embedder, refresh=tuple(args.refresh))

    if args.source in ("all", "seed"):
        report["seed"] = load_seed(conn, embedder)

    if args.source in ("all", "clean"):
        report["clean"] = load_clean(conn, embedder)

    if args.source in ("all", "noise"):
        report["noise"] = load_noise(conn, embedder)

    conn.close()

    total = sum(v.get("inserted", 0) for v in report.values())
    logger.info(f"  결과: {report}")
    logger.info(f"========== DB 적재 완료 (총 {total}건) ==========")


if __name__ == "__main__":
    main()
