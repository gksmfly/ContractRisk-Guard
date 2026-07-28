"""initial schema

기존 backend/db/loader.py의 DDL(chunks/seed_clauses/clean_clauses/noise_clauses)을
그대로 옮긴 베이스라인. CREATE ... IF NOT EXISTS라서 loader.py의 init_schema()로
이미 스키마를 만들어둔 기존 DB에 대해서도 안전하게 실행할 수 있다(alembic_version
테이블만 새로 생김). 이 리비전 이후의 스키마 변경은 새 alembic revision으로 관리한다.

Revision ID: 91afb62fa923
Revises:
Create Date: 2026-07-28 02:24:51.358993

"""
from alembic import op

from backend.db.loader import EMBED_DIM

# revision identifiers, used by Alembic.
revision = "91afb62fa923"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id    TEXT        PRIMARY KEY,
            source      VARCHAR(20) NOT NULL,
            doc_id      VARCHAR(200),
            rec_index   INTEGER,
            chunk_index INTEGER,
            text        TEXT        NOT NULL,
            metadata    JSONB,
            embedding   vector({EMBED_DIM})
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks (source)")
    op.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m=16, ef_construction=64)
    """)
    # Retrieval Strategy Agent의 Sparse(어휘) 검색용 — 문자 3-gram 유사도.
    op.execute("""
        CREATE INDEX IF NOT EXISTS chunks_text_trgm_idx ON chunks
        USING gin (text gin_trgm_ops)
    """)

    op.execute(f"""
        CREATE TABLE IF NOT EXISTS seed_clauses (
            chunk_id         TEXT        PRIMARY KEY,
            source           VARCHAR(30) NOT NULL,
            doc_id           VARCHAR(200),
            text             TEXT        NOT NULL,
            domain           VARCHAR(20),
            risk_level       VARCHAR(10),
            risk_basis       TEXT,
            patterns_matched JSONB,
            metadata         JSONB,
            embedding        vector({EMBED_DIM})
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS seed_domain_idx ON seed_clauses (domain)")
    op.execute("CREATE INDEX IF NOT EXISTS seed_risk_idx ON seed_clauses (risk_level)")
    op.execute("""
        CREATE INDEX IF NOT EXISTS seed_embedding_idx ON seed_clauses
        USING hnsw (embedding vector_cosine_ops)
        WITH (m=16, ef_construction=64)
    """)

    op.execute(f"""
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
            embedding        vector({EMBED_DIM})
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS clean_domain_idx ON clean_clauses (domain)")
    op.execute("""
        CREATE INDEX IF NOT EXISTS clean_embedding_idx ON clean_clauses
        USING hnsw (embedding vector_cosine_ops)
        WITH (m=16, ef_construction=64)
    """)

    op.execute(f"""
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
            embedding        vector({EMBED_DIM})
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS noise_domain_idx ON noise_clauses (domain)")
    op.execute("CREATE INDEX IF NOT EXISTS noise_reason_idx ON noise_clauses (noise_reason)")
    op.execute("""
        CREATE INDEX IF NOT EXISTS noise_embedding_idx ON noise_clauses
        USING hnsw (embedding vector_cosine_ops)
        WITH (m=16, ef_construction=64)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS noise_clauses")
    op.execute("DROP TABLE IF EXISTS clean_clauses")
    op.execute("DROP TABLE IF EXISTS seed_clauses")
    op.execute("DROP TABLE IF EXISTS chunks")
    # vector/pg_trgm 확장은 다른 DB 객체가 의존하고 있을 수 있어 downgrade에서 지우지 않는다.
