# backend/db/migrations/versions/3d077f342ac4_analyses_table_for_saved_results.py
"""analyses table for saved results

로그인한 사용자가 "저장하기"를 누른 분석 결과를 담는 개인 히스토리 테이블.
c60a634f06aa와 마찬가지로 frontend(Node, pg 패키지)가 직접 읽고 쓴다 — 저장 시점의
FullAnalyzeResult(위험도·법령 인용 등 전부 포함)를 그대로 JSONB로 넣어서, 다시 열 때
백엔드 분석을 재실행하지 않고 그때 본 결과를 그대로 재현한다.

Revision ID: 3d077f342ac4
Revises: c60a634f06aa
Create Date: 2026-08-16 06:48:50.193758

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "3d077f342ac4"
down_revision = "c60a634f06aa"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id          SERIAL PRIMARY KEY,
            user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title       VARCHAR(255) NOT NULL,
            result      JSONB NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS analyses_user_id_created_at_idx
        ON analyses (user_id, created_at DESC)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS analyses")
