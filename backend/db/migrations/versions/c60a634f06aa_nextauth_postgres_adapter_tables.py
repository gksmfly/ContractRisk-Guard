"""nextauth postgres adapter tables

프론트엔드(Next.js) Google 로그인용 — @auth/pg-adapter(next-auth 공식 Postgres
어댑터)가 요구하는 스키마다. 컬럼명(userId, providerAccountId 등)이 camelCase인
건 우리 컨벤션이 아니라 어댑터가 고정으로 요구하는 이름이라 그대로 따랐다
(frontend/node_modules/@auth/pg-adapter/src/index.ts의 쿼리문 기준으로 정확히
맞춤 — 임의로 지어낸 스키마가 아님).

이 테이블들은 backend(Python)가 아니라 frontend(Node, pg 패키지)가 직접 읽고 쓴다
— 같은 Postgres를 두 런타임이 각자의 클라이언트로 나눠 쓰는 구조.

Revision ID: c60a634f06aa
Revises: 91afb62fa923
Create Date: 2026-07-29 01:42:03.707970

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "c60a634f06aa"
down_revision = "91afb62fa923"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id             SERIAL PRIMARY KEY,
            name           VARCHAR(255),
            email          VARCHAR(255),
            "emailVerified" TIMESTAMPTZ,
            image          TEXT
        )
    """)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS users_email_idx ON users (email)
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            id                  SERIAL PRIMARY KEY,
            "userId"            INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            type                VARCHAR(255) NOT NULL,
            provider            VARCHAR(255) NOT NULL,
            "providerAccountId" VARCHAR(255) NOT NULL,
            refresh_token       TEXT,
            access_token        TEXT,
            expires_at          BIGINT,
            id_token            TEXT,
            scope               TEXT,
            session_state       TEXT,
            token_type          TEXT
        )
    """)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS accounts_provider_provider_account_id_idx
        ON accounts (provider, "providerAccountId")
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id             SERIAL PRIMARY KEY,
            "userId"       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires        TIMESTAMPTZ NOT NULL,
            "sessionToken" VARCHAR(255) NOT NULL
        )
    """)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS sessions_session_token_idx
        ON sessions ("sessionToken")
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS verification_token (
            identifier TEXT NOT NULL,
            expires    TIMESTAMPTZ NOT NULL,
            token      TEXT NOT NULL,
            PRIMARY KEY (identifier, token)
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS verification_token")
    op.execute("DROP TABLE IF EXISTS sessions")
    op.execute("DROP TABLE IF EXISTS accounts")
    op.execute("DROP TABLE IF EXISTS users")
