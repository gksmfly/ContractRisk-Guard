# backend/api/services/retrieval.py
"""
근거 법령·판례 검색 — chunks 테이블(법령 조문 + 판례) 대상 Dense Retrieval

analyze.py가 하드코딩된 LEGAL_BASIS 대신 실제 조항 텍스트로 관련 법조문·판례를
의미 검색하도록 지원한다. backend/db/loader.py와 동일한 KoE5 임베딩 모델을
재사용해, 적재 시점과 검색 시점의 임베딩이 어긋나지 않도록 한다.

법령(source='law')과 판례(source='precedent')를 각각 top_k씩 따로 검색한다 —
판례가 법령보다 10배 가까이 많아(30,154 vs 3,323 청크), 하나로 합쳐서
유사도 순 정렬하면 법령 인용이 거의 안 나올 수 있어 소스별로 결과를 보장한다.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from backend.api.schemas import LegalBasis
from backend.db.loader import embed_texts, get_embedder
from backend.utils import load_logger

logger = load_logger("retrieval.log")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/crg")

_embedder = None


def _get_cached_embedder():
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
    return _embedder


def _get_conn():
    import psycopg2
    return psycopg2.connect(DATABASE_URL)


def _truncate(text: str, max_len: int = 100) -> str:
    text = text.strip()
    return text[:max_len] + "..." if len(text) > max_len else text


def _law_to_legal_basis(metadata: dict, text: str) -> LegalBasis:
    metadata = metadata or {}
    article_no    = metadata.get("article_no", "")
    article_title = metadata.get("article_title", "")
    article = f"제{article_no}조" if article_no else ""
    if article_title:
        article = f"{article}({article_title})" if article else article_title
    return LegalBasis(law=metadata.get("law_name", ""), article=article, description=_truncate(text))


def _precedent_to_legal_basis(metadata: dict, text: str) -> LegalBasis:
    metadata = metadata or {}
    court     = metadata.get("court", "")
    case_name = metadata.get("case_name", "")
    law = f"{court} 판례" if court else "판례"
    if case_name:
        law = f"{law}({case_name})"
    return LegalBasis(law=law, article=metadata.get("case_number", ""), description=_truncate(text))


def _search_source(cur, source: str, vec_literal: str, top_k: int) -> list[tuple[dict, str]]:
    cur.execute(
        """
        SELECT metadata, text
        FROM chunks
        WHERE source = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (source, vec_literal, top_k),
    )
    return cur.fetchall()


def search_legal_basis(query_text: str, top_k: int = 2) -> list[LegalBasis]:
    """조항 텍스트(또는 근거 문구)로 관련 법령 조문·판례를 의미 검색한다.

    법령 top_k건 + 판례 top_k건을 각각 검색해 합쳐 반환한다 (법령이 판례 대비
    청크 수가 훨씬 적어, 하나로 합쳐 유사도순 정렬하면 법령이 밀려날 수 있음).
    DB 연결 실패나 임베딩 차원 불일치(예: DB가 아직 재적재되지 않은 경우) 등
    검색이 불가능한 상황에서는 빈 리스트를 반환한다 — 호출자가 정적 fallback을
    적용할 수 있게 한다.
    """
    if not query_text.strip():
        return []

    try:
        embedder = _get_cached_embedder()
        query_vec = embed_texts(embedder, [query_text], prefix="query: ")[0]
        vec_literal = "[" + ",".join(repr(x) for x in query_vec) + "]"

        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                law_rows       = _search_source(cur, "law", vec_literal, top_k)
                precedent_rows = _search_source(cur, "precedent", vec_literal, top_k)
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"법령/판례 검색 실패, 빈 결과 반환: {e}")
        return []

    return (
        [_law_to_legal_basis(metadata, text) for metadata, text in law_rows]
        + [_precedent_to_legal_basis(metadata, text) for metadata, text in precedent_rows]
    )
