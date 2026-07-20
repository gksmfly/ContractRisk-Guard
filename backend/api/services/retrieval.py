# backend/api/services/retrieval.py
"""
근거 법령·판례 검색 — chunks 테이블(법령 조문 + 판례) 대상 Hybrid(Dense+Sparse) Retrieval,
clean_clauses 테이블 대상 유사 라벨 조항 검색(Red-team Agent용)

analyze.py가 하드코딩된 LEGAL_BASIS 대신 실제 조항 텍스트로 관련 법조문·판례를
검색하도록 지원한다. backend.agents의 retrieval_strategy_agent/red_team_agent가
이 모듈을 호출하는 얇은 LangGraph 노드다 — 실제 검색 로직(DB 접근, 임베딩,
랭킹 융합)은 SRP상 여기(DB 서비스 레이어)에 남겨둔다.

법령(source='law')과 판례(source='precedent')를 각각 top_k씩 따로 검색한다 —
판례가 법령보다 10배 가까이 많아(30,154 vs 3,323 청크), 하나로 합쳐서
유사도 순 정렬하면 법령 인용이 거의 안 나올 수 있어 소스별로 결과를 보장한다.

Dense(의미 검색, KoE5 임베딩 + pgvector 코사인) 단독으로는 "제9조" 같은 정확한
조문 번호나 특정 법률 용어를 놓칠 수 있어, Sparse(어휘 검색, pg_trgm 문자
3-gram 유사도) 결과를 Reciprocal Rank Fusion(RRF)으로 합친다. 한국어 형태소
분석기(예: Mecab)는 시스템 바이너리 설치가 추가로 필요해 이번 단계에서는
쓰지 않고, 토크나이저 없이도 바로 쓸 수 있는 pg_trgm으로 Sparse 신호를 만든다.

Evidence Selection Agent 구축 전 사전 실험(clean_clauses 478건, 정답 조문
적중률 기준)에서 Cross-Encoder(BAAI/bge-reranker-v2-m3) 재랭킹이 RRF-only보다
오히려 낮은 결과(10.7% vs 20.1%)를 보여 채택하지 않았다 — 이 모듈은 RRF까지만
하고, 그 이상의 재랭킹(법원 심급 가중치 등)은 backend.agents.evidence_selection_agent
가 담당한다.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from backend.api.schemas import LegalBasis
from backend.db.loader import embed_texts, get_embedder
from backend.utils import load_logger

logger = load_logger("retrieval.log")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/crg")

# RRF 등 랭킹 융합 단계에서 후보를 넉넉히 확보하기 위한 Dense/Sparse 각각의 기본 후보 수.
# top_k(최종 반환 개수)보다 넉넉히 크게 잡아야 두 검색 결과가 겹치지 않을 때도
# 융합 순위가 의미를 가진다.
_CANDIDATE_K = 8

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


def candidate_to_legal_basis(candidate: dict) -> LegalBasis:
    """fetch_candidates()가 반환한 후보 dict 하나를 LegalBasis로 변환한다."""
    if candidate["source"] == "law":
        return _law_to_legal_basis(candidate["metadata"], candidate["text"])
    return _precedent_to_legal_basis(candidate["metadata"], candidate["text"])


def _search_dense(cur, sources: list[str], vec_literal: str, top_k: int) -> list[tuple[str, str, dict, str]]:
    cur.execute(
        """
        SELECT chunk_id, source, metadata, text
        FROM chunks
        WHERE source = ANY(%s)
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (sources, vec_literal, top_k),
    )
    return cur.fetchall()


def _search_sparse(
    cur, sources: list[str], query_text: str, top_k: int, similarity_threshold: float = 0.1,
) -> list[tuple[str, str, dict, str]]:
    """pg_trgm 문자 3-gram 유사도 기반 어휘(Sparse) 검색.

    기본 유사도 임계값(pg_trgm.similarity_threshold=0.3)은 조항 원문처럼 긴
    쿼리 대비 짧은 법조문 텍스트를 매칭하기엔 너무 엄격해서, 세션 단위로
    낮춰 재현율을 확보한다(정밀도는 이후 RRF 융합·Dense 신호가 보완).
    """
    cur.execute("SET pg_trgm.similarity_threshold = %s", (similarity_threshold,))
    cur.execute(
        """
        SELECT chunk_id, source, metadata, text
        FROM chunks
        WHERE source = ANY(%s) AND text %% %s
        ORDER BY similarity(text, %s) DESC
        LIMIT %s
        """,
        (sources, query_text, query_text, top_k),
    )
    return cur.fetchall()


def _reciprocal_rank_fusion(
    dense_rows: list[tuple[str, str, dict, str]],
    sparse_rows: list[tuple[str, str, dict, str]],
    k: int = 60,
) -> list[dict]:
    """Dense/Sparse 두 랭킹을 chunk_id 기준 RRF로 합쳐 점수 내림차순으로 정렬한다.

    RRF는 두 검색 방식의 점수 스케일(코사인 거리 vs 문자열 유사도)이 서로 달라
    직접 비교할 수 없을 때 널리 쓰이는 융합 방법이다 — 각 결과의 "순위"만 보고
    score = Σ 1/(k+rank)로 합산하므로 점수 정규화가 필요 없다. 어느 한쪽에만
    나온 항목도 그쪽 순위로 점수를 받아 후보에서 빠지지 않는다.

    반환하는 각 dict의 "in_both"는 Dense·Sparse 양쪽에서 다 나온 후보인지를
    나타낸다 — Evidence Verification Agent가 근거 신뢰도 신호로 쓴다(사전 실험에서
    "양쪽 다 동의한 후보"가 "한쪽만 찾은 후보"보다 정답 적중률이 뚜렷이 높았음:
    24.5% vs 14.9%, clean_clauses 478건 기준).
    """
    scores: dict[str, float] = {}
    payload: dict[str, dict] = {}
    dense_ids  = {row[0] for row in dense_rows}
    sparse_ids = {row[0] for row in sparse_rows}

    for rank, (chunk_id, source, metadata, text) in enumerate(dense_rows):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        payload[chunk_id] = {
            "chunk_id": chunk_id, "source": source, "metadata": metadata, "text": text,
            "in_both": chunk_id in sparse_ids,
        }
    for rank, (chunk_id, source, metadata, text) in enumerate(sparse_rows):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        payload.setdefault(chunk_id, {
            "chunk_id": chunk_id, "source": source, "metadata": metadata, "text": text,
            "in_both": chunk_id in dense_ids,
        })

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [payload[cid] for cid in ranked_ids]


def fetch_candidates(
    query_text: str,
    top_k_per_source: int = _CANDIDATE_K,
    sparse_similarity_threshold: float = 0.1,
    unified: bool = False,
) -> dict[str, list[dict]]:
    """법령·판례 검색 후보 풀을 소스별로 반환한다(Evidence Selection Agent가 재랭킹할 원본).

    각 후보 dict: {chunk_id, source, metadata, text, in_both}.
    unified=True면 law/precedent를 나누지 않고 한 번에 검색한 뒤 소스별로 재분류한다
    (Evidence Verification의 마지막 재검색 단계 — 소스를 나눠서 못 찾았으면 안 나눠서라도
    찾아본다는 전략).
    """
    if not query_text.strip():
        return {"law": [], "precedent": []}

    try:
        embedder = _get_cached_embedder()
        query_vec = embed_texts(embedder, [query_text], prefix="query: ")[0]
        vec_literal = "[" + ",".join(repr(x) for x in query_vec) + "]"

        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                if unified:
                    sources = ["law", "precedent"]
                    dense  = _search_dense(cur, sources, vec_literal, top_k_per_source)
                    sparse = _search_sparse(cur, sources, query_text, top_k_per_source, sparse_similarity_threshold)
                    fused = _reciprocal_rank_fusion(dense, sparse)
                    result = {"law": [], "precedent": []}
                    for c in fused:
                        result.setdefault(c["source"], []).append(c)
                else:
                    result = {}
                    for source in ("law", "precedent"):
                        dense  = _search_dense(cur, [source], vec_literal, top_k_per_source)
                        sparse = _search_sparse(cur, [source], query_text, top_k_per_source, sparse_similarity_threshold)
                        result[source] = _reciprocal_rank_fusion(dense, sparse)
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"법령/판례 후보 검색 실패, 빈 결과 반환: {e}")
        return {"law": [], "precedent": []}

    return result


def search_legal_basis(query_text: str, top_k: int = 2) -> list[LegalBasis]:
    """조항 텍스트(또는 근거 문구)로 관련 법령 조문·판례를 Hybrid 검색한다.

    소스별 RRF 융합 결과 상위 top_k를 그대로 반환한다(재랭킹 없음 — 재랭킹은
    backend.agents.evidence_selection_agent가 fetch_candidates()로 후보 풀을
    직접 받아서 처리한다). 이 함수는 하위호환·간단한 조회용으로 남겨둔다.
    """
    candidates = fetch_candidates(query_text, top_k_per_source=_CANDIDATE_K)
    law_top       = candidates.get("law", [])[:top_k]
    precedent_top = candidates.get("precedent", [])[:top_k]
    return [candidate_to_legal_basis(c) for c in law_top + precedent_top]


def search_similar_labeled_clauses(
    query_text: str, top_k: int = 5, exclude_chunk_id: str | None = None,
) -> list[dict]:
    """clean_clauses에서 의미상 가장 비슷한 검증 완료 조항을 찾는다(Red-team Agent용).

    Judgment Agent가 방금 내린 판단이 기존에 검증된 유사 사례들의 라벨과
    일관적인지 확인하는 데 쓴다. FB-Check로 검증된 데이터만 대상으로 하므로
    (clean_clauses, 478건) 비교 기준 자체의 신뢰도가 높다.
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
                cur.execute(
                    """
                    SELECT chunk_id, domain, risk_level,
                           COALESCE(NULLIF(evidence_span, ''), text) AS span,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM clean_clauses
                    WHERE chunk_id != %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vec_literal, exclude_chunk_id or "", vec_literal, top_k),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"유사 라벨 조항 검색 실패, 빈 결과 반환: {e}")
        return []

    return [
        {"chunk_id": chunk_id, "domain": domain, "risk_level": risk_level, "text": text, "similarity": float(sim)}
        for chunk_id, domain, risk_level, text, sim in rows
    ]
