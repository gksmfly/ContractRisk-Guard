# backend/agents/retrieval_strategy_agent.py
"""Retrieval Strategy Agent — 근거 문구로 관련 법령·판례 후보를 Hybrid(Dense+Sparse) 검색한다.

실제 검색 로직(DB 접근, KoE5 임베딩, pg_trgm Sparse 검색, RRF 융합)은
backend.api.services.retrieval에 있다 — 이 노드는 그래프 상태를 읽어 후보 풀을
조회하고 상태에 써 넣는 역할만 한다. 최종 선택(재랭킹)은 Evidence Selection Agent가
한다.

Evidence Verification Agent가 "재검색이 필요하다"고 판단해 되돌아올 때마다
(state["retry_count"] > 0), 이전과 같은 쿼리로 다시 검색해봐야 결과가 똑같으므로
매회 다른 전략으로 검색 범위를 넓힌다:
  1회차(retry_count=1): evidence_span 대신 조항 전체로 쿼리 확대, top_k 6→10
  2회차(retry_count=2): top_k 10→16, pg_trgm 임계값 0.1→0.05로 낮춰 Sparse 재현율 확보
  3회차(retry_count=3): law/precedent 구분 없이 통합 검색, top_k 24(마지막 시도라 재현율 최우선)
"""

from backend.agents.state import ClauseState
from backend.api.services.retrieval import fetch_candidates

_BASE_TOP_K = 6


def _search_params(retry_count: int) -> dict:
    if retry_count == 0:
        return {"top_k_per_source": _BASE_TOP_K, "sparse_similarity_threshold": 0.10, "unified": False, "use_full_clause": False}
    if retry_count == 1:
        return {"top_k_per_source": 10, "sparse_similarity_threshold": 0.10, "unified": False, "use_full_clause": True}
    if retry_count == 2:
        return {"top_k_per_source": 16, "sparse_similarity_threshold": 0.05, "unified": False, "use_full_clause": True}
    return {"top_k_per_source": 24, "sparse_similarity_threshold": 0.05, "unified": True, "use_full_clause": True}


def retrieval_strategy_node(state: ClauseState) -> dict:
    params = _search_params(state.get("retry_count", 0))
    use_full_clause = params.pop("use_full_clause")
    query = state["clause"] if use_full_clause else (state.get("evidence_span") or state["clause"])

    candidates = fetch_candidates(query, **params)
    return {"retrieval_candidates": candidates}
