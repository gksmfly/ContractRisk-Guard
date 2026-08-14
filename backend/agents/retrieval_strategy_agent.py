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

첫 시도(retry_count=0)에서는 검색 전에 query_router.route_law_names()로 이 조항이
어느 법령에 해당할지 로컬 LLM(EXAONE)이 먼저 판단해 법령 검색 범위를 좁힌다 —
법령 코퍼스가 43청크~1,305청크로 극단적으로 불균형해서 필터 없이 전체를 경쟁시키면
소수 법령의 정답 조문이 밀려난다(`backend/eval/retrieval_alternatives_survey.md`
실측: RRF 8%→33%, McNemar p<0.0001). 재시도(1~3회차)는 라우팅을 다시 쓰지 않고
필터 없이 검색한다 — 1회차 라우팅이 틀렸을 가능성이 있는 상황이라, 재시도의
"범위를 넓힌다"는 원래 취지와 좁히는 필터가 상충하기 때문. 라우팅 실패(모델 오류·
JSON 파싱 실패 등)는 route_law_names()가 None을 반환해 필터 없이 검색되므로
검색 자체가 막히지 않는다.
"""

from backend.agents.query_router import route_law_names
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
    retry_count = state.get("retry_count", 0)
    params = _search_params(retry_count)
    use_full_clause = params.pop("use_full_clause")
    query = state["clause"] if use_full_clause else (state.get("evidence_span") or state["clause"])

    law_names = route_law_names(state["clause"]) if retry_count == 0 else None

    candidates = fetch_candidates(query, law_names=law_names, **params)
    return {"retrieval_candidates": candidates}
