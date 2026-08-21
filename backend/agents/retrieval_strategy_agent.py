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

첫 시도(retry_count=0)에서는 법령 검색을 약관규제법 파티션으로 고정한다. 법령
코퍼스가 43청크~1,305청크로 극단적으로 불균형해서, 필터 없이 전체를 경쟁시키면
소수 법령의 정답 조문이 밀려나기 때문이다.

**왜 LLM 라우팅(EXAONE)이 아니라 고정인가** — FTC 의결서 100건 기준 실측:

    RRF(필터 없음)              18/100
    EXAONE top-2 라우팅          37/100
    약관규제법 고정              65/100   ← EXAONE만 맞은 케이스 0건, McNemar p<0.00001
    약관규제법 + 민법 고정        23/100   ← 파티션을 하나만 더해도 42%p 폭락
    약관규제법 + EXAONE 예측 추가  40/100   ← 추가만 해도 해롭다(단독만 맞음 26 / 합집합만 0)

라우팅이 필요 없었다 — 평가 100건 **전부** 정답에 약관규제법이 들어 있는데
EXAONE은 23건에서 조항 소재(전자상거래·방문판매 등)에 끌려 다른 법을 골랐다.
게다가 민법(1,337조)처럼 큰 파티션이 섞이면 후보가 희석돼 약관규제법 조문이
top-K 밖으로 밀린다. **상수 기준선을 먼저 재지 않은 것이 원인이었다.**

한계: 이 평가셋은 전부 FTC 약관 사건이라 약관규제법이 100% 정답이다. "라우팅이
무용하다"는 일반화가 아니라 "이 도메인에서는 약관규제법이 항상 관련된다"는 뜻으로
읽어야 한다. 약관규제법이 정답이 아닌 사례를 포함한 평가셋이 생기면 재검토할 것.
재현: `backend/eval/law_router_compare.py`

재시도(1~3회차)는 필터 없이 검색한다 — 재시도의 "범위를 넓힌다"는 취지와
좁히는 필터가 상충하기 때문.
"""

from backend.agents.state import ClauseState
from backend.api.services.retrieval import fetch_candidates

_BASE_TOP_K = 6

# 1회차 법령 검색을 이 파티션·이 조문 구간으로 고정한다(위 docstring의 실측 근거 참고).
_PRIMARY_LAW = "약관의 규제에 관한 법률"

# 약관규제법 46청크 중 "이 조항이 불공정한가"를 정하는 실질 규범은 제6~14조 9개뿐이고,
# 나머지 37개는 심사청구·분쟁조정·협의회 구성·과태료 같은 절차 조문이다. 필터가 없으면
# 그 절차 조문이 상위를 차지한다(전속관할 조항 질의에 제19조 약관의 심사청구, 제27조
# 분쟁조정의 신청, 제30조 독점규제법 준용이 2~4위로 올라왔다). top-5 66% → 81%.
_SUBSTANTIVE_ARTICLES = (6, 14)


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

    law_names = [_PRIMARY_LAW] if retry_count == 0 else None

    article_range = _SUBSTANTIVE_ARTICLES if law_names else None
    candidates = fetch_candidates(
        query, law_names=law_names, law_article_range=article_range, **params
    )
    return {"retrieval_candidates": candidates}
