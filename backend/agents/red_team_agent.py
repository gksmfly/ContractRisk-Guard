# backend/agents/red_team_agent.py
"""Red-team Agent — 검증된 유사 사례와 비교해 Judgment의 판단 편향 가능성을 probe한다.

LLM을 호출하지 않는다 — clean_clauses(FB-Check로 검증된 478건)에서 임베딩
최근접 이웃을 찾아, 지금 조항과 아주 비슷한데(코사인 유사도 0.75 이상) risk_level이
다른 사례가 있으면 "이 판단, 비슷한데 반대로 판정된 사례가 있다"는 신호를 남긴다.

임계값 0.75는 clean_clauses leave-one-out 실험으로 검증했다: 탐지율 2.3%(11/478)이고,
표본을 직접 확인한 결과 "즉시 해지(High) vs 그냥 해지(Low)"처럼 실제로 의미 있는
차이를 잡아내는 것으로 확인됨(오탐이 아님).
"""

from backend.agents.state import ClauseState
from backend.api.services.retrieval import search_similar_labeled_clauses

_SIMILARITY_THRESHOLD = 0.75
_TOP_K = 5


def _build_note(neighbor: dict) -> str:
    return (
        f"유사도 {neighbor['similarity']:.2f}로 매우 비슷한 조항이 '{neighbor['risk_level']}'로 "
        f"판정된 사례가 있습니다: \"{neighbor['text'][:80]}\""
    )


def red_team_node(state: ClauseState) -> dict:
    query = state.get("evidence_span") or state["clause"]
    risk_level = state.get("risk_level")

    neighbors = search_similar_labeled_clauses(query, top_k=_TOP_K)
    for neighbor in neighbors:
        if neighbor["similarity"] >= _SIMILARITY_THRESHOLD and neighbor["risk_level"] != risk_level:
            return {"redteam_note": _build_note(neighbor)}

    return {"redteam_note": ""}
