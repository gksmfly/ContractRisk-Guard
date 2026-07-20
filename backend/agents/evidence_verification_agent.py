# backend/agents/evidence_verification_agent.py
"""Evidence Verification Agent — 선택된 근거의 신뢰도를 확인하고, 부족하면 재검색을 지시한다.

사전 실험에서 "evidence_span과 legal_basis 사이 코사인 유사도"는 정답 조문
적중 여부와 거의 상관관계가 없었다(hit 평균 0.567 vs miss 평균 0.559, clean_clauses
478건 기준) — 그래서 정합성 신호로 쓰지 않는다. 대신 "최종 선택된 근거가
Dense·Sparse 검색 양쪽에서 다 나왔는가(evidence_agreement)"를 쓴다 — 같은
데이터로 측정했을 때 양쪽 다 동의한 후보의 적중률(24.5%)이 한쪽만 찾은
후보(14.9%)보다 뚜렷이 높았다. 추가 임베딩 계산이 필요 없어 코사인 방식보다 가볍다.

MAX_RETRIES(=3)까지 retrieval_strategy_agent로 되돌아간다 — 매회 다른 전략으로
검색 범위를 넓히므로(retrieval_strategy_agent.py 참고) 무한루프가 아니다.
"""

from backend.agents.state import ClauseState

MAX_RETRIES = 3


def evidence_verification_node(state: ClauseState) -> dict:
    legal_basis = state.get("legal_basis") or []
    agreement   = state.get("evidence_agreement", False)
    retry_count = state.get("retry_count", 0)

    sufficient = bool(legal_basis) and agreement
    if sufficient or retry_count >= MAX_RETRIES:
        return {"evidence_verified": sufficient, "should_retry": False}

    return {"retry_count": retry_count + 1, "evidence_verified": False, "should_retry": True}
