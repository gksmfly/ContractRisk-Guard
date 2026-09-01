# backend/agents/state.py
"""LangGraph 파이프라인이 조항 하나를 처리하며 공유하는 상태.

Analysis → Retrieval Strategy → Judgment 순서로 노드를 지나며 필드가 하나씩
채워진다. Evidence Selection/Red-team/Evidence Verification 에이전트가
추가되면 그때 필요한 필드를 여기 보강한다.
"""

from typing import TypedDict

from backend.api.schemas import LegalBasis


class ClauseState(TypedDict, total=False):
    clause: str            # 그래프 시작 시 필수로 채워짐, 나머지는 각 노드가 순서대로 채움
    articles: list[str]    # 약관규제법 위반 소지 유형("제9조" 등, 복수) — Analysis가 채우는 1차 라벨
    domain: str            # articles에서 파생된 옛 2-도메인 값 — judgment_agent의 verified 비교용
    evidence_span: str
    reasoning: str
    retrieval_candidates: dict[str, list[dict]]  # 중간 산물 — Evidence Selection이 소비, 최종 응답엔 안 씀
    legal_basis: list[LegalBasis]      # 예측한 조에서 매핑한 약관규제법 조문 — 근거
    precedent_refs: list[LegalBasis]   # 검색으로 찾은 유사 판례 — **참고**(hit@5 14%)
    evidence_agreement: bool   # 최종 legal_basis 중 Dense·Sparse 양쪽에서 다 나온 게 있는지(중간 산물)
    # judgment_agent(KoELECTRA)가 낸 조 목록. **`articles`(GPT 1차 라벨)와 별도 필드다** —
    # 같은 이름을 쓰면 judgment 노드가 GPT 값을 덮어써서 `verified` 비교가 자기 자신과의
    # 비교가 된다.
    model_articles: list[str]
    needs_review: bool     # 모델이 조를 하나라도 지목했는가 = 사용자에게 보여줄 것인가
    verified: bool         # GPT와 모델이 같은 조를 짚었는가. 신뢰도가 아니라 **일치 여부**다
    redteam_note: str
    evidence_verified: bool
    retry_count: int
    should_retry: bool     # 그래프 라우팅 전용 신호(중간 산물) — evidence_verification_agent가 씀
