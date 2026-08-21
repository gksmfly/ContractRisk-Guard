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
    legal_basis: list[LegalBasis]
    evidence_agreement: bool   # 최종 legal_basis 중 Dense·Sparse 양쪽에서 다 나온 게 있는지(중간 산물)
    risk_level: str
    verified: bool
    confidence_band: str            # "높음"/"중간"/"낮음" — 원시 확률은 보정이 안 돼 있어 구간으로만 노출
    confidence_band_accuracy: float # 그 구간의 실측 정확도(v4, n=453) — 화면에 함께 표시용
    redteam_note: str
    evidence_verified: bool
    retry_count: int
    should_retry: bool     # 그래프 라우팅 전용 신호(중간 산물) — evidence_verification_agent가 씀
