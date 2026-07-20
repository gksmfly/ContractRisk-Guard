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
    domain: str
    evidence_span: str
    reasoning: str
    retrieval_candidates: dict[str, list[dict]]  # 중간 산물 — Evidence Selection이 소비, 최종 응답엔 안 씀
    legal_basis: list[LegalBasis]
    evidence_agreement: bool   # 최종 legal_basis 중 Dense·Sparse 양쪽에서 다 나온 게 있는지(중간 산물)
    risk_level: str
    verified: bool
    redteam_note: str
    evidence_verified: bool
    retry_count: int
    should_retry: bool     # 그래프 라우팅 전용 신호(중간 산물) — evidence_verification_agent가 씀
