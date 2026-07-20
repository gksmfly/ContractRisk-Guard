# backend/agents/evidence_selection_agent.py
"""Evidence Selection Agent — Retrieval Strategy가 넘긴 후보 풀을 재랭킹해 최종 근거(top-2)를 고른다.

사전 실험(clean_clauses 478건, 정답 조문 적중률 기준)에서 Cross-Encoder
(BAAI/bge-reranker-v2-m3) 재랭킹이 RRF-only보다 오히려 낮은 결과(10.7% vs 20.1%)를
보여 채택하지 않았다. 대신 RRF 융합 순위를 그대로 쓰고, 판례에 한해서만 법원
심급 가중치를 소폭 가산해 재정렬한다 — 대법원 판례가 더 권위 있는 근거라는
도메인 지식 반영. 법령은 재정렬 없이 RRF 순서 그대로 채택한다.
"""

from backend.agents.state import ClauseState
from backend.api.schemas import LegalBasis
from backend.api.services.retrieval import candidate_to_legal_basis

_COURT_WEIGHT = {
    "대법원": 0.10,
    "고등법원": 0.05,
}
_FINAL_K = 2

# 후보가 하나도 없을 때(DB 미가동·미적재 등)만 쓰는 최종 fallback.
LEGAL_BASIS_FALLBACK: dict[str, list[dict]] = {
    "해지_조항": [
        {"law": "약관규제법", "article": "제9조",      "description": "고객에게 부당하게 불리한 해제·해지권 부여 조항 무효"},
        {"law": "민법",      "article": "제543~553조", "description": "계약 해제·해지의 일반 요건 및 효과"},
    ],
    "책임제한_조항": [
        {"law": "약관규제법", "article": "제7조",      "description": "사업자 책임 부당 면제·제한 조항 무효"},
        {"law": "민법",      "article": "제750~766조", "description": "불법행위 손해배상 책임"},
    ],
    "해당없음": [],
}


_RRF_K = 60  # backend.api.services.retrieval._reciprocal_rank_fusion과 동일한 k — 점수 스케일을 맞춰야 가중치가 의미를 가짐


def _rerank_precedents(candidates: list[dict]) -> list[dict]:
    """RRF 순위를 점수로 환산한 뒤 법원 심급 가중치를 더해 재정렬한다.

    RRF 원점수(k=60)는 상위권끼리도 차이가 아주 작다(예: 1위 0.0164 vs 2위 0.0161,
    차이 0.0003) — 여기서도 같은 k로 점수를 매겨야 법원 가중치(0.05~0.10)가
    "동률에 가까운 상위권 후보들 사이의 소폭 가산"으로 실제로 작동한다. k=0
    기준(1/(rank+1))처럼 상위권 격차가 큰 척도로 계산하면 가중치가 사실상
    무력화된다.
    """
    def score(indexed_candidate: tuple[int, dict]) -> float:
        rank, candidate = indexed_candidate
        base  = 1.0 / (_RRF_K + rank + 1)
        boost = _COURT_WEIGHT.get((candidate["metadata"] or {}).get("court", ""), 0.0)
        return base + boost

    ranked = sorted(enumerate(candidates), key=score, reverse=True)
    return [candidate for _, candidate in ranked]


def evidence_selection_node(state: ClauseState) -> dict:
    candidates = state.get("retrieval_candidates", {}) or {}
    law_top       = candidates.get("law", [])[:_FINAL_K]
    precedent_top = _rerank_precedents(candidates.get("precedent", []))[:_FINAL_K]
    selected = law_top + precedent_top

    if not selected:
        fallback = LEGAL_BASIS_FALLBACK.get(state.get("domain", "해당없음"), [])
        return {"legal_basis": [LegalBasis(**b) for b in fallback], "evidence_agreement": False}

    legal_basis = [candidate_to_legal_basis(c) for c in selected]
    evidence_agreement = any(c.get("in_both") for c in selected)
    return {"legal_basis": legal_basis, "evidence_agreement": evidence_agreement}
