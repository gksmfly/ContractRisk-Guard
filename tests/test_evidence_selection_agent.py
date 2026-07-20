# tests/test_evidence_selection_agent.py
"""
backend/agents/evidence_selection_agent.py의 순수 로직(법원 심급 가중치 재정렬,
후보 없을 때 fallback) 단위 테스트. DB·CrossEncoder 불필요.

실행: pytest tests/test_evidence_selection_agent.py
"""

from backend.agents.evidence_selection_agent import (
    _rerank_precedents, evidence_selection_node, LEGAL_BASIS_FALLBACK,
)


def _candidate(chunk_id: str, court: str = "", in_both: bool = False) -> dict:
    return {
        "chunk_id": chunk_id, "source": "precedent",
        "metadata": {"court": court, "case_name": chunk_id, "case_number": chunk_id},
        "text": f"text-{chunk_id}", "in_both": in_both,
    }


class TestRerankPrecedents:
    def test_supreme_court_ranks_above_lower_court(self):
        candidates = [_candidate("low", court="지방법원"), _candidate("supreme", court="대법원")]
        ranked = _rerank_precedents(candidates)
        assert ranked[0]["chunk_id"] == "supreme"

    def test_supreme_court_beats_higher_original_rrf_rank(self):
        # RRF 순위상 "low"가 1위였어도 대법원 판례("supreme")가 가중치로 역전해야 한다
        candidates = [_candidate("low", court="지방법원"), _candidate("mid", court="고등법원"), _candidate("supreme", court="대법원")]
        ranked = _rerank_precedents(candidates)
        assert ranked[0]["chunk_id"] == "supreme"
        assert ranked.index(next(c for c in ranked if c["chunk_id"] == "mid")) < ranked.index(next(c for c in ranked if c["chunk_id"] == "low"))

    def test_no_court_metadata_keeps_rrf_order(self):
        candidates = [_candidate("a"), _candidate("b")]
        ranked = _rerank_precedents(candidates)
        assert [c["chunk_id"] for c in ranked] == ["a", "b"]

    def test_empty_list_returns_empty(self):
        assert _rerank_precedents([]) == []


class TestEvidenceSelectionNode:
    def test_no_candidates_uses_fallback(self):
        state = {"domain": "해지_조항", "retrieval_candidates": {"law": [], "precedent": []}}
        result = evidence_selection_node(state)
        assert len(result["legal_basis"]) == len(LEGAL_BASIS_FALLBACK["해지_조항"])
        assert result["evidence_agreement"] is False

    def test_missing_retrieval_candidates_key_uses_fallback(self):
        state = {"domain": "책임제한_조항"}
        result = evidence_selection_node(state)
        assert len(result["legal_basis"]) == len(LEGAL_BASIS_FALLBACK["책임제한_조항"])

    def test_evidence_agreement_true_when_any_selected_in_both(self):
        state = {
            "domain": "해지_조항",
            "retrieval_candidates": {
                "law": [{"chunk_id": "l1", "source": "law", "metadata": {"law_name": "민법", "article_no": "543"}, "text": "t", "in_both": True}],
                "precedent": [],
            },
        }
        result = evidence_selection_node(state)
        assert result["evidence_agreement"] is True

    def test_evidence_agreement_false_when_none_in_both(self):
        state = {
            "domain": "해지_조항",
            "retrieval_candidates": {
                "law": [{"chunk_id": "l1", "source": "law", "metadata": {"law_name": "민법", "article_no": "543"}, "text": "t", "in_both": False}],
                "precedent": [],
            },
        }
        result = evidence_selection_node(state)
        assert result["evidence_agreement"] is False
