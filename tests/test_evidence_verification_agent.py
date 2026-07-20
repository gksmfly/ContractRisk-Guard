# tests/test_evidence_verification_agent.py
"""
backend/agents/evidence_verification_agent.py의 재검색 트리거 로직 단위 테스트.
순수 함수라 DB·GPU 불필요.

실행: pytest tests/test_evidence_verification_agent.py
"""

from backend.agents.evidence_verification_agent import evidence_verification_node, MAX_RETRIES


class TestEvidenceVerificationNode:
    def test_sufficient_evidence_stops_without_retry(self):
        state = {"legal_basis": [object()], "evidence_agreement": True, "retry_count": 0}
        result = evidence_verification_node(state)
        assert result["evidence_verified"] is True
        assert result["should_retry"] is False

    def test_empty_legal_basis_triggers_retry(self):
        state = {"legal_basis": [], "evidence_agreement": False, "retry_count": 0}
        result = evidence_verification_node(state)
        assert result["should_retry"] is True
        assert result["retry_count"] == 1

    def test_no_agreement_triggers_retry(self):
        state = {"legal_basis": [object()], "evidence_agreement": False, "retry_count": 0}
        result = evidence_verification_node(state)
        assert result["should_retry"] is True
        assert result["retry_count"] == 1

    def test_retry_count_increments_each_round(self):
        state = {"legal_basis": [], "evidence_agreement": False, "retry_count": 2}
        result = evidence_verification_node(state)
        assert result["retry_count"] == 3
        assert result["should_retry"] is True

    def test_stops_after_max_retries_even_if_insufficient(self):
        state = {"legal_basis": [], "evidence_agreement": False, "retry_count": MAX_RETRIES}
        result = evidence_verification_node(state)
        assert result["should_retry"] is False
        assert result["evidence_verified"] is False
        assert "retry_count" not in result  # 소진 후엔 더 안 올림

    def test_full_retry_cycle_reaches_max_then_stops(self):
        retry_count = 0
        for _ in range(MAX_RETRIES):
            result = evidence_verification_node({"legal_basis": [], "evidence_agreement": False, "retry_count": retry_count})
            assert result["should_retry"] is True
            retry_count = result["retry_count"]
        assert retry_count == MAX_RETRIES

        final = evidence_verification_node({"legal_basis": [], "evidence_agreement": False, "retry_count": retry_count})
        assert final["should_retry"] is False
