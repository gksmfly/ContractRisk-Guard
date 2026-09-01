# tests/agents/test_retrieval_strategy_agent.py
"""
backend/api/services/retrieval.py의 _fuse_reciprocal_rank() 단위 테스트 +
backend/agents/retrieval_strategy_agent.py의 법령 라우팅 통합(1회차만 라우팅,
재시도는 필터 해제) 단위 테스트.

Dense/Sparse 검색 자체는 DB가 있어야 하지만, 두 랭킹을 합치는 RRF 융합 로직은
순수 함수라 DB 없이도 검증할 수 있다. 라우팅 통합 테스트는 route_law_names/
fetch_candidates를 monkeypatch로 대체해 GPU·DB 없이 노드의 분기 로직만 검증한다.

실행: pytest tests/test_retrieval_strategy_agent.py
"""

from typing import Any

import pytest

import backend.agents.retrieval_strategy_agent as retrieval_strategy_agent
from backend.api.services.retrieval import _fuse_reciprocal_rank


def _row(chunk_id: str, source: str = "law") -> tuple[str, str, dict, str]:
    return (chunk_id, source, {"law_name": chunk_id}, f"text-{chunk_id}")


class TestReciprocalRankFusion:
    def test_both_empty_returns_empty_list(self) -> None:
        assert _fuse_reciprocal_rank([], []) == []

    def test_dense_only_preserves_dense_order(self) -> None:
        dense = [_row("a"), _row("b"), _row("c")]
        fused = _fuse_reciprocal_rank(dense, [])
        assert [c["chunk_id"] for c in fused] == ["a", "b", "c"]

    def test_sparse_only_preserves_sparse_order(self) -> None:
        sparse = [_row("x"), _row("y")]
        fused = _fuse_reciprocal_rank([], sparse)
        assert [c["chunk_id"] for c in fused] == ["x", "y"]

    def test_item_ranked_first_in_both_lists_wins(self) -> None:
        dense  = [_row("a"), _row("b"), _row("c")]
        sparse = [_row("a"), _row("c"), _row("b")]
        fused = _fuse_reciprocal_rank(dense, sparse)
        assert fused[0]["chunk_id"] == "a"

    def test_item_only_in_one_list_still_included(self) -> None:
        dense  = [_row("a"), _row("b")]
        sparse = [_row("c")]
        fused = _fuse_reciprocal_rank(dense, sparse)
        ids = {c["chunk_id"] for c in fused}
        assert ids == {"a", "b", "c"}

    def test_duplicate_across_lists_not_duplicated_in_output(self) -> None:
        dense  = [_row("a"), _row("b")]
        sparse = [_row("b"), _row("a")]
        fused = _fuse_reciprocal_rank(dense, sparse)
        ids = [c["chunk_id"] for c in fused]
        assert len(ids) == len(set(ids)) == 2

    def test_appearing_in_both_ranks_above_appearing_in_one(self) -> None:
        dense  = [_row("only_dense"), _row("both")]
        sparse = [_row("only_sparse"), _row("both")]
        fused = _fuse_reciprocal_rank(dense, sparse)
        ranked_ids = [c["chunk_id"] for c in fused]
        assert ranked_ids.index("both") < ranked_ids.index("only_dense")
        assert ranked_ids.index("both") < ranked_ids.index("only_sparse")

    def test_in_both_flag_set_correctly(self) -> None:
        dense  = [_row("both"), _row("only_dense")]
        sparse = [_row("both"), _row("only_sparse")]
        fused = _fuse_reciprocal_rank(dense, sparse)
        by_id = {c["chunk_id"]: c["in_both"] for c in fused}
        assert by_id["both"] is True
        assert by_id["only_dense"] is False
        assert by_id["only_sparse"] is False

    def test_source_field_preserved(self) -> None:
        dense = [_row("a", source="precedent")]
        fused = _fuse_reciprocal_rank(dense, [])
        assert fused[0]["source"] == "precedent"


class TestRetrievalStrategyNodeRouting:
    """1회차는 약관규제법 파티션으로 고정 검색, 재시도는 필터 없이 검색한다.

    예전에는 로컬 LLM(EXAONE)이 top-2 법령을 예측해 그 파티션만 검색했다. FTC 의결서
    100건 실측에서 그 라우팅이 고정 검색에 완패해(37 vs 65, EXAONE만 맞은 케이스 0건,
    McNemar p<0.00001) 제거했다 — 평가 100건 전부 정답에 약관규제법이 들어 있어
    라우팅할 대상 자체가 없었고, 민법처럼 큰 파티션이 섞이면 후보가 희석돼 오히려
    떨어졌다(약관규제법+민법 23%).

    이 테스트는 라우터를 되살리는 변경에 대한 회귀 방지다.
    """

    def _patch(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        calls = {"fetch": []}

        def fake_fetch(query: str, law_names: list[str] | None = None, **kwargs: Any) -> dict:
            calls["fetch"].append({"query": query, "law_names": law_names, **kwargs})
            return {"law": [], "precedent": []}

        monkeypatch.setattr(retrieval_strategy_agent, "fetch_candidates", fake_fetch)
        return calls

    def test_first_attempt_pins_primary_law(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._patch(monkeypatch)
        state = {"clause": "제9조 계약 해지 조항", "evidence_span": "해지 조항", "retry_count": 0}

        retrieval_strategy_agent.retrieval_strategy_node(state)

        assert calls["fetch"][0]["law_names"] == [retrieval_strategy_agent._PRIMARY_LAW]
        assert retrieval_strategy_agent._PRIMARY_LAW == "약관의 규제에 관한 법률"

    def test_first_attempt_uses_single_partition(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """파티션을 하나만 더 붙여도 성능이 42%p 떨어졌다 — 반드시 1개여야 한다."""
        calls = self._patch(monkeypatch)
        state = {"clause": "제9조 계약 해지 조항", "evidence_span": "해지 조항", "retry_count": 0}

        retrieval_strategy_agent.retrieval_strategy_node(state)

        assert len(calls["fetch"][0]["law_names"]) == 1

    def test_retry_passes_no_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._patch(monkeypatch)
        state = {"clause": "제9조 계약 해지 조항", "evidence_span": "해지 조항", "retry_count": 1}

        result = retrieval_strategy_agent.retrieval_strategy_node(state)

        assert calls["fetch"][0]["law_names"] is None
        assert result == {"retrieval_candidates": {"law": [], "precedent": []}}

    def test_no_llm_router_is_called(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """모듈이 EXAONE 라우터를 다시 import·호출하지 않는지 확인하는 트립와이어."""
        import backend.agents.query_router as qr

        called = []
        monkeypatch.setattr(qr, "route_law_names", lambda *a, **k: called.append(a) or [])
        self._patch(monkeypatch)
        retrieval_strategy_agent.retrieval_strategy_node(
            {"clause": "제9조 계약 해지 조항", "evidence_span": "해지 조항", "retry_count": 0}
        )
        assert called == []
