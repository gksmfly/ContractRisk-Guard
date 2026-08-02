# tests/test_retrieval_strategy_agent.py
"""
backend/api/services/retrieval.py의 _reciprocal_rank_fusion() 단위 테스트.

Dense/Sparse 검색 자체는 DB가 있어야 하지만, 두 랭킹을 합치는 RRF 융합 로직은
순수 함수라 DB 없이도 검증할 수 있다.

실행: pytest tests/test_retrieval_strategy_agent.py
"""

from backend.api.services.retrieval import _reciprocal_rank_fusion


def _row(chunk_id: str, source: str = "law") -> tuple[str, str, dict, str]:
    return (chunk_id, source, {"law_name": chunk_id}, f"text-{chunk_id}")


class TestReciprocalRankFusion:
    def test_both_empty_returns_empty_list(self):
        assert _reciprocal_rank_fusion([], []) == []

    def test_dense_only_preserves_dense_order(self):
        dense = [_row("a"), _row("b"), _row("c")]
        fused = _reciprocal_rank_fusion(dense, [])
        assert [c["chunk_id"] for c in fused] == ["a", "b", "c"]

    def test_sparse_only_preserves_sparse_order(self):
        sparse = [_row("x"), _row("y")]
        fused = _reciprocal_rank_fusion([], sparse)
        assert [c["chunk_id"] for c in fused] == ["x", "y"]

    def test_item_ranked_first_in_both_lists_wins(self):
        dense  = [_row("a"), _row("b"), _row("c")]
        sparse = [_row("a"), _row("c"), _row("b")]
        fused = _reciprocal_rank_fusion(dense, sparse)
        assert fused[0]["chunk_id"] == "a"

    def test_item_only_in_one_list_still_included(self):
        dense  = [_row("a"), _row("b")]
        sparse = [_row("c")]
        fused = _reciprocal_rank_fusion(dense, sparse)
        ids = {c["chunk_id"] for c in fused}
        assert ids == {"a", "b", "c"}

    def test_duplicate_across_lists_not_duplicated_in_output(self):
        dense  = [_row("a"), _row("b")]
        sparse = [_row("b"), _row("a")]
        fused = _reciprocal_rank_fusion(dense, sparse)
        ids = [c["chunk_id"] for c in fused]
        assert len(ids) == len(set(ids)) == 2

    def test_appearing_in_both_ranks_above_appearing_in_one(self):
        dense  = [_row("only_dense"), _row("both")]
        sparse = [_row("only_sparse"), _row("both")]
        fused = _reciprocal_rank_fusion(dense, sparse)
        ranked_ids = [c["chunk_id"] for c in fused]
        assert ranked_ids.index("both") < ranked_ids.index("only_dense")
        assert ranked_ids.index("both") < ranked_ids.index("only_sparse")

    def test_in_both_flag_set_correctly(self):
        dense  = [_row("both"), _row("only_dense")]
        sparse = [_row("both"), _row("only_sparse")]
        fused = _reciprocal_rank_fusion(dense, sparse)
        by_id = {c["chunk_id"]: c["in_both"] for c in fused}
        assert by_id["both"] is True
        assert by_id["only_dense"] is False
        assert by_id["only_sparse"] is False

    def test_source_field_preserved(self):
        dense = [_row("a", source="precedent")]
        fused = _reciprocal_rank_fusion(dense, [])
        assert fused[0]["source"] == "precedent"
