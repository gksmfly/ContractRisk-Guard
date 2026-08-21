# tests/test_evidence_selection_agent.py
"""
backend/agents/evidence_selection_agent.py의 순수 로직(후보 선택, fallback) 단위 테스트.
DB·CrossEncoder 불필요.

**이 파일의 이전 버전은 버그를 검증하고 있었다.** 법원 심급 가중치 테스트가
`court="고등법원"`이라는 값을 썼는데, DB의 실제 `court` 값은 58종이고
(`서울고법`·`부산고등법원`·`대전고법(청주)` …) 정확히 `"고등법원"`인 값은 **하나도 없다**.
이상적인 입력을 지어내 테스트했기 때문에, 완전일치 조회가 프로덕션에서 한 번도 매칭되지
않는다는 사실이 드러나지 않았다. 가중치는 2026-08-16 실측으로 폐지됐다(agent docstring 참고).

교훈이 아래 `TestPrecedentOrdering::test_real_world_court_values_do_not_reorder`에 남아 있다 —
**실제 DB에 존재하는 값**으로 테스트해야 한다.

실행: pytest tests/agents/test_evidence_selection_agent.py
"""

from backend.agents.evidence_selection_agent import (
    _FINAL_K, evidence_selection_node, LEGAL_BASIS_FALLBACK,
)

# DB의 metadata->>'court' 실측값에서 가져온 표기들(총 58종 중 대표).
# 지어낸 "고등법원"·"지방법원" 대신 이 값들을 쓴다.
REAL_COURT_VALUES = ["대법원", "서울고법", "부산고등법원", "수원고등법원",
                     "대전고법(청주)", "서울중앙지법", "대구지법"]


def _candidate(chunk_id: str, court: str = "", in_both: bool = False) -> dict:
    return {
        "chunk_id": chunk_id, "source": "precedent",
        "metadata": {"court": court, "case_name": chunk_id, "case_number": chunk_id},
        "text": f"text-{chunk_id}", "in_both": in_both,
    }


class TestPrecedentOrdering:
    """판례는 RRF 순서를 그대로 써야 한다 — 재랭킹을 되살리는 변경에 대한 회귀 방지."""

    def test_rrf_order_is_preserved(self):
        # _FINAL_K가 몇이든 성립하도록 후보를 넉넉히 만든다 — 상수를 조정할 때마다
        # 테스트를 같이 고치게 되면 "값이 바뀐 것"과 "동작이 깨진 것"을 구분할 수 없다.
        names = [f"cand{i}" for i in range(_FINAL_K + 3)]
        candidates = [_candidate(n) for n in names]
        state = {"domain": "해지_조항", "retrieval_candidates": {"law": [], "precedent": candidates}}
        result = evidence_selection_node(state)
        assert [b.article for b in result["legal_basis"]] == names[:_FINAL_K]

    def test_real_world_court_values_do_not_reorder(self):
        """실제 DB 표기값을 넣어도 순서가 바뀌지 않는다.

        대법원이 뒤에 있어도 앞으로 끌어올려지면 안 된다 — 실측에서 그 가산이
        관련도 순위를 덮어써 성능을 떨어뜨렸다(@2 6.4% vs 폐지 9.4%, p=0.035).
        """
        courts = list(reversed(REAL_COURT_VALUES))
        assert len(courts) > _FINAL_K, "법원 표기 표본이 _FINAL_K보다 많아야 순서 검증이 의미 있다"
        candidates = [_candidate(f"c{i}", court=c) for i, c in enumerate(courts)]
        state = {"domain": "해지_조항", "retrieval_candidates": {"law": [], "precedent": candidates}}
        result = evidence_selection_node(state)
        assert [b.article for b in result["legal_basis"]] == [f"c{i}" for i in range(_FINAL_K)]

    def test_empty_precedents_falls_back(self):
        state = {"domain": "해지_조항", "retrieval_candidates": {"law": [], "precedent": []}}
        result = evidence_selection_node(state)
        assert len(result["legal_basis"]) == len(LEGAL_BASIS_FALLBACK["해지_조항"])


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
