# tests/agents/test_judgment_agent.py
"""judgment_node의 **입력**을 고정하는 회귀 테스트.

2026-08-31까지 `evidence_span or clause`를 모델에 넣었다. `models/v4`(span 증강 학습)
시절의 규칙인데 `article_v1`은 원문 전용으로 학습·채점했으므로, 모델만 갈아끼우고
입력이 남아 보고값(재현 78.0%)이 운영을 설명하지 못했다.

페어드 실측(`backend/eval/input_parity_eval.py`, span 보유 136건):

    조항 재현  원문 81.6% → 조각 72.1%   -9.6%p [-16.9,-2.9]  유의

**타입 검사로는 안 잡힌다** — 두 값 다 str이다. 그래서 테스트로 박는다.
"""

from unittest.mock import patch

from backend.agents.judgment_agent import judgment_node


class TestJudgmentInput:
    def test_full_clause_is_passed_not_evidence_span(self) -> None:
        """span이 있어도 **조항 원문**이 모델에 들어가야 한다."""
        with patch("backend.agents.judgment_agent.predict_articles", return_value=[]) as pred:
            judgment_node({"clause": "제9조 (해지) 회사는 언제든 계약을 해지할 수 있다",
                           "evidence_span": "언제든 계약을 해지할 수 있다"})
        pred.assert_called_once_with("제9조 (해지) 회사는 언제든 계약을 해지할 수 있다", doc_style=None)

    def test_missing_span_still_uses_clause(self) -> None:
        with patch("backend.agents.judgment_agent.predict_articles", return_value=[]) as pred:
            judgment_node({"clause": "조항 원문"})
        pred.assert_called_once_with("조항 원문", doc_style=None)


class TestJudgmentOutput:
    def test_needs_review_is_binary_on_article_hit(self) -> None:
        with patch("backend.agents.judgment_agent.predict_articles", return_value=["제9조"]):
            out = judgment_node({"clause": "c", "articles": ["제9조"]})
        assert out["model_articles"] == ["제9조"]
        assert out["needs_review"] is True
        assert out["verified"] is True          # GPT와 같은 조를 짚었다

    def test_no_article_means_no_review(self) -> None:
        with patch("backend.agents.judgment_agent.predict_articles", return_value=[]):
            out = judgment_node({"clause": "c", "articles": ["제9조"]})
        assert out["needs_review"] is False
        assert out["verified"] is False

    def test_verified_is_intersection_not_gpt_echo(self) -> None:
        """`verified`는 GPT 라벨의 복사가 아니라 **교집합**이다."""
        with patch("backend.agents.judgment_agent.predict_articles", return_value=["제9조"]):
            out = judgment_node({"clause": "c", "articles": ["제6조"]})
        assert out["verified"] is False


class TestDocStyleRouting:
    """문서 종류별 기준 분리(paper_plan §7-12·7-13). 라우터가 있을 때만 켜지고, 판별은 문서 단위다."""

    def test_doc_style_is_forwarded_to_predict(self) -> None:
        with patch("backend.agents.judgment_agent.predict_articles", return_value=[]) as pred:
            judgment_node({"clause": "조항 원문", "doc_style": "company"})
        pred.assert_called_once_with("조항 원문", doc_style="company")

    def test_thresholds_pick_company_set_only_when_loaded(self) -> None:
        import numpy as np

        import backend.agents.judgment_agent as ja
        base, comp = np.array([0.5, 0.5]), np.array([0.05, 0.05])
        with patch.object(ja, "_thresholds", base), patch.object(ja, "_thresholds_company", comp):
            assert ja._thresholds_for("company") is comp
            assert ja._thresholds_for("standard") is base
            assert ja._thresholds_for(None) is base
        with patch.object(ja, "_thresholds", base), patch.object(ja, "_thresholds_company", None):
            assert ja._thresholds_for("company") is base      # 라우터 없는 체크포인트 = 예전과 같다

    def test_route_document_uses_document_mean(self) -> None:
        """조항 하나가 회사 약관처럼 보여도 문서 평균이 표준계약서면 문서 전체가 표준계약서다."""
        import numpy as np

        import backend.agents.judgment_agent as ja

        class _Clf:
            def predict_proba(self, X):
                return np.array([[0.1, 0.9], [0.8, 0.2], [0.8, 0.2]])   # 평균 p(회사) = 0.43

        class _Vec:
            def transform(self, xs):
                return xs

        router = {"vectorizer": _Vec(), "classifier": _Clf(), "cutoff": 0.5}
        with patch.object(ja, "_load_router", return_value=None), patch.object(ja, "_router", router):
            assert ja.route_document(["a", "b", "c"]) == "standard"
        with patch.object(ja, "_load_router", return_value=None), patch.object(ja, "_router", None):
            assert ja.route_document(["a"]) is None
