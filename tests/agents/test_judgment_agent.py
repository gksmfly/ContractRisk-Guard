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
    def test_full_clause_is_passed_not_evidence_span(self):
        """span이 있어도 **조항 원문**이 모델에 들어가야 한다."""
        with patch("backend.agents.judgment_agent.electra_predict", return_value=[]) as pred:
            judgment_node({"clause": "제9조 (해지) 회사는 언제든 계약을 해지할 수 있다",
                           "evidence_span": "언제든 계약을 해지할 수 있다"})
        pred.assert_called_once_with("제9조 (해지) 회사는 언제든 계약을 해지할 수 있다")

    def test_missing_span_still_uses_clause(self):
        with patch("backend.agents.judgment_agent.electra_predict", return_value=[]) as pred:
            judgment_node({"clause": "조항 원문"})
        pred.assert_called_once_with("조항 원문")


class TestJudgmentOutput:
    def test_needs_review_is_binary_on_article_hit(self):
        with patch("backend.agents.judgment_agent.electra_predict", return_value=["제9조"]):
            out = judgment_node({"clause": "c", "articles": ["제9조"]})
        assert out["model_articles"] == ["제9조"]
        assert out["needs_review"] is True
        assert out["verified"] is True          # GPT와 같은 조를 짚었다

    def test_no_article_means_no_review(self):
        with patch("backend.agents.judgment_agent.electra_predict", return_value=[]):
            out = judgment_node({"clause": "c", "articles": ["제9조"]})
        assert out["needs_review"] is False
        assert out["verified"] is False

    def test_verified_is_intersection_not_gpt_echo(self):
        """`verified`는 GPT 라벨의 복사가 아니라 **교집합**이다."""
        with patch("backend.agents.judgment_agent.electra_predict", return_value=["제9조"]):
            out = judgment_node({"clause": "c", "articles": ["제6조"]})
        assert out["verified"] is False
