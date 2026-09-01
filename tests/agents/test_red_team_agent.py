# tests/agents/test_red_team_agent.py
"""
backend/agents/red_team_agent.py 단위 테스트.
search_similar_labeled_clauses()는 DB 호출, OpenAI 호출은 client.chat.completions.create()라
둘 다 monkeypatch/가짜 클라이언트로 대체한다.

실행: pytest tests/agents/test_red_team_agent.py
"""

import json
from collections.abc import Iterator
from typing import Any

import pytest

import backend.agents.red_team_agent as red_team_agent


@pytest.fixture(autouse=True)
def _reset_cache() -> Iterator[None]:
    """모듈 캐시(`_NEIGHBORS_LABELED`)가 테스트 간에 새지 않게 한다."""
    red_team_agent.reset_neighbor_cache()
    yield
    red_team_agent.reset_neighbor_cache()

_CONFIG = {"configurable": {"client": None}}  # client는 각 테스트에서 필요시 교체


def _neighbor(articles: list[str] | None, similarity: float, text: str = "비슷한 조항") -> dict:
    """조 라벨을 가진 이웃. `articles=None`이면 **옛 라벨만 있는 이웃**(비교 불가)."""
    n = {"chunk_id": "x", "domain": "해지_조항", "risk_level": "High", "text": text, "similarity": similarity}
    if articles is not None:
        n["articles"] = articles
    return n


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})]


class _FakeChatCompletions:
    def __init__(self, content: str | None, raise_error: bool = False) -> None:
        self._content = content
        self._raise_error = raise_error
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        if self._raise_error:
            raise RuntimeError("API 오류(테스트용)")
        return _FakeResponse(self._content)


class _FakeClient:
    def __init__(self, content: str | None = None, raise_error: bool = False) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeChatCompletions(content, raise_error)


class TestRedTeamNode:
    def test_no_neighbors_returns_empty_note_and_no_llm_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(red_team_agent, "search_similar_labeled_clauses", lambda *a, **k: [])
        client = _FakeClient(content=json.dumps({"rebuttal": "안 나와야 함"}))
        result = red_team_agent.red_team_node(
            {"clause": "c", "evidence_span": "e", "model_articles": ["제9조"]}, {"configurable": {"client": client}}
        )
        assert result["redteam_note"] == ""
        assert client.chat.completions.calls == 0  # 충돌 없으면 LLM 호출 자체를 안 함(비용 절감)

    def test_similar_but_same_label_returns_empty_note_and_no_llm_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            red_team_agent, "search_similar_labeled_clauses",
            lambda *a, **k: [_neighbor(["제9조"], 0.90)],
        )
        client = _FakeClient(content=json.dumps({"rebuttal": "안 나와야 함"}))
        result = red_team_agent.red_team_node(
            {"clause": "c", "evidence_span": "e", "model_articles": ["제9조"]}, {"configurable": {"client": client}}
        )
        assert result["redteam_note"] == ""
        assert client.chat.completions.calls == 0

    def test_conflict_above_threshold_calls_llm_for_rebuttal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            red_team_agent, "search_similar_labeled_clauses",
            lambda *a, **k: [_neighbor(["제7조"], 0.80)],
        )
        client = _FakeClient(content=json.dumps({"rebuttal": "이 조항은 실질적으로 다른 맥락이라 재검토가 필요합니다."}))
        result = red_team_agent.red_team_node(
            {"clause": "c", "evidence_span": "e", "model_articles": ["제9조"]}, {"configurable": {"client": client}}
        )
        assert result["redteam_note"] == "이 조항은 실질적으로 다른 맥락이라 재검토가 필요합니다."
        assert client.chat.completions.calls == 1

    def test_llm_failure_falls_back_to_template_note(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            red_team_agent, "search_similar_labeled_clauses",
            lambda *a, **k: [_neighbor(["제7조"], 0.80)],
        )
        client = _FakeClient(raise_error=True)
        result = red_team_agent.red_team_node(
            {"clause": "c", "evidence_span": "e", "model_articles": ["제9조"]}, {"configurable": {"client": client}}
        )
        assert result["redteam_note"] != ""
        assert "제7조" in result["redteam_note"]  # 안전망 템플릿 문구로 대체됨(이웃의 조를 인용)

    def test_different_label_below_threshold_does_not_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            red_team_agent, "search_similar_labeled_clauses",
            lambda *a, **k: [_neighbor("Low", 0.60)],
        )
        client = _FakeClient(content=json.dumps({"rebuttal": "안 나와야 함"}))
        result = red_team_agent.red_team_node(
            {"clause": "c", "evidence_span": "e", "model_articles": ["제9조"]}, {"configurable": {"client": client}}
        )
        assert result["redteam_note"] == ""
        assert client.chat.completions.calls == 0

    def test_stops_at_first_qualifying_neighbor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 첫 이웃(제9조, 0.95)은 조가 같아서 안 걸리고, 두 번째(제7조, 0.80)에서 걸려야 함
        neighbors = [_neighbor(["제9조"], 0.95), _neighbor(["제7조"], 0.80)]
        monkeypatch.setattr(red_team_agent, "search_similar_labeled_clauses", lambda *a, **k: neighbors)
        client = _FakeClient(content=json.dumps({"rebuttal": "반박 근거"}))
        result = red_team_agent.red_team_node(
            {"clause": "c", "evidence_span": "e", "model_articles": ["제9조"]}, {"configurable": {"client": client}}
        )
        assert result["redteam_note"] == "반박 근거"
        assert client.chat.completions.calls == 1  # 두 번째(걸리는) 이웃에서 딱 한 번만 호출


class TestArticleTransition:
    """조 taxonomy 전환 중의 계약 — 이웃에 조 라벨이 없으면 **침묵**한다.

    judgment_agent가 risk_level을 더 이상 내지 않는데 `clean_clauses` 테이블은 아직
    옛 라벨로 적재돼 있다. 예전 비교식(`neighbor["risk_level"] != risk_level`)을 그대로
    두면 `None`과 비교하게 돼 **유사도만 넘으면 무조건 반박이 발동한다**. 틀린 근거로
    말하는 것보다 아무 말도 안 하는 편이 낫다.
    """

    def test_neighbor_without_articles_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            red_team_agent, "search_similar_labeled_clauses",
            lambda *a, **k: [_neighbor(None, 0.99)],       # 옛 라벨만 있는 이웃
        )
        client = _FakeClient(content=json.dumps({"rebuttal": "발동하면 안 됨"}))
        result = red_team_agent.red_team_node(
            {"clause": "c", "evidence_span": "e", "model_articles": ["제9조"]},
            {"configurable": {"client": client}},
        )
        assert result["redteam_note"] == ""
        assert client.chat.completions.calls == 0

    def test_article_conflict_fires(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            red_team_agent, "search_similar_labeled_clauses",
            lambda *a, **k: [_neighbor(["제7조"], 0.95)],
        )
        client = _FakeClient(content=json.dumps({"rebuttal": "조가 다릅니다"}))
        result = red_team_agent.red_team_node(
            {"clause": "c", "evidence_span": "e", "model_articles": ["제9조"]},
            {"configurable": {"client": client}},
        )
        assert result["redteam_note"] == "조가 다릅니다"
        assert client.chat.completions.calls == 1
