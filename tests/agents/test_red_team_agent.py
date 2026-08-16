# tests/agents/test_red_team_agent.py
"""
backend/agents/red_team_agent.py 단위 테스트.
search_similar_labeled_clauses()는 DB 호출, OpenAI 호출은 client.chat.completions.create()라
둘 다 monkeypatch/가짜 클라이언트로 대체한다.

실행: pytest tests/agents/test_red_team_agent.py
"""

import json

import backend.agents.red_team_agent as red_team_agent

_CONFIG = {"configurable": {"client": None}}  # client는 각 테스트에서 필요시 교체


def _neighbor(risk_level: str, similarity: float, text: str = "비슷한 조항") -> dict:
    return {"chunk_id": "x", "domain": "해지_조항", "risk_level": risk_level, "text": text, "similarity": similarity}


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})]


class _FakeChatCompletions:
    def __init__(self, content: str | None, raise_error: bool = False):
        self._content = content
        self._raise_error = raise_error
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self._raise_error:
            raise RuntimeError("API 오류(테스트용)")
        return _FakeResponse(self._content)


class _FakeClient:
    def __init__(self, content: str | None = None, raise_error: bool = False):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeChatCompletions(content, raise_error)


class TestRedTeamNode:
    def test_no_neighbors_returns_empty_note_and_no_llm_call(self, monkeypatch):
        monkeypatch.setattr(red_team_agent, "search_similar_labeled_clauses", lambda *a, **k: [])
        client = _FakeClient(content=json.dumps({"rebuttal": "안 나와야 함"}))
        result = red_team_agent.red_team_node(
            {"clause": "c", "evidence_span": "e", "risk_level": "High"}, {"configurable": {"client": client}}
        )
        assert result["redteam_note"] == ""
        assert client.chat.completions.calls == 0  # 충돌 없으면 LLM 호출 자체를 안 함(비용 절감)

    def test_similar_but_same_label_returns_empty_note_and_no_llm_call(self, monkeypatch):
        monkeypatch.setattr(
            red_team_agent, "search_similar_labeled_clauses",
            lambda *a, **k: [_neighbor("High", 0.90)],
        )
        client = _FakeClient(content=json.dumps({"rebuttal": "안 나와야 함"}))
        result = red_team_agent.red_team_node(
            {"clause": "c", "evidence_span": "e", "risk_level": "High"}, {"configurable": {"client": client}}
        )
        assert result["redteam_note"] == ""
        assert client.chat.completions.calls == 0

    def test_conflict_above_threshold_calls_llm_for_rebuttal(self, monkeypatch):
        monkeypatch.setattr(
            red_team_agent, "search_similar_labeled_clauses",
            lambda *a, **k: [_neighbor("Low", 0.80)],
        )
        client = _FakeClient(content=json.dumps({"rebuttal": "이 조항은 실질적으로 다른 맥락이라 재검토가 필요합니다."}))
        result = red_team_agent.red_team_node(
            {"clause": "c", "evidence_span": "e", "risk_level": "High"}, {"configurable": {"client": client}}
        )
        assert result["redteam_note"] == "이 조항은 실질적으로 다른 맥락이라 재검토가 필요합니다."
        assert client.chat.completions.calls == 1

    def test_llm_failure_falls_back_to_template_note(self, monkeypatch):
        monkeypatch.setattr(
            red_team_agent, "search_similar_labeled_clauses",
            lambda *a, **k: [_neighbor("Low", 0.80)],
        )
        client = _FakeClient(raise_error=True)
        result = red_team_agent.red_team_node(
            {"clause": "c", "evidence_span": "e", "risk_level": "High"}, {"configurable": {"client": client}}
        )
        assert result["redteam_note"] != ""
        assert "Low" in result["redteam_note"]  # 안전망 템플릿 문구로 대체됨

    def test_different_label_below_threshold_does_not_flag(self, monkeypatch):
        monkeypatch.setattr(
            red_team_agent, "search_similar_labeled_clauses",
            lambda *a, **k: [_neighbor("Low", 0.60)],
        )
        client = _FakeClient(content=json.dumps({"rebuttal": "안 나와야 함"}))
        result = red_team_agent.red_team_node(
            {"clause": "c", "evidence_span": "e", "risk_level": "High"}, {"configurable": {"client": client}}
        )
        assert result["redteam_note"] == ""
        assert client.chat.completions.calls == 0

    def test_stops_at_first_qualifying_neighbor(self, monkeypatch):
        # 첫 이웃(High, 0.95)은 risk_level이 같아서 안 걸리고, 두 번째(Low, 0.80)에서 걸려야 함
        neighbors = [_neighbor("High", 0.95), _neighbor("Low", 0.80)]
        monkeypatch.setattr(red_team_agent, "search_similar_labeled_clauses", lambda *a, **k: neighbors)
        client = _FakeClient(content=json.dumps({"rebuttal": "반박 근거"}))
        result = red_team_agent.red_team_node(
            {"clause": "c", "evidence_span": "e", "risk_level": "High"}, {"configurable": {"client": client}}
        )
        assert result["redteam_note"] == "반박 근거"
        assert client.chat.completions.calls == 1  # 두 번째(걸리는) 이웃에서 딱 한 번만 호출
