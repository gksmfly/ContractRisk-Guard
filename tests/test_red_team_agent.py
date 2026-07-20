# tests/test_red_team_agent.py
"""
backend/agents/red_team_agent.py의 판정 로직 단위 테스트.
search_similar_labeled_clauses()는 DB 호출이라 monkeypatch로 대체한다.

실행: pytest tests/test_red_team_agent.py
"""

import backend.agents.red_team_agent as red_team_agent


def _neighbor(risk_level: str, similarity: float, text: str = "비슷한 조항") -> dict:
    return {"chunk_id": "x", "domain": "해지_조항", "risk_level": risk_level, "text": text, "similarity": similarity}


class TestRedTeamNode:
    def test_no_neighbors_returns_empty_note(self, monkeypatch):
        monkeypatch.setattr(red_team_agent, "search_similar_labeled_clauses", lambda *a, **k: [])
        result = red_team_agent.red_team_node({"clause": "c", "evidence_span": "e", "risk_level": "High"})
        assert result["redteam_note"] == ""

    def test_similar_but_same_label_returns_empty_note(self, monkeypatch):
        monkeypatch.setattr(
            red_team_agent, "search_similar_labeled_clauses",
            lambda *a, **k: [_neighbor("High", 0.90)],
        )
        result = red_team_agent.red_team_node({"clause": "c", "evidence_span": "e", "risk_level": "High"})
        assert result["redteam_note"] == ""

    def test_similar_and_different_label_above_threshold_flags(self, monkeypatch):
        monkeypatch.setattr(
            red_team_agent, "search_similar_labeled_clauses",
            lambda *a, **k: [_neighbor("Low", 0.80)],
        )
        result = red_team_agent.red_team_node({"clause": "c", "evidence_span": "e", "risk_level": "High"})
        assert result["redteam_note"] != ""
        assert "Low" in result["redteam_note"]

    def test_different_label_below_threshold_does_not_flag(self, monkeypatch):
        monkeypatch.setattr(
            red_team_agent, "search_similar_labeled_clauses",
            lambda *a, **k: [_neighbor("Low", 0.60)],
        )
        result = red_team_agent.red_team_node({"clause": "c", "evidence_span": "e", "risk_level": "High"})
        assert result["redteam_note"] == ""

    def test_stops_at_first_qualifying_neighbor(self, monkeypatch):
        neighbors = [_neighbor("High", 0.95), _neighbor("Low", 0.80)]
        monkeypatch.setattr(red_team_agent, "search_similar_labeled_clauses", lambda *a, **k: neighbors)
        result = red_team_agent.red_team_node({"clause": "c", "evidence_span": "e", "risk_level": "High"})
        assert result["redteam_note"] != ""
