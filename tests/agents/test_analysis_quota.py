# tests/agents/test_analysis_quota.py
"""GPT 크레딧이 없어도 서비스 분석이 죽지 않는다 — analysis 단계만 빈 근거로 넘어간다 (2026-10-12)."""

from backend.agents import analysis_agent
from backend.fb_check.api_errors import FatalAPIError


def test_fatal_api_error_falls_back_to_empty(monkeypatch):
    def boom(client, clause):
        raise FatalAPIError("insufficient_quota")

    monkeypatch.setattr(analysis_agent, "run_forward", boom)
    out = analysis_agent.analysis_node({"clause": "제2조 회사는 언제든지 해지할 수 있다."},
                                       {"configurable": {"client": None}})
    assert out["articles"] == [] and out["evidence_span"] == "" and out["reasoning"] == ""
