# tests/agents/test_explain_agent.py
"""Explain Agent — 꺼져 있으면 불변, 켜져도 환각이면 기존 설명을 유지한다 (2026-10-12)."""

from backend.agents import explain_agent as ex
from backend.retrieval.schemas import LegalBasis

CLAUSE = "회사는 사전 통보 없이 언제든지 이용계약을 해지할 수 있다."
BASIS = [LegalBasis(law="약관의 규제에 관한 법률", article="제9조", description="계약의 해제·해지 …")]


def _state():
    return {"clause": CLAUSE, "legal_basis": BASIS, "reasoning": "기존 설명"}


def test_disabled_is_noop(monkeypatch):
    monkeypatch.delenv("EXPLAIN_MODEL", raising=False)
    assert ex.explain_node(_state()) == {}


def test_non_local_model_is_disabled(monkeypatch):
    monkeypatch.setenv("EXPLAIN_MODEL", "gpt-4o-mini")      # GPT는 지원하지 않는다
    assert ex.explain_node(_state()) == {}


def test_valid_output_replaces_reasoning(monkeypatch):
    monkeypatch.setenv("EXPLAIN_MODEL", "local:qwen2.5-14b")
    monkeypatch.setattr(ex, "generate", lambda *a, **k: {
        "evidence_span": "사전 통보 없이 언제든지", "explanation": "제9조에 비추어 일방 해지가 문제될 수 있다.",
        "cited_articles": ["제9조"], "suggestion": "통보 기간을 둔다."})
    out = ex.explain_node(_state())
    assert "제9조" in out["reasoning"] and "수정 제안" in out["reasoning"]
    assert out["evidence_span"] == "사전 통보 없이 언제든지"


def test_fabricated_span_is_dropped_but_explanation_kept(monkeypatch):
    monkeypatch.setenv("EXPLAIN_MODEL", "local:qwen2.5-14b")
    monkeypatch.setattr(ex, "generate", lambda *a, **k: {
        "evidence_span": "원문에 없는 문장", "explanation": "일방 해지가 문제될 수 있다.", "cited_articles": ["제9조"], "suggestion": ""})
    out = ex.explain_node(_state())
    assert "evidence_span" not in out and out["reasoning"].startswith("일방 해지")


def test_unprovided_article_mention_falls_back(monkeypatch):
    monkeypatch.setenv("EXPLAIN_MODEL", "local:qwen2.5-14b")
    monkeypatch.setattr(ex, "generate", lambda *a, **k: {
        "evidence_span": "", "explanation": "제12조에 따라 무효이다.", "cited_articles": ["제9조"], "suggestion": ""})
    assert ex.explain_node(_state()) == {}


def test_clause_own_article_mention_is_allowed(monkeypatch):
    monkeypatch.setenv("EXPLAIN_MODEL", "local:qwen2.5-14b")
    clause = "제5조에 따라 회사는 사전 통보 없이 해지할 수 있다."
    monkeypatch.setattr(ex, "generate", lambda *a, **k: {
        "evidence_span": "", "explanation": "제5조를 근거로 한 일방 해지가 제9조상 문제될 수 있다.", "cited_articles": ["제9조"], "suggestion": ""})
    assert "reasoning" in ex.explain_node({"clause": clause, "legal_basis": BASIS})


def test_generation_failure_keeps_existing(monkeypatch):
    monkeypatch.setenv("EXPLAIN_MODEL", "local:qwen2.5-14b")

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(ex, "generate", boom)
    assert ex.explain_node(_state()) == {}


def test_bare_article_numbers_are_accepted(monkeypatch):
    """모델이 "6"·"6조"처럼 적어도 제공된 조(제6조)면 위반이 아니다 (v2 프롬프트 사고 재발 방지)."""
    monkeypatch.setenv("EXPLAIN_MODEL", "local:qwen2.5-14b")
    basis = [LegalBasis(law="약관의 규제에 관한 법률", article="제6조", description="일반원칙 …")]
    for cited in (["6"], ["6조"], ["제6조"], [6]):
        monkeypatch.setattr(ex, "generate", lambda *a, _c=cited, **k: {
            "evidence_span": "", "explanation": "불공정할 수 있다.", "cited_articles": _c, "suggestion": ""})
        assert "reasoning" in ex.explain_node({"clause": CLAUSE, "legal_basis": basis}), cited


def test_unprovided_number_is_still_a_violation(monkeypatch):
    monkeypatch.setenv("EXPLAIN_MODEL", "local:qwen2.5-14b")
    monkeypatch.setattr(ex, "generate", lambda *a, **k: {
        "evidence_span": "", "explanation": "불공정할 수 있다.", "cited_articles": ["8"], "suggestion": ""})
    assert ex.explain_node(_state()) == {}      # BASIS는 제9조만 제공
