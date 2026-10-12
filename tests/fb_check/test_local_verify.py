# tests/fb_check/test_local_verify.py
"""`VERIFY_MODEL=local:...`이면 OpenAI를 부르지 않고 로컬 백엔드로 간다 (2026-10-12)."""

from unittest.mock import MagicMock

from backend.fb_check import consistency_verification as cv


def test_local_verify_skips_openai(monkeypatch):
    calls = []

    def fake_chat_json(model, messages, max_new_tokens=200):
        calls.append((model, messages[-1]["content"]))
        return {"articles": ["제8조", "제99조"], "risk_level": "High"}

    monkeypatch.setattr(cv.local_llm, "chat_json", fake_chat_json)
    client = MagicMock()
    out = cv.run_verify(client, "예약금은 반환하지 않는다", mode="span", model="local:qwen2.5-14b")
    client.chat.completions.create.assert_not_called()
    assert calls and calls[0][0] == "local:qwen2.5-14b"
    assert out["articles"] == ["제8조"] and out["risk_level"] == "High" and out["model"] == "local:qwen2.5-14b"


def test_local_verify_parse_failure_returns_none(monkeypatch):
    monkeypatch.setattr(cv.local_llm, "chat_json", lambda *a, **k: None)
    assert cv.run_verify(MagicMock(), "조항", mode="clause", model="local:qwen2.5-14b") is None


def test_is_local():
    assert cv.local_llm.is_local("local:qwen2.5-14b")
    assert not cv.local_llm.is_local("gpt-4o-mini")


def test_local_forward_skips_openai(monkeypatch):
    from backend.fb_check import forward_labeling as fl
    monkeypatch.setattr(fl.local_llm, "chat_json",
                        lambda model, messages, max_new_tokens=300: {"articles": ["제6조"], "risk_level": "Medium",
                                                                      "evidence_span": "x", "reasoning": "y"})
    client = MagicMock()
    out = fl.run_forward(client, "조항", model="local:qwen2.5-14b")
    client.chat.completions.create.assert_not_called()
    assert out is not None and out["articles"] == ["제6조"]
