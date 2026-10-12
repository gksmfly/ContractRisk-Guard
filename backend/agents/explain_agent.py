# backend/agents/explain_agent.py
"""Explain Agent — 예측한 조의 **조문 원문**을 읽고 근거 기반 설명을 만든다 (RAG 안 A, 2026-10-12).

## 왜 이 단계가 있나

지금까지 설명(`reasoning`)은 `analysis` 단계에서 GPT가 **판단 전에, 아무 근거도 읽지 않고** 만들었다.
여기서는 순서를 뒤집는다: KoELECTRA가 조를 지목하고 → 그 조의 조문 원문을 조회하고 →
LLM이 조항과 조문을 **함께 읽고** 설명한다. 검색한 내용을 LLM이 읽고 답을 만드는 구조(RAG)가 처음 생긴다.

신뢰의 근거는 판례 검색(hit@5 14%)이 아니라 **조회한 조문 원문**이다. 판례는
`EXPLAIN_USE_PRECEDENTS=1`일 때만 프롬프트에 넣는다(기본 꺼짐).

## 켜는 법 / 기본 동작

`EXPLAIN_MODEL=local:qwen2.5-14b`일 때만 동작한다. 비어 있으면 이 노드는 **아무것도 하지 않는다**(기존 동작 불변).
`local:` 이 아닌 값은 지원하지 않는다 — GPT 결제를 쓰지 않는다는 방침이다(`local_llm.py` 참고).

## 환각 방어 (후처리)

원출력을 그대로 믿지 않는다:
  ① 근거 문구가 조항 원문에 그대로 있지 않으면 비운다
  ② 제공하지 않은 "제N조"를 설명이 언급하면(조항 원문에 나온 번호는 제외) **기존 설명으로 되돌린다**
평가(`eval/rag/explain_eval.py`)는 후처리 **전** 원출력으로 잰다 — 후처리가 가린 위반율을 보려는 것이다.
"""

import os
import re
from typing import Any

from backend.agents.state import ClauseState
from backend.fb_check import local_llm
from backend.fb_check.backward_grounding import check_snippet_exists
from backend.utils import load_logger

logger = load_logger("explain_agent.log")

MAX_ARTICLE_CHARS = 700     # 조문 하나당 프롬프트에 넣는 길이
MAX_PRECEDENT_CHARS = 250
MAX_PRECEDENTS = 2
MAX_NEW_TOKENS = 450

_ART_RE = re.compile(r"제\s*(\d+)\s*조(?:의\s*(\d+))?")
# 판례 사건번호 형태(2008약관1149, 2019다12345, 2020두1234 …) — 환각 검사용
CASE_NO_RE = re.compile(r"(?:19|20)\d{2}\s*(?:약관|[가-힣]{1,3})\s*\d{2,6}")

PROMPT_VERSION = "expl-v2"     # v1은 출력 형식 예시에 실제 조 번호("제8조")를 적어, 모델이 그대로 복사했다(평가 A0 인용 위반 58/58건)

SYSTEM = """당신은 한국 약관규제법 전문가입니다. 계약 조항 하나와, 그 조항에 해당할 가능성이 있다고 판단된 약관규제법 조문이 주어집니다.
주어진 조문만 근거로, 이 조항이 왜 문제가 될 수 있는지 설명하세요.

규칙:
- cited_articles에는 [적용 조문]에 적힌 조 번호만 넣으세요. 목록에 없는 조 번호는 설명에도 쓰지 마세요.
- evidence_span은 [계약 조항]에서 한 구간을 **글자 그대로 이어서 복사한** 가장 핵심적인 문구입니다. 바꿔 쓰거나, 중간을 생략("...")하거나, 따옴표 종류를 바꾸지 마세요.
- explanation은 2~3문장, suggestion은 1문장으로 쓰세요. 단정하지 말고 "문제가 될 수 있다"처럼 표현하세요.
- [참고 판례]가 있으면 참고만 하세요. 관련이 없을 수 있으며, 주어지지 않은 사건번호는 만들지 마세요.

다음 형식의 JSON 하나만 출력하세요. 꺾쇠 안은 설명이므로 실제 내용으로 바꿔 쓰세요:
{"evidence_span": "<조항에서 그대로 복사한 문구>", "explanation": "<설명>", "cited_articles": ["<[적용 조문]에 있는 조 번호>"], "suggestion": "<수정 제안>"}"""


def enabled() -> bool:
    return local_llm.is_local(os.environ.get("EXPLAIN_MODEL", ""))


def use_precedents() -> bool:
    return os.environ.get("EXPLAIN_USE_PRECEDENTS", "0").strip().lower() in ("1", "true", "yes")


def _norm_article(num: str, sub: str | None = None) -> str:
    return f"제{int(num)}조" + (f"의{int(sub)}" if sub else "")


def build_messages(clause: str, legal_basis: list[Any], precedents: list[Any] | None = None) -> list[dict]:
    arts = "\n".join(f"- {b.law} {b.article}: {b.description[:MAX_ARTICLE_CHARS]}" for b in legal_basis)
    user = f"[계약 조항]\n{clause}\n\n[적용 조문]\n{arts}"
    if precedents:
        refs = "\n".join(f"- {p.law} {p.article}: {p.description[:MAX_PRECEDENT_CHARS]}" for p in precedents[:MAX_PRECEDENTS])
        user += f"\n\n[참고 판례]\n{refs}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def generate(clause: str, legal_basis: list[Any], precedents: list[Any] | None = None,
             model: str | None = None) -> dict | None:
    """프롬프트를 만들어 한 번 생성한다. JSON 파싱 실패면 None. **후처리하지 않은 원출력**이다."""
    model = model or os.environ.get("EXPLAIN_MODEL", "")
    return local_llm.chat_json(model, build_messages(clause, legal_basis, precedents), max_new_tokens=MAX_NEW_TOKENS)


def mentioned_articles(text: str) -> set[str]:
    return {_norm_article(m.group(1), m.group(2)) for m in _ART_RE.finditer(text or "")}


def audit(clause: str, raw: dict | None, allowed: list[str], precedents: list[Any] | None = None) -> dict:
    """원출력의 환각·형식 문제를 센다. 평가와 후처리가 같은 규칙을 쓴다."""
    ok_format = (isinstance(raw, dict) and isinstance(raw.get("explanation"), str) and raw["explanation"].strip() != ""
                 and isinstance(raw.get("evidence_span"), str))
    if not ok_format:
        return {"format_ok": False, "span_verbatim": False, "bad_cited": False, "bad_mention": False, "bad_case_no": False}
    allowed_set = set(allowed)
    in_clause = mentioned_articles(clause)             # 조항 원문이 스스로 언급한 번호는 허용
    cited = raw.get("cited_articles") if isinstance(raw.get("cited_articles"), list) else []
    text = f"{raw['explanation']} {raw.get('suggestion', '')}"
    stray = mentioned_articles(text) - allowed_set - in_clause
    given_cases = {m.group(0).replace(" ", "") for p in (precedents or []) for m in CASE_NO_RE.finditer(f"{p.article} {p.description}")}
    cases = {m.group(0).replace(" ", "") for m in CASE_NO_RE.finditer(text)}
    return {
        "format_ok": True,
        "span_verbatim": bool(raw["evidence_span"].strip()) and check_snippet_exists(clause, raw["evidence_span"]),
        "bad_cited": any(a not in allowed_set for a in cited if isinstance(a, str)),
        "bad_mention": bool(stray),
        "bad_case_no": bool(cases - given_cases),
    }


def explain_node(state: ClauseState) -> dict:
    """LangGraph 노드. 꺼져 있거나 실패하면 빈 dict — 기존 `reasoning`·`evidence_span`을 그대로 둔다."""
    if not enabled():
        return {}
    legal_basis = state.get("legal_basis") or []
    if not legal_basis:
        return {}
    clause = state["clause"]
    precedents = (state.get("precedent_refs") or []) if use_precedents() else None
    try:
        raw = generate(clause, legal_basis, precedents)
    except Exception as e:   # GPU 메모리 부족 등 — 설명이 없어도 판단 결과는 보여 줘야 한다
        logger.warning(f"  설명 생성 실패(기존 설명 유지): {type(e).__name__}: {e}")
        return {}
    allowed = [b.article for b in legal_basis]
    a = audit(clause, raw, allowed, precedents)
    if not a["format_ok"] or a["bad_cited"] or a["bad_mention"] or a["bad_case_no"]:
        logger.info(f"  설명 폐기(기존 설명 유지): {a}")
        return {}
    out: dict[str, Any] = {"reasoning": (raw["explanation"].strip() + (f" 수정 제안: {raw['suggestion'].strip()}" if str(raw.get("suggestion", "")).strip() else ""))}
    if a["span_verbatim"]:
        out["evidence_span"] = raw["evidence_span"].strip()
    return out
