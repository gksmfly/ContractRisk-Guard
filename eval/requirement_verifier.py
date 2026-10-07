# eval/requirement_verifier.py
"""요건 검증기 V1 / 통제군 A1 — 프롬프트 초안 (사전 등록: eval/prereg_requirement_verification.md)

두 변형은 **요건 카드를 주느냐 하나만 다르다.** 시스템 프롬프트의 판정 규칙·입력 형식·출력 스키마를 공유한다.

    V1  forward가 붙인 각 조에 대해 요건 카드(호 원문 + 판단 질문)를 주고 호별로 판정
    A1  같은 구조에서 카드를 빼고 조 번호·제목만 준다. 조 단위로 판정(호가 없으니 호별 판정은 불가)

설계 선택 (검수 대상):
  - 프롬프트는 forward가 이 조를 붙였다는 사실을 말하지 않는다(확인 편향 방지). 대상 조를 "검토 대상"으로만 준다.
  - 출력은 reason을 verdict **앞에** 둔다(근거를 쓴 뒤 판정). 계획서 스키마와 키 순서만 다르다.
  - 퓨샷 없음. 예시를 넣으면 gold 누설 위험이 있고, 비-gold 예시를 고르는 일 자체가 튜닝이다.
  - 입력은 조항 전문 C + 근거 문구 E (§4: dev 비교 없이 C+E로 고정).

    .venv/bin/python -m eval.requirement_verifier --dry-run    # 프롬프트 렌더링만 출력, API 0
"""

import json
import re

from backend.utils import PROJECT_ROOT

CARDS_PATH = PROJECT_ROOT / "data/eval/requirement_cards.json"
PROMPT_VERSION = "req-v2-draft"
MAX_CLAUSE_CHARS = 3000      # 기존 forward/verify와 같은 상한
VERDICTS_V1 = ("충족", "불충족", "판단불가")
VERDICTS_A1 = ("해당", "비해당", "판단불가")

_RULES = """## 판단 규칙

- 입력은 약관 조항 전문, 그 조항에서 문제가 될 수 있는 근거 문구, 검토 대상 조입니다.
  근거 문구는 참고용이며, 판단은 조항 전문의 문언에 근거해야 합니다.
- {unit}를 서로 독립적으로 판단하세요. 하나가 {yes}이라고 해서 다른 것까지 {yes}으로 표시하면 안 됩니다.
- "부당하게", "상당한 이유 없이" 같은 한정어는 요건의 일부입니다. 조항에 사업자 이익을 정당화하는 사정이나
  고객 보호 장치(단서, 사전 고지, 합리적 기한 등)가 보이면 판단에 반영하세요.
- 조항의 실질이 요건에 해당하는지를 보세요. 용어나 주제가 비슷하다는 이유만으로 {yes}으로 표시하면 안 됩니다.
- {yes}: 조항의 문언에서 요건에 해당함을 직접 확인할 수 있음.
  {no}: 조항의 문언에 요건이 나타나지 않거나 요건과 반대임.
  판단불가: 조항만으로는 판단할 수 없음(법률의 내용이나 거래 관행을 알아야 하는 경우 등).
- reason은 조항에서 직접 인용한 구절을 포함해 한 문장으로 쓰세요.
- 반드시 아래 JSON 형식으로만 응답하세요."""

_SYSTEM_V1 = (
    "당신은 한국 약관 심사 전문가입니다. 「약관의 규제에 관한 법률」의 특정 조에 대해, "
    "주어진 약관 조항이 그 조의 각 요건에 해당하는지 호별로 판단합니다.\n\n"
    + _RULES.format(unit="각 호", yes="충족", no="불충족")
    + """

{
  "article": "<검토 대상 조 번호>",
  "items": [
    {"ho": "<요건 이름>", "reason": "...", "verdict": "충족" 또는 "불충족" 또는 "판단불가"}
  ]
}
article에는 검토 대상에 적힌 조 번호를 쓰세요. items에는 검토 대상에 나열된 모든 요건을 입력된 순서대로 한 번씩 포함하고,
ho에는 입력에 적힌 요건 이름(예: 1호, ①항, 본문)을 그대로 쓰세요. 요건을 빼거나 이름을 바꾸면 안 됩니다."""
)

_SYSTEM_A1 = (
    "당신은 한국 약관 심사 전문가입니다. 「약관의 규제에 관한 법률」의 특정 조에 대해, "
    "주어진 약관 조항이 그 조에 해당하는지 판단합니다. 조문의 내용은 제공되지 않으니 "
    "당신이 알고 있는 해당 조의 내용으로 판단하세요.\n\n"
    + _RULES.format(unit="검토 대상 조", yes="해당", no="비해당")
    + """

{
  "article": "<검토 대상 조 번호>",
  "reason": "...",
  "verdict": "해당" 또는 "비해당" 또는 "판단불가"
}
article에는 검토 대상에 적힌 조 번호를 쓰세요."""
)


def load_cards() -> dict[str, dict]:
    return {c["article"]: c for c in json.loads(CARDS_PATH.read_text(encoding="utf-8"))}


def _render_card(card: dict) -> str:
    lines = [f"「약관의 규제에 관한 법률」 {card['lead']}", ""]
    for it in card["items"]:
        lines += [f"- {it['ho']}: {it['text']}", f"  판단 질문: {it['check']}"]
    return "\n".join(lines)


def build_messages(variant: str, clause: str, evidence: str, article: str, cards: dict[str, dict]) -> list[dict]:
    """한 (조항, 조) 쌍에 대한 메시지. forward가 붙인 조마다 한 번씩 호출한다."""
    if variant not in ("V1", "A1"):
        raise ValueError(variant)
    card = cards[article]
    target = _render_card(card) if variant == "V1" else f"「약관의 규제에 관한 법률」 {article}({card['title']})"
    user = (f"## 조항 전문\n{clause[:MAX_CLAUSE_CHARS]}\n\n"
            f"## 근거 문구\n{evidence or '(없음)'}\n\n"
            f"## 검토 대상\n{target}")
    return [{"role": "system", "content": _SYSTEM_V1 if variant == "V1" else _SYSTEM_A1},
            {"role": "user", "content": user}]


def parse(variant: str, text: str, article: str, cards: dict[str, dict]) -> dict | None:
    """응답 검증. 스키마·호 목록·판정값이 어긋나면 None(재시도 대상)."""
    try:
        m = re.search(r"\{.*\}", text, re.S)
        out = json.loads(m.group(0)) if m else None
    except (ValueError, AttributeError):
        return None
    if not isinstance(out, dict) or not _same_article(out.get("article"), article):
        return None
    if variant == "A1":
        return out if out.get("verdict") in VERDICTS_A1 else None
    want = [it["ho"] for it in cards[article]["items"]]
    items = out.get("items")
    if not isinstance(items, list) or [i.get("ho") for i in items] != want:
        return None
    return out if all(i.get("verdict") in VERDICTS_V1 for i in items) else None


def _same_article(got, want: str) -> bool:
    """`article` 필드만 느슨하게 비교한다 — "제6조"/"6조"/"6" 모두 같은 조. 호 이름(`ho`)에는 쓰지 않는다."""
    m = re.search(r"\d+", str(got))
    return bool(m) and m.group(0) == re.search(r"\d+", want).group(0)


def keep(variant: str, parsed: dict | None) -> bool:
    """이 조를 유지하는가 (사전 등록 §9.1·§9.2).

    V1: '충족' 또는 '판단불가'인 호가 하나라도 있으면 유지(확신이 없으면 버리지 않는다). 모든 호가 '불충족'일 때만 제거.
    A1: '해당' 또는 '판단불가'면 유지. '비해당'만 제거.
    파싱 실패(parsed가 None): **유지** — 필터를 적용하지 않고 실패 건수를 따로 보고한다.
    """
    if parsed is None:
        return True
    if variant == "A1":
        return parsed["verdict"] != "비해당"
    return any(i["verdict"] != "불충족" for i in parsed["items"])


def _dry_run() -> None:
    from backend.training.train_article import load_ftc_gold
    from backend.utils import load_jsonl
    cards = load_cards()
    gold_text = {g["text"].strip() for g in load_ftc_gold("clean")}
    rec = next(r for r in load_jsonl(PROJECT_ROOT / "data/fb_check/clean.jsonl")
               if r.get("source") == "ftc_case" and "제9조" in (r.get("forward_articles") or [])
               and r["text"].strip() not in gold_text)
    for v in ("V1", "A1"):
        msgs = build_messages(v, rec["text"], rec.get("evidence_span", ""), "제9조", cards)
        print(f"\n{'=' * 30} {v} ({PROMPT_VERSION}) {'=' * 30}")
        for m in msgs:
            print(f"--- {m['role']} ---\n{m['content']}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    if ap.parse_args().dry_run:
        _dry_run()
