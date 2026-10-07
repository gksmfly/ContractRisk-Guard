# eval/requirement_cards.py
"""요건 카드 생성 — 약관규제법 제6~14조를 호 단위로 (사전 등록: eval/prereg_requirement_verification.md §4)

`text`는 `data/raw/law/260021.json`(국가법령정보센터 원문)에서 **그대로** 가져온다. 앞의 번호("1.  ")만 뗀다.
`check`는 원문을 예/아니오 질문 하나로 바꾼 것이며 **사람이 쓴 초안**이다 — 원문의 한정어
("상당한 이유 없이", "부당하게")를 빼지 않았는지가 검수 포인트다.

    .venv/bin/python -m eval.requirement_cards          # data/eval/requirement_cards.json 생성 + 원문 일치 검증
"""

import json
import re

from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("requirement_cards.log")

LAW_PATH = PROJECT_ROOT / "data/raw/law/260021.json"
OUT_PATH = PROJECT_ROOT / "data/eval/requirement_cards.json"
ARTICLE_NOS = [str(n) for n in range(6, 15)]

# (조, 호) → 판단 질문. 호가 없는 조(8·13)는 "본문", 제6조 ①항은 "①항"을 키로 쓴다.
CHECKS: dict[tuple[str, str], str] = {
    ("제6조", "①항"): "이 조항은 신의성실의 원칙을 위반하여 공정성을 잃은 조항인가?",
    ("제6조", "1호"): "이 조항은 고객에게 부당하게 불리한가?",
    ("제6조", "2호"): "이 조항은 계약의 거래형태 등 관련된 모든 사정에 비추어 고객이 예상하기 어려운가?",
    ("제6조", "3호"): "이 조항은 계약의 목적을 달성할 수 없을 정도로 계약에 따르는 본질적 권리를 제한하는가?",
    ("제7조", "1호"): "이 조항은 사업자·이행 보조자·피고용자의 고의 또는 중대한 과실로 인한 법률상의 책임을 배제하는가?",
    ("제7조", "2호"): "이 조항은 상당한 이유 없이 사업자의 손해배상 범위를 제한하거나 사업자가 부담할 위험을 고객에게 떠넘기는가?",
    ("제7조", "3호"): "이 조항은 상당한 이유 없이 사업자의 담보책임을 배제·제한하거나 그에 따르는 고객의 권리행사 요건을 가중하는가?",
    ("제7조", "4호"): "이 조항은 상당한 이유 없이, 견본이나 품질·성능 표시로 보장된 내용에 대한 책임을 배제·제한하는가?",
    ("제8조", "본문"): "이 조항은 고객에게 부당하게 과중한 지연 손해금 등의 손해배상 의무를 부담시키는가?",
    ("제9조", "1호"): "이 조항은 법률에 따른 고객의 해제권 또는 해지권을 배제하거나 그 행사를 제한하는가?",
    ("제9조", "2호"): "이 조항은 사업자에게 법률에 없는 해제권 또는 해지권을 부여하여 고객에게 부당하게 불이익을 줄 우려가 있는가?",
    ("제9조", "3호"): "이 조항은 법률에 따른 사업자의 해제권·해지권 행사 요건을 완화하여 고객에게 부당하게 불이익을 줄 우려가 있는가?",
    ("제9조", "4호"): "이 조항은 해제·해지에 따른 원상회복의무를 상당한 이유 없이 고객에게 과중하게 부담시키거나 고객의 원상회복 청구권을 부당하게 포기시키는가?",
    ("제9조", "5호"): "이 조항은 해제·해지에 따른 사업자의 원상회복의무나 손해배상의무를 부당하게 경감하는가?",
    ("제9조", "6호"): "이 조항은 계속적 채권관계 계약에서 존속기간을 부당하게 단기 또는 장기로 하거나 묵시적 연장·갱신이 가능하도록 정하여 고객에게 부당하게 불이익을 줄 우려가 있는가?",
    ("제10조", "1호"): "이 조항은 상당한 이유 없이 급부의 내용을 사업자가 일방적으로 결정하거나 변경할 수 있게 하는가?",
    ("제10조", "2호"): "이 조항은 상당한 이유 없이 사업자가 이행할 급부를 일방적으로 중지할 수 있게 하거나 제3자에게 대행할 수 있게 하는가?",
    ("제11조", "1호"): "이 조항은 고객의 항변권·상계권 등 법률상 권리를 상당한 이유 없이 배제하거나 제한하는가?",
    ("제11조", "2호"): "이 조항은 고객에게 주어진 기한의 이익을 상당한 이유 없이 박탈하는가?",
    ("제11조", "3호"): "이 조항은 고객이 제3자와 계약을 체결하는 것을 부당하게 제한하는가?",
    ("제11조", "4호"): "이 조항은 사업자가 업무상 알게 된 고객의 비밀을 정당한 이유 없이 누설하는 것을 허용하는가?",
    ("제12조", "1호"): "이 조항은 일정한 작위·부작위가 있을 때 고객의 의사표시가 표명되거나 표명되지 않은 것으로 보면서, 그 뜻을 명확히 따로 고지하지 않는가(또는 고지 불가 사유도 없는가)?",
    ("제12조", "2호"): "이 조항은 고객의 의사표시 형식이나 요건에 부당하게 엄격한 제한을 두는가?",
    ("제12조", "3호"): "이 조항은 고객의 이익에 중대한 영향을 미치는 사업자의 의사표시가 상당한 이유 없이 고객에게 도달된 것으로 보는가?",
    ("제12조", "4호"): "이 조항은 고객의 이익에 중대한 영향을 미치는 사업자의 의사표시 기한을 부당하게 길게 또는 불확정하게 정하는가?",
    ("제13조", "본문"): "이 조항은 고객의 대리인이 계약을 체결한 경우, 고객이 의무를 이행하지 않으면 대리인에게 그 의무의 전부 또는 일부를 이행할 책임을 지우는가?",
    ("제14조", "1호"): "이 조항은 고객에게 부당하게 불리한 소송 제기 금지 조항 또는 재판관할 합의 조항인가?",
    ("제14조", "2호"): "이 조항은 상당한 이유 없이 고객에게 입증책임을 부담시키는가?",
}

_NUM = re.compile(r"^\s*\d+\.\s+")


def _as_list(v) -> list:
    return v if isinstance(v, list) else ([v] if v else [])


def build() -> list[dict]:
    units = json.loads(LAW_PATH.read_text(encoding="utf-8"))["법령"]["조문"]["조문단위"]
    cards = []
    for u in units:
        if u.get("조문여부") != "조문" or str(u.get("조문번호")) not in ARTICLE_NOS:
            continue
        art = f"제{u['조문번호']}조"
        lead = u["조문내용"].strip()
        items = []
        hang = _as_list(u.get("항"))
        if not hang:
            # 호가 없는 조는 본문 전체가 요건이다 (제8조·제13조)
            items.append({"ho": "본문", "text": lead})
        for h in (hang if "항번호" in (hang[0] if hang else {}) else []):
            if "호" not in h:
                items.append({"ho": f"{h['항번호']}항", "text": h["항내용"].strip()})
            else:
                lead = f"{lead} {h['항내용'].strip()}".strip()   # 제6조: 제목 + ②항 도입문
                for ho in _as_list(h["호"]):
                    items.append({"ho": f"{int(ho['호번호'].rstrip('.'))}호", "text": _NUM.sub("", ho["호내용"]).strip()})
        # 제7·9·10~12·14조: 항 없이 호만 붙는다
        if hang and "항번호" not in hang[0]:
            for ho in _as_list(hang[0].get("호")):
                items.append({"ho": f"{int(ho['호번호'].rstrip('.'))}호", "text": _NUM.sub("", ho["호내용"]).strip()})
        for it in items:
            it["check"] = CHECKS[(art, it["ho"])]
        cards.append({"article": art, "title": u["조문제목"], "lead": lead, "items": items})
    return cards


def verify(cards: list[dict]) -> None:
    """모든 `text`가 원문에 그대로 들어 있고, 모든 `check`가 카드와 1:1인지 확인."""
    raw = re.sub(r"\s+", " ", LAW_PATH.read_text(encoding="utf-8"))
    used = set()
    for c in cards:
        for it in c["items"]:
            assert re.sub(r"\s+", " ", it["text"]) in raw, f"원문 불일치: {c['article']} {it['ho']}"
            used.add((c["article"], it["ho"]))
    assert used == set(CHECKS), f"check 불일치: {used ^ set(CHECKS)}"


if __name__ == "__main__":
    cards = build()
    verify(cards)
    save_json(cards, OUT_PATH)
    logger.info(f"요건 카드 {len(cards)}개 조, {sum(len(c['items']) for c in cards)}개 요건 → {OUT_PATH}")
    for c in cards:
        print(c["article"], c["title"], [i["ho"] for i in c["items"]])
