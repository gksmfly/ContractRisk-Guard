# backend/preprocess/extractor.py
import re
from typing import Any

from backend.preprocess.cleaner import clean_precedent_content, clean_text

_LAW_HEADER = re.compile(r"^제\d+장\s")


def _as_list(x: Any) -> list:
    """법령 API는 항목이 하나면 dict, 여럿이면 list로 준다 — 항상 list로 맞춘다."""
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def article_text(a: dict[str, Any]) -> str:
    """조문 하나를 본문 + 항 + 호 + 목까지 합쳐 한 덩어리 텍스트로 만든다.

    예전에는 `조문내용`(본문)만 썼다. 그런데 약관규제법 제6~14조처럼 **규범 내용이 전부
    각 호에 들어 있는** 조문이 많아, 본문만 담으면 껍데기만 색인된다:

        제6조(일반원칙)                                  ← 실제 적재된 전체 텍스트(9자)
        제7조(면책조항의 금지) ... 조항은 무효로 한다.     ← 무엇이 무효인지가 빠짐

    도메인 법령 3,449개 조문 기준 조문당 59자 → 205자(3.5배)로 늘고,
    120자 미만 조문 비율이 89.3% → 47.9%로 떨어진다. 검색이 조문을 정확히 찾아와도
    돌려주는 내용에 규범이 없으면 판단 근거로 쓸 수 없다.

    각 항목을 따로 `clean_text`한 뒤 개행으로 잇는다 — `clean_text`가 공백을 전부
    한 칸으로 접기 때문에, 먼저 합치면 호 사이 경계가 사라진다.
    """
    parts = [clean_text(str(a.get("조문내용", "")))]
    for h in _as_list(a.get("항")):
        if not isinstance(h, dict):
            parts.append(clean_text(str(h)))
            continue
        parts.append(clean_text(str(h.get("항내용", ""))))
        for ho in _as_list(h.get("호")):
            if not isinstance(ho, dict):
                parts.append(clean_text(str(ho)))
                continue
            parts.append(clean_text(str(ho.get("호내용", ""))))
            for mok in _as_list(ho.get("목")):
                parts.append(clean_text(str(mok.get("목내용", "")) if isinstance(mok, dict) else str(mok)))
    return "\n".join(p for p in parts if p)


def extract_law(doc: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    law  = doc.get("법령", {})
    info = law.get("기본정보", {})
    articles = _as_list(law.get("조문", {}).get("조문단위", []))

    base_meta = {
        "law_name":   info.get("법령명_한글", ""),
        "law_id":     info.get("법령ID", ""),
        "enacted_at": info.get("시행일자", ""),
    }

    results = []
    for a in articles:
        if not isinstance(a, dict):
            continue
        head = clean_text(str(a.get("조문내용", "")))
        if not head:
            continue
        if _LAW_HEADER.match(head) or head.startswith("부칙"):
            continue
        text = article_text(a)
        # 조문가지번호: "제19조의2"의 "2". 이걸 안 담으면 제19조·제19조의2·제19조의3이
        # 모두 article_no="19"로 뭉개져 **인용이 틀린다**(상법 제287조는 가지 조문이 54개다).
        # 기존 지표·매칭이 article_no 기준이라 그 값은 그대로 두고, 가지번호와 표시용
        # 라벨을 따로 추가한다.
        branch = str(a.get("조문가지번호") or "").strip()
        article_no = str(a.get("조문번호", ""))
        meta = {
            **base_meta,
            "article_no":     article_no,
            "article_branch": branch,
            "article_label":  f"제{article_no}조의{branch}" if branch else (f"제{article_no}조" if article_no else ""),
            "article_title":  str(a.get("조문제목", "")),
            # 조문이 길어 여러 청크로 쪼개질 때 각 조각 앞에 다시 붙일 머리말.
            # 조각만 검색돼도 "몇 조인지" 알 수 있어야 한다.
            "article_head":   head,
        }
        results.append((text, meta))
    return results


def extract_precedent(doc: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    svc = doc.get("PrecService", {})
    base_meta = {
        "case_name":   svc.get("사건명", ""),
        "case_number": svc.get("사건번호", ""),
        "decided_at":  svc.get("선고일자", ""),
        "court":       svc.get("법원명", ""),
    }

    results = []
    for section in ["판시사항", "판결요지", "판례내용"]:
        raw = str(svc.get(section, ""))
        text = clean_precedent_content(raw) if section == "판례내용" else clean_text(raw)
        if not text:
            continue
        meta = {**base_meta, "section": section}
        results.append((text, meta))
    return results


def extract_interpretation(doc: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    svc  = doc.get("ExpcService", {})
    text = " ".join(
        clean_text(str(svc.get(f, "")))
        for f in ["회답", "이유"] if svc.get(f)
    )
    if not text:
        return []
    meta = {
        "title":       svc.get("안건명", ""),
        "number":      svc.get("안건번호", ""),
        "issued_at":   svc.get("해석일자", ""),
        "institution": svc.get("해석기관명", ""),
    }
    return [(text, meta)]


EXTRACTORS: dict[str, Any] = {
    "law":            extract_law,
    "precedent":      extract_precedent,
    "interpretation": extract_interpretation,
}
