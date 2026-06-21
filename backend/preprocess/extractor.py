# backend/preprocess/extractor.py
import re
from typing import Any

from backend.preprocess.cleaner import clean_text, clean_precedent_content

_LAW_HEADER = re.compile(r"^제\d+장\s")


def extract_law(doc: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    law  = doc.get("법령", {})
    info = law.get("기본정보", {})
    articles = law.get("조문", {}).get("조문단위", [])
    if isinstance(articles, dict):
        articles = [articles]

    base_meta = {
        "law_name":   info.get("법령명_한글", ""),
        "law_id":     info.get("법령ID", ""),
        "enacted_at": info.get("시행일자", ""),
    }

    results = []
    for a in articles:
        text = clean_text(str(a.get("조문내용", "")))
        if not text:
            continue
        if _LAW_HEADER.match(text) or text.startswith("부칙"):
            continue
        meta = {
            **base_meta,
            "article_no":    str(a.get("조문번호", "")),
            "article_title": str(a.get("조문제목", "")),
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
