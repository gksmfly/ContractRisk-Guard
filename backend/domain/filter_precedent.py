# backend/domain/filter_precedent.py
import json
from pathlib import Path
from typing import Any

from backend.domain.common import logger, copy_domain_docs
from backend.domain.config import PREC_KEYWORDS

EXCLUDE_KINDS: frozenset[str] = frozenset({"형사", "가사", "세무", "특허", "선거,특별"})


def _build_prec_text(doc: dict) -> str:
    svc = doc.get("PrecService", {})
    parts = [svc.get("사건명", ""), svc.get("판시사항", ""), svc.get("판결요지", "")]
    return " ".join(p for p in parts if p).strip()


def _keyword_ids(prec_dir: Path) -> set[str]:
    matched: set[str] = set()
    excluded = 0
    for fp in prec_dir.glob("*.json"):
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
            svc = doc.get("PrecService", {})
            if svc.get("사건종류명", "") in EXCLUDE_KINDS:
                excluded += 1
                continue
            text = _build_prec_text(doc)
            if any(kw in text for kw in PREC_KEYWORDS):
                matched.add(fp.stem)
        except Exception:
            pass
    logger.info(f"  [판례] 사건종류명 제외: {excluded:,}건 (형사/가사/세무/특허)")
    return matched


def filter_precedents(
    prec_dir: Path,
    dst_dir: Path,
    device: str = "cuda:1",
) -> dict[str, Any]:
    total = sum(1 for _ in prec_dir.glob("*.json"))
    logger.info(f"  [판례] 사건종류명 제외 + 키워드 필터링 시작 (전체 {total:,}건)")

    candidate_ids = _keyword_ids(prec_dir)
    logger.info(f"  [판례] 키워드 선택: {len(candidate_ids):,}건")

    copied = copy_domain_docs(prec_dir, dst_dir, candidate_ids)
    logger.info(f"  [판례] 완료: {total:,}건 → {copied}건 복사 → {dst_dir}")
    return {
        "방법":        "사건종류명제외+키워드",
        "전체_판례수":  total,
        "키워드_선택":  len(candidate_ids),
        "복사_완료":    copied,
    }
