# backend/domain/filter_interpretation.py
import json
from pathlib import Path
from typing import Any

from backend.domain.common import logger, copy_domain_docs

TARGET_LAWS: tuple[str, ...] = (
    "약관의 규제에 관한 법률",
    "소비자기본법",
    "전자상거래 등에서의 소비자보호에 관한 법률",
    "방문판매 등에 관한 법률",
    "할부거래에 관한 법률",
    "약관규제",
)


def _is_domain(doc: dict) -> bool:
    svc  = doc.get("ExpcService", {})
    name = svc.get("안건명", "")
    return any(law in name for law in TARGET_LAWS)


def filter_interpretations(
    expc_dir: Path,
    dst_dir: Path,
    device: str = "cuda:1",
) -> dict[str, Any]:
    total = sum(1 for _ in expc_dir.glob("*.json"))
    logger.info(f"  [해석례] 안건명 법령명 필터링 시작 (전체 {total:,}건)")

    candidate_ids: set[str] = set()
    for fp in expc_dir.glob("*.json"):
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
            if _is_domain(doc):
                candidate_ids.add(fp.stem)
        except Exception:
            pass

    logger.info(f"  [해석례] 안건명 매칭: {len(candidate_ids):,}건")
    copied = copy_domain_docs(expc_dir, dst_dir, candidate_ids)
    logger.info(f"  [해석례] 완료: {total:,}건 → {copied}건 복사 → {dst_dir}")
    return {
        "방법":        "안건명_법령명_메타데이터",
        "전체_문서수":  total,
        "안건명_선택":  len(candidate_ids),
        "복사_완료":    copied,
    }
