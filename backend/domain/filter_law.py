# backend/domain/filter_law.py
import json
from pathlib import Path
from typing import Any

from backend.domain.common import logger, copy_domain_docs
from backend.domain.config import ALLOWED_LAW_NAMES


def _is_domain_law(doc: dict[str, Any]) -> bool:
    law_name = doc.get("법령", {}).get("기본정보", {}).get("법령명_한글", "") or ""
    if law_name in ALLOWED_LAW_NAMES:
        return True
    return any(law_name.startswith(base + " 시행") for base in ALLOWED_LAW_NAMES)


def filter_laws(law_dir: Path, dst_dir: Path) -> dict[str, Any]:
    logger.info("  [법령] 법령명 메타데이터 필터링 시작")
    passed: set[str] = set()
    total = 0

    for fp in sorted(law_dir.glob("*.json")):
        total += 1
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
            if _is_domain_law(doc):
                passed.add(fp.stem)
        except Exception as e:
            logger.error(f"  파일 읽기 실패 {fp.name}: {e}")

    copied = copy_domain_docs(law_dir, dst_dir, passed)
    logger.info(f"  [법령] 완료: {total}건 검사 → {copied}건 복사 → {dst_dir}")
    return {"방법": "법령명_메타데이터", "전체_문서수": total, "필터링_통과": len(passed), "복사_완료": copied}
