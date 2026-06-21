# backend/domain/common.py
import shutil
from pathlib import Path
from typing import Any

from backend.utils import load_logger

logger = load_logger("domain.log")


"""raw 디렉토리에서 해당 문서 ID의 JSON 파일만 domain 디렉토리로 복사한다.

Args:
    src_dir (Path): 원본 raw 디렉토리
    dst_dir (Path): 복사 대상 domain 디렉토리
    doc_ids (set[str]): 복사할 문서 ID 집합

Returns:
    int: 복사된 파일 수
"""

def copy_domain_docs(src_dir: Path, dst_dir: Path, doc_ids: set[str]) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for doc_id in doc_ids:
        src = src_dir / f"{doc_id}.json"
        if src.exists():
            shutil.copy2(src, dst_dir / f"{doc_id}.json")
            copied += 1
        else:
            logger.warning(f"  파일 없음: {src}")
    return copied
