# backend/scripts/utils.py
"""
크롤링 스크립트 공통 유틸리티

- save_json  : JSON 데이터를 파일로 저장
- setup_logger: 콘솔+파일 핸들러 로거 생성
- PROJECT_ROOT: 프로젝트 루트 Path
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ContractRisk-Guard/


"""JSON 데이터를 파일로 저장한다.

Args:
    data (Any): 저장할 데이터
    filepath (Path): 저장 경로 (부모 디렉터리 없으면 자동 생성)
"""

def save_json(data: Any, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


"""로거를 초기화하고 반환한다.

LOG_DIR 환경변수(기본값: PROJECT_ROOT/data/logs)에 로그 파일을 생성하며
콘솔과 파일 양쪽에 동일 포맷으로 출력한다.

Args:
    filename (str): 저장할 로그 파일명 (예: "crawl_law.log")

Returns:
    logging.Logger: 설정된 로거 인스턴스
"""

def setup_logger(filename: str) -> logging.Logger:
    log_filename = filename if filename.endswith(".log") else f"{filename}.log"
    logger_name  = log_filename.removesuffix(".log")
    logger       = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if logger.handlers: return logger

    log_dir = Path(os.environ.get("LOG_DIR", str(PROJECT_ROOT / "data/logs")))
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_dir / log_filename, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
