# backend/utils.py
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ContractRisk-Guard/


def save_json(data: Any, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_jsonl(records: list[dict], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_logger(filename: str) -> logging.Logger:
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
