# backend/utils.py
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ContractRisk-Guard/


# ─────────────────────────────────────────────────────────────────────────────
# 환경변수 — **모듈 최상단에서 `os.environ["X"]`를 쓰지 말 것**
# ─────────────────────────────────────────────────────────────────────────────
#
# 서빙 import 체인이 오프라인 라벨링 모듈을 지나간다:
#
#     server → routers.analyze → services.analyze → agents.graph
#            → analysis_agent → fb_check.forward_labeling
#                               └ 오프라인 전용 모듈
#
# 논문 설계상 의도된 재사용이다(Analysis Agent가 `run_forward`를 그대로 쓴다).
# 그래서 **오프라인 모듈의 제약이 서빙으로 샌다.** 최상단에서 `os.environ["X"]`를 읽으면
# 그 변수가 없는 배포에서 `import backend.api.server`가 `os.py` 깊은 곳의 raw `KeyError`로
# 죽고, 친절한 에러를 주려고 만든 `_validate_required_env()`는 **돌지도 못한다.**
#
# 실제로 `FORWARD_MODEL`이 그랬다(2026-09-01). 그때 `VERIFY_MODEL`은 체인 밖이라 안 걸렸을
# 뿐이고 — 체인은 오늘도 바뀌었다(게이트를 GPT에서 모델로 옮기면서). 그러니 인스턴스가
# 아니라 **부류를 막는다**: 최상단에서는 `lazy_env()`로 읽고, 실제로 쓰는 자리에서
# `require_env()`로 검사한다. `tests/api/test_env_contract.py`가 이 규칙을 강제한다.


def lazy_env(name: str, default: str = "") -> str:
    """모듈 최상단용 환경변수 읽기. **없어도 import를 깨뜨리지 않는다.**

    값이 비어 있을 수 있으므로, 실제로 쓰는 자리에서 반드시 `require_env()`를 통과시킬 것.
    """
    return os.environ.get(name, default)


def require_env(value: str, name: str, used_by: str) -> str:
    """사용 시점 검사. 비어 있으면 **무엇이·어디에 필요한지** 말해주고 멈춘다.

    `os.environ["X"]`의 원래 의도("`.env`에 없으면 즉시 에러")는 그대로 지키되, 터지는
    시점만 import에서 사용으로 옮긴다 — 그래야 서버는 기동 시점에
    `_validate_required_env()`로 한꺼번에 잡아 목록을 보여줄 수 있다.
    """
    if not value:
        raise RuntimeError(
            f"환경변수 {name}이(가) 설정되지 않았습니다 — {used_by}에 필요합니다. "
            f"`.env.example`을 참고해 `.env`에 추가하세요."
        )
    return value


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
