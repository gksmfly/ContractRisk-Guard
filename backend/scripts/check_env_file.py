# backend/scripts/check_env_file.py
"""`.env`가 **`docker run --env-file`에서도** 읽히는 형식인지 검사한다. 값은 절대 찍지 않는다.

## 왜 필요한가 — 세 파서가 서로 다르게 관대하다

    python-dotenv          `KEY = value` 를 받아준다 (양쪽 공백을 벗긴다)
    docker compose         받아준다 (compose-go/dotenv, godotenv 포팅이라 같이 벗긴다)
    docker run --env-file  **받아주지 않는다** — 키에 공백이 있으면 거부하거나
                           `KEY `(뒤에 공백)라는 다른 변수로 넣는다

그래서 로컬 uvicorn과 `docker compose up`은 되는데 `docker run --env-file .env`만 죽는다.
그리고 죽는 모양이 "`OPENAI_API_KEY`가 없습니다"라서 **환경변수를 안 넣은 것처럼 보인다** —
실제로는 넣었고 등호 앞에 공백 하나가 있을 뿐이다.

이 프로젝트가 이미 같은 계열로 두 번 데었다(`FORWARD_MODEL` import-time `os.environ[...]`,
`.env.example` 부재). 셋 다 **"로컬에서는 되는데 컨테이너에서만 안 되는"** 형태다.

## 무엇을 검사하나

    등호 앞뒤 공백        `KEY = v` · `KEY= v`(값 앞 공백은 키에 붙는 건 아니지만 값에 붙는다)
    키에 허용되지 않는 문자
    따옴표로 감싼 값        docker는 따옴표를 **값의 일부로** 넣는다(dotenv는 벗긴다)
    CRLF 줄바꿈            값 끝에 `\\r`이 붙는다 — API 키에 붙으면 인증이 실패한다

실행:
    .venv/bin/python -m backend.scripts.check_env_file .env
"""

import argparse
import re
import sys
from pathlib import Path

from backend.utils import load_logger

# `print()` 금지 규칙(`Claude.md`)에 따라 logger를 쓴다. **값은 절대 담지 않는다** —
# 시크릿 파일을 읽는 도구라 로그 파일에 값이 새면 안 된다. `check()`가 돌려주는 문제
# 문자열에도 값이 없다(키 이름과 사유만).
logger = load_logger("check_env_file.log")

_KEY_OK = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def check(path: Path) -> list[str]:
    """문제 목록을 돌려준다. **값은 담지 않는다** — 시크릿 파일이다."""
    problems: list[str] = []
    raw = path.read_bytes()
    if b"\r\n" in raw:
        problems.append("CRLF 줄바꿈 — docker가 값 끝에 \\r을 붙인다. LF로 저장할 것")
    for i, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "=" not in line:
            problems.append(f"{i}행: `=`가 없다")
            continue
        key, val = line.split("=", 1)
        if key != key.strip():
            problems.append(f"{i}행 {key.strip()}: **등호 앞에 공백** — "
                            f"`docker run --env-file`이 거부하거나 다른 변수로 넣는다")
        elif not _KEY_OK.match(key):
            problems.append(f"{i}행: 키에 쓸 수 없는 문자")
        if val[:1] in (" ", "\t"):
            problems.append(f"{i}행 {key.strip()}: 등호 뒤 공백이 **값에 포함**된다 "
                            f"(dotenv는 벗기지만 docker는 안 벗긴다)")
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            problems.append(f"{i}행 {key.strip()}: 따옴표가 **값의 일부로** 들어간다")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description="env 파일이 docker --env-file 호환인지 검사 (값 미출력)")
    ap.add_argument("path", nargs="?", default=".env")
    a = ap.parse_args()
    p = Path(a.path)
    if not p.is_file():
        sys.exit(f"{p} 가 없다")
    problems = check(p)
    if not problems:
        logger.info(f"{p}: docker --env-file 호환 OK")
        return
    logger.error(f"{p}: 문제 {len(problems)}건 — `docker run --env-file`에서 실패한다")
    for x in problems:
        logger.error(f"  - {x}")
    logger.warning("로컬 uvicorn과 `docker compose up`은 이 형식을 받아주므로 여기서는 안 드러난다.")
    sys.exit(1)


if __name__ == "__main__":
    main()
