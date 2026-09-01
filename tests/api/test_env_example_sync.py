# tests/api/test_env_example_sync.py
"""`.env.example`과 서버 필수 변수 목록이 어긋나지 않게 고정한다.

손으로 맞추면 언젠가 어긋난다 — 실제로 어긋나 있었다. `_REQUIRED_ENV_VARS`가
`OPENAI_API_KEY`를 요구하는데 `.env`·`.env.example` 어디에도 없어서,
`docker compose`(env_file 사용)로 띄우면 기동 거부되는 상태였다.
로컬 셸에 값이 있어 개발 중에는 드러나지 않았다.
"""

import re

from backend.api.server import _REQUIRED_ENV_VARS
from backend.utils import PROJECT_ROOT

_KEY = re.compile(r"^([A-Z_]+)=", re.MULTILINE)


def _example_keys() -> set[str]:
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    return set(_KEY.findall(text))


def test_env_example_exists() -> None:
    """README의 `cp .env.example .env`가 실제로 동작해야 한다."""
    assert (PROJECT_ROOT / ".env.example").is_file()


def test_required_vars_are_documented() -> None:
    """서버가 요구하는 변수는 전부 예시 파일에 (주석이 아닌 실제 줄로) 있어야 한다."""
    missing = sorted(set(_REQUIRED_ENV_VARS) - _example_keys())
    assert not missing, f".env.example에 없는 필수 변수: {missing}"


def test_env_example_is_docker_env_file_compatible() -> None:
    """예시 파일이 **`docker run --env-file`에서도** 읽히는 형식이어야 한다.

    python-dotenv와 `docker compose`는 `KEY = value`를 받아주는데 `docker run --env-file`은
    거부한다. 예시 파일이 관대한 형식을 가르치면 사용자가 `.env`에 그대로 옮겨 적고,
    **로컬에서는 되는데 컨테이너에서만 죽는다.** 실제로 이 저장소의 `.env`가 그 상태였다.
    """
    from backend.scripts.check_env_file import check
    problems = check(PROJECT_ROOT / ".env.example")
    assert not problems, "`.env.example` 형식 문제: " + "; ".join(problems)
