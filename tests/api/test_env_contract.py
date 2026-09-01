# tests/api/test_env_contract.py
"""환경변수 계약 — **서버가 어떤 변수 없이 죽는가**를 고정한다.

## 왜 이 테스트가 따로 필요한가

pytest도 tsc도 이걸 못 잡는다. 테스트가 이미 `.env`가 로드된 환경에서 돌기 때문이다.
`monkeypatch.delenv`로 **명시적으로 지워야** 재현된다.

실제 사고(2026-09-01): `fb_check/forward_labeling.py`가 최상단에서
`os.environ["FORWARD_MODEL"]`을 읽었는데, `agents/analysis_agent.py`가 `run_forward`를
재사용하므로 **서빙 import 체인이 그 줄을 지났다.** 결과:

    FORWARD_MODEL 없는 배포  →  import backend.api.server 가 raw KeyError로 사망
                             →  _validate_required_env()는 돌지도 못함
                             →  README는 "Set: OPENAI_API_KEY, DATABASE_URL"만 안내

`docker compose up`은 `env_file: ../.env`에 값이 있어 떴기 때문에 아무도 몰랐다.

## 세 층으로 막는다

    ① import 가능성   깨끗한 환경에서 서버 모듈이 import 되는가
    ② 실패 방식       기동 검사가 **이름을 말해주는** 에러를 내는가
    ③ 부류 차단       최상단 `os.environ[...]`가 backend/ 전체에 없는가

③이 핵심이다. ①②만 있으면 `FORWARD_MODEL` 하나만 막힌 것이고, 다음에 다른 변수가
체인에 들어오는 순간 같은 사고가 난다. 체인은 실제로 바뀐다 — 2026-08-31에 게이트를
GPT에서 모델로 옮기면서 한 번 바뀌었다.
"""

import ast
import importlib
import pathlib
from pathlib import Path

import pytest

from backend.utils import PROJECT_ROOT, lazy_env, require_env

_SERVING_ENTRYPOINT = "backend.api.server"


def _module_level_env_subscripts(path: pathlib.Path) -> list[int]:
    """import 시점에 실행되는 `os.environ[...]`의 줄 번호.

    함수/람다 **안**은 제외한다 — 호출될 때만 도므로 import를 깨뜨리지 않는다.
    클래스 본문은 import 시점에 실행되므로 **포함**한다.
    """
    hits: list[int] = []

    def walk(node: ast.AST, inside_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            nested = inside_function or isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
            )
            if (
                not nested
                and isinstance(child, ast.Subscript)
                and isinstance(child.value, ast.Attribute)
                and child.value.attr == "environ"
            ):
                hits.append(child.lineno)
            walk(child, nested)

    walk(ast.parse(path.read_text(encoding="utf-8")), False)
    return hits


class TestNoModuleLevelEnvSubscript:
    """③ 부류 차단 — 인스턴스가 아니라 패턴 자체를 금지한다."""

    def test_backend_has_no_import_time_env_subscript(self) -> None:
        offenders = []
        for f in sorted((PROJECT_ROOT / "backend").rglob("*.py")):
            for lineno in _module_level_env_subscripts(f):
                offenders.append(f"{f.relative_to(PROJECT_ROOT)}:{lineno}")

        assert not offenders, (
            "모듈 최상단에서 os.environ[...]를 읽고 있다 — 그 변수가 없는 배포에서 "
            "import 자체가 raw KeyError로 죽는다.\n"
            "  → backend.utils.lazy_env()로 읽고, 쓰는 자리에서 require_env()로 검사할 것.\n"
            "  위반: " + ", ".join(offenders)
        )

    def test_detector_actually_catches_the_pattern(self, tmp_path: Path) -> None:
        """검사기가 실제로 잡는지 확인한다 — 늘 통과하는 테스트는 테스트가 아니다."""
        bad = tmp_path / "bad.py"
        bad.write_text('import os\nX = os.environ["NOPE"]\n', encoding="utf-8")
        assert _module_level_env_subscripts(bad) == [2]

        good = tmp_path / "good.py"
        good.write_text('import os\ndef f():\n    return os.environ["NOPE"]\n', encoding="utf-8")
        assert _module_level_env_subscripts(good) == []


class TestServerImportsWithoutOfflineEnv:
    """① 오프라인 파이프라인 전용 변수가 없어도 서빙 모듈이 import 되어야 한다."""

    @pytest.mark.parametrize("missing", ["FORWARD_MODEL", "VERIFY_MODEL", "HF_TOKEN"])
    def test_import_survives(self, monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
        monkeypatch.delenv(missing, raising=False)
        # 이미 import된 모듈이 캐시를 태우지 않도록 강제로 다시 읽는다
        module = importlib.import_module(_SERVING_ENTRYPOINT)
        importlib.reload(module)
        assert module.app is not None


class TestValidateRequiredEnvNamesTheMissingVar:
    """② 실패 방식 — 무엇이 없는지 말해주는 에러여야 한다."""

    @pytest.mark.parametrize("missing", ["OPENAI_API_KEY", "DATABASE_URL", "FORWARD_MODEL"])
    def test_names_the_variable(self, monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
        from backend.api import server

        monkeypatch.delenv(missing, raising=False)
        with pytest.raises(RuntimeError) as exc:
            server._validate_required_env()
        assert missing in str(exc.value)

    def test_passes_when_all_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from backend.api import server

        for name in server._REQUIRED_ENV_VARS:
            monkeypatch.setenv(name, "dummy")
        server._validate_required_env()      # 예외가 없어야 한다


# `.env.example` ↔ `_REQUIRED_ENV_VARS` 동기화 검사는 여기 두지 않는다 —
# `tests/api/test_env_example_sync.py`가 더 넓게 한다(형식이 `docker run --env-file`에서도
# 읽히는지까지 검사). 같은 검사를 두 군데 두면 한쪽만 고쳐지고 나머지가 낡는다.


class TestEnvHelpers:
    """헬퍼 자체의 계약."""

    def test_lazy_env_returns_empty_instead_of_raising(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SOME_ABSENT_VAR", raising=False)
        assert lazy_env("SOME_ABSENT_VAR") == ""

    def test_require_env_names_variable_and_consumer(self) -> None:
        with pytest.raises(RuntimeError) as exc:
            require_env("", "FORWARD_MODEL", "Forward Labeling")
        assert "FORWARD_MODEL" in str(exc.value)
        assert "Forward Labeling" in str(exc.value)

    def test_require_env_passes_through_explicit_value(self) -> None:
        assert require_env("gpt-4o", "FORWARD_MODEL", "Forward Labeling") == "gpt-4o"
