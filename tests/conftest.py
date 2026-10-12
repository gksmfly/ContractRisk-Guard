# tests/conftest.py
"""테스트 공통 설정.

## 문서 종류 판별(라우터)을 기본으로 끈다 (2026-10-08)

`backend/utils.py`가 프로젝트 루트의 `.env`를 읽으므로 로컬 `.env`의 `MODEL_DIR`이 테스트에도 들어온다.
`MODEL_DIR=models/article_v3`(라우터 있음)이면 `run_analyze`가 실제 라우터를 읽어 문서 종류를 조항 처리에
넘기고, 결과가 **개발자 PC의 `.env`에 따라 달라진다.** API·동시성 테스트는 라우팅과 무관하므로 여기서 끈다.
라우팅 자체는 `tests/agents/test_judgment_agent.py::TestDocStyleRouting`이 따로 검사한다.
"""

import os

import pytest

# ## 모델 이름 환경변수를 테스트가 정한다 (2026-10-12)
#
# `FORWARD_MODEL`·`VERIFY_MODEL`은 모듈 최상단에서 **import 시점에** 읽힌다(`lazy_env`).
# 로컬에는 `.env`가 있어 값이 채워지지만 CI에는 `.env`가 없어, `run_forward`가 맨 앞의
# `require_env`에서 RuntimeError로 멈췄다 — `test_api_errors`의 두 테스트가 CI에서 계속 실패했다.
# 반대로 개발자 PC의 `.env`가 `VERIFY_MODEL=local:qwen2.5-14b`면 기본 모델로 부르는 테스트가
# 진짜 로컬 모델을 올리려 든다. 어느 쪽이든 테스트 결과가 환경에 좌우되면 안 되므로 여기서 고정한다.
# conftest는 테스트 모듈보다 먼저 로드되고 `.env`는 기존 값을 덮어쓰지 않으므로 이 값이 이긴다.
os.environ["FORWARD_MODEL"] = "test-model"
os.environ["VERIFY_MODEL"] = "test-model"
os.environ.pop("EXPLAIN_MODEL", None)


@pytest.fixture(autouse=True)
def _no_doc_style_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("backend.api.services.analyze.route_document", lambda clauses: None)
