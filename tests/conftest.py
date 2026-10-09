# tests/conftest.py
"""테스트 공통 설정.

## 문서 종류 판별(라우터)을 기본으로 끈다 (2026-10-08)

`backend/utils.py`가 프로젝트 루트의 `.env`를 읽으므로 로컬 `.env`의 `MODEL_DIR`이 테스트에도 들어온다.
`MODEL_DIR=models/article_v3`(라우터 있음)이면 `run_analyze`가 실제 라우터를 읽어 문서 종류를 조항 처리에
넘기고, 결과가 **개발자 PC의 `.env`에 따라 달라진다.** API·동시성 테스트는 라우팅과 무관하므로 여기서 끈다.
라우팅 자체는 `tests/agents/test_judgment_agent.py::TestDocStyleRouting`이 따로 검사한다.
"""

import pytest


@pytest.fixture(autouse=True)
def _no_doc_style_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("backend.api.services.analyze.route_document", lambda clauses: None)
