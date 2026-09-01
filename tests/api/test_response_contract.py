# tests/api/test_response_contract.py
"""응답 JSON의 **키 이름**을 고정한다. 이름이 바뀌면 소비자가 조용히 NULL을 받는다.

2026-09-01에 실제로 겪었다 — 일괄 rename이 함수명 `get_model_version`을 스키마 필드까지
끌고 가서 응답 키가 `get_model_version`이 됐는데, 대시보드는 `result->>'model_version'`을
읽고 있었다. **화면에는 빈칸으로 렌더되므로 아무도 몰랐고**, `judgment_agent`의 설명 주석까지
같이 바뀌어 있어서 문서를 봐도 못 찾았다.

타입 검사·기존 테스트는 전부 통과한다 — 양쪽 다 `str`이고 필드가 존재하기는 하기 때문이다.
그래서 **이름 자체**를 검사한다.
"""

from backend.api.schemas import AnalyzeResponse, ClauseResult

# 저장된 판정의 세대를 가르는 유일한 수단. `analyses.result`(JSONB)에 이 키로 들어간다.
_RESPONSE_KEYS = {"total_clauses", "review_count", "clauses", "model_version",
                  "input_clauses", "truncated_clauses", "out_of_scope"}
_CLAUSE_KEYS = {"id", "original", "articles", "needs_review", "evidence_spans",
                "legal_basis", "precedent_refs", "reasoning", "verified"}


def test_response_field_names_are_pinned() -> None:
    missing = sorted(_RESPONSE_KEYS - set(AnalyzeResponse.model_fields))
    assert not missing, (
        f"AnalyzeResponse에서 사라진 키: {missing}. 소비자(대시보드 SQL·프론트)가 이 이름으로 "
        f"읽는다 — 이름을 바꾸면 조용히 NULL을 받는다")


def test_clause_field_names_are_pinned() -> None:
    missing = sorted(_CLAUSE_KEYS - set(ClauseResult.model_fields))
    assert not missing, f"ClauseResult에서 사라진 키: {missing}"


def test_model_version_is_not_prefixed_with_get() -> None:
    """함수명(`get_model_version`)이 필드로 새어 나오는 것을 막는다 — 실제로 있었던 사고다."""
    assert "get_model_version" not in AnalyzeResponse.model_fields
