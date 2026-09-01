# tests/test_analyze_unit.py
"""
backend/api/services/analyze.py의 순수 함수 단위 테스트.

외부 의존성(OpenAI API, KoELECTRA GPU 추론, DB)이 없어서 빠르고,
API 키/GPU 없이도 항상 돌아간다.

실행: pytest tests/test_analyze_unit.py
"""

from backend.api.services.analyze import _MAX_CLAUSES, _extract_spans, split_clauses


class TestSplitClauses:
    """`split_clauses`는 (조항 목록, 상한 초과로 잘린 개수)를 반환한다.

    잘린 개수를 같이 돌려주는 이유: 예전에는 상한 20을 그냥 슬라이스해서 30조항
    계약서의 10개가 **응답 어디에도 안 드러난 채** 사라졌다(지연 벤치마크에서
    입력 20과 30의 소요 시간이 같았던 원인). 절삭은 하되 반드시 보고해야 한다.
    """

    def test_empty_text_returns_empty_list(self):
        assert split_clauses("") == ([], 0)

    def test_too_short_text_returns_empty_list(self):
        assert split_clauses("안녕") == ([], 0)

    def test_splits_numbered_articles(self):
        text = (
            "제1조(목적) 이 약관은 서비스 이용에 관한 사항을 규정한다.\n"
            "제2조(정의) 이 약관에서 사용하는 용어의 정의는 다음과 같다."
        )
        clauses, truncated = split_clauses(text)
        assert len(clauses) == 2
        assert truncated == 0
        assert clauses[0].startswith("제1조")
        assert clauses[1].startswith("제2조")

    def test_caps_at_max_clauses_and_reports_truncation(self):
        over = _MAX_CLAUSES + 7
        text = "\n".join(f"제{i}조(조항) 이것은 상한을 넘기는 조항 테스트를 위한 내용입니다." for i in range(1, over + 1))
        clauses, truncated = split_clauses(text)
        assert len(clauses) == _MAX_CLAUSES
        assert truncated == 7, "잘라낸 개수를 보고하지 않으면 조항이 조용히 사라진다"

    def test_realistic_30_clause_contract_is_not_truncated(self):
        """실제 계약서 규모(30조항)가 통째로 분석되는지 — 옛 상한 20의 회귀 방지."""
        text = "\n".join(f"제{i}조(조항) 삼십 개 조항짜리 실제 계약서 규모를 가정한 본문입니다." for i in range(1, 31))
        clauses, truncated = split_clauses(text)
        assert len(clauses) == 30
        assert truncated == 0

    def test_non_korean_text_treated_as_single_clause(self):
        text = "This is a test contract with termination clause and liability limitation wording that is long enough."
        clauses, truncated = split_clauses(text)
        assert len(clauses) == 1 and truncated == 0

    def test_plain_non_clause_text_does_not_crash(self):
        text = "이것은 계약서가 아니라 그냥 일반적인 안내문입니다. 아무 조항도 없고 그냥 설명글입니다."
        clauses, truncated = split_clauses(text)
        assert len(clauses) == 1 and truncated == 0


class TestExtractSpans:
    def test_empty_evidence_span_returns_empty_list(self):
        assert _extract_spans("원문 텍스트입니다", "") == []

    def test_evidence_not_in_clause_returns_empty_list(self):
        assert _extract_spans("원문 텍스트입니다", "존재하지않는문구") == []

    def test_finds_matching_span_with_correct_offsets(self):
        clause = "제15조(해지) 사전 통보 없이 즉시 해지할 수 있으며 책임지지 아니한다."
        evidence = "사전 통보 없이 즉시 해지할 수 있으며"
        spans = _extract_spans(clause, evidence)
        assert len(spans) == 1
        assert spans[0].text == evidence
        assert clause[spans[0].start:spans[0].end] == evidence
