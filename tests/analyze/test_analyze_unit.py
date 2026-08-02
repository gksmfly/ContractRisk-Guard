# tests/test_analyze_unit.py
"""
backend/api/services/analyze.py의 순수 함수 단위 테스트.

외부 의존성(OpenAI API, KoELECTRA GPU 추론, DB)이 없어서 빠르고,
API 키/GPU 없이도 항상 돌아간다.

실행: pytest tests/test_analyze_unit.py
"""

from backend.api.services.analyze import split_clauses, _extract_spans


class TestSplitClauses:
    def test_empty_text_returns_empty_list(self):
        assert split_clauses("") == []

    def test_too_short_text_returns_empty_list(self):
        # 20자 이하는 조항으로 취급하지 않는다
        assert split_clauses("안녕") == []

    def test_splits_numbered_articles(self):
        text = (
            "제1조(목적) 이 약관은 서비스 이용에 관한 사항을 규정한다.\n"
            "제2조(정의) 이 약관에서 사용하는 용어의 정의는 다음과 같다."
        )
        clauses = split_clauses(text)
        assert len(clauses) == 2
        assert clauses[0].startswith("제1조")
        assert clauses[1].startswith("제2조")

    def test_caps_at_20_clauses(self):
        text = "\n".join(f"제{i}조(조항) 이것은 스무 개가 넘는 조항 테스트를 위한 내용입니다." for i in range(1, 31))
        clauses = split_clauses(text)
        assert len(clauses) == 20

    def test_non_korean_text_treated_as_single_clause(self):
        # 구분자(제n조 등)가 없으면 전체를 한 덩어리로 취급 — 에러 없이 통과해야 한다
        text = "This is a test contract with termination clause and liability limitation wording that is long enough."
        clauses = split_clauses(text)
        assert len(clauses) == 1

    def test_plain_non_clause_text_does_not_crash(self):
        text = "이것은 계약서가 아니라 그냥 일반적인 안내문입니다. 아무 조항도 없고 그냥 설명글입니다."
        clauses = split_clauses(text)
        assert len(clauses) == 1


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
