# tests/test_analyze_integration.py
"""
backend/api/services/analyze.py의 run_analyze() 통합 테스트.

실제 OpenAI API(비용 발생) + KoELECTRA GPU 추론 + DB(판례/법령 검색)를
전부 거친다 — .env(OPENAI_API_KEY, DATABASE_URL, FORWARD_MODEL 등)가
설정돼 있어야 하고 GPU가 있어야 한다. 느리고 비용이 드니 평소엔
pytest -m "not integration"으로 건너뛰고, 배포 전 등에만 돌린다.

실행: pytest tests/test_analyze_integration.py -m integration
"""

import pytest
from fastapi import HTTPException

from backend.api.services.analyze import run_analyze

pytestmark = pytest.mark.integration


class TestNormalContracts:
    async def test_high_risk_termination_clause(self):
        text = (
            "제15조(해지) 회사는 이용자에게 사전 통보 없이 언제든지 본 서비스 이용계약을 "
            "즉시 해지할 수 있으며, 이로 인해 발생하는 손해에 대하여 회사는 어떠한 책임도 지지 아니한다."
        )
        result = await run_analyze(text)
        assert result.total_clauses == 1
        clause = result.clauses[0]
        assert clause.domain == "해지_조항"
        assert clause.risk_level == "High"
        assert len(clause.legal_basis) > 0

    async def test_mixed_risk_multi_clause_contract(self):
        text = (
            "제15조(해지) 회사는 이용자에게 사전 통보 없이 언제든지 본 서비스 이용계약을 "
            "즉시 해지할 수 있으며, 이로 인해 발생하는 손해에 대하여 회사는 어떠한 책임도 지지 아니한다.\n\n"
            "제8조(계약 해지) 이용자는 언제든지 서비스 해지를 신청할 수 있으며, "
            "회사는 30일 이내에 처리한다."
        )
        result = await run_analyze(text)
        assert result.total_clauses == 2
        assert result.high_count + result.medium_count + result.low_count == 2


class TestEdgeCases:
    async def test_empty_text_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            await run_analyze("")
        assert exc_info.value.status_code == 400

    async def test_too_short_text_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            await run_analyze("짧음")
        assert exc_info.value.status_code == 400

    async def test_non_korean_text_does_not_crash(self):
        text = (
            "This is a sample termination clause. The company may terminate this agreement "
            "at any time without prior notice and shall not be liable for any resulting damages."
        )
        result = await run_analyze(text)
        # 도메인 판단이 어느 쪽이든(해당없음 포함) 예외 없이 응답 스키마를 지켜야 한다
        assert result.total_clauses == len(result.clauses)

    async def test_plain_non_clause_text_does_not_crash(self):
        text = (
            "이것은 계약서가 아니라 그냥 일반적인 안내문입니다. 아무 조항도 없고 "
            "그냥 설명글입니다. 이런 텍스트가 들어와도 서버가 죽으면 안 됩니다."
        )
        result = await run_analyze(text)
        assert result.total_clauses == len(result.clauses)

    async def test_repeated_whitespace_only_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            await run_analyze("   \n\n   \n   ")
        assert exc_info.value.status_code == 400
