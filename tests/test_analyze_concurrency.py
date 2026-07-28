# tests/test_analyze_concurrency.py
"""backend/api/services/analyze.py의 동시 요청 제한(세마포어) 단위 테스트.

_process_clause/OpenAI를 스텁으로 바꿔서 실제 파이프라인 없이 세마포어
획득/해제/타임아웃 분기만 검증한다.
"""

import asyncio

import pytest
from fastapi import HTTPException

from backend.api.services import analyze

_TEXT = "제1조(목적) 이 약관은 서비스 이용에 관한 사항을 규정한다."


@pytest.fixture(autouse=True)
def _single_slot_semaphore(monkeypatch):
    # 기본값(MAX_CONCURRENT_ANALYSES=4)이면 슬롯 하나만 점유해선 타임아웃을
    # 재현할 수 없으므로, 테스트에선 슬롯 1개짜리로 고정한다.
    monkeypatch.setattr(analyze, "_analyze_semaphore", asyncio.Semaphore(1))
    monkeypatch.setattr(analyze, "_QUEUE_TIMEOUT_SECONDS", 0.2)


class TestConcurrencyLimit:
    async def test_times_out_with_503_when_slot_unavailable(self):
        async with analyze._analyze_semaphore:
            with pytest.raises(HTTPException) as exc_info:
                await analyze.run_analyze(_TEXT)
        assert exc_info.value.status_code == 503

    async def test_releases_semaphore_after_success(self, monkeypatch):
        monkeypatch.setattr(analyze, "_get_openai", lambda: object())
        monkeypatch.setattr(analyze, "_process_clause", lambda client, clause, index: None)

        # 슬롯이 하나뿐이라, 첫 호출에서 반납이 안 되면 두 번째 호출이 타임아웃으로 실패한다.
        await analyze.run_analyze(_TEXT)
        await analyze.run_analyze(_TEXT)
