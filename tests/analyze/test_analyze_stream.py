# tests/analyze/test_analyze_stream.py
"""backend/api/services/analyze.py의 run_analyze_stream() 단위 테스트.

정상 완료 시 이벤트 모양과 세마포어 반납, 그리고 스트림이 끝까지 소비되지
않고 중간에 닫히는 경우(클라이언트 연결 끊김과 동일한 상황)에도 세마포어가
즉시 반납되는지를 검증한다 — 후자는 실제로 있었던 누수 버그의 회귀 테스트다.
"""

import asyncio

import pytest

from backend.api.services import analyze

_TEXT = "제1조(목적) 이 약관은 서비스 이용에 관한 사항을 규정한다."


@pytest.fixture(autouse=True)
def _single_slot_semaphore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analyze, "_analyze_semaphore", asyncio.Semaphore(1))
    monkeypatch.setattr(analyze, "_QUEUE_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(analyze, "_get_openai", lambda: object())
    monkeypatch.setattr(analyze, "_process_clause", lambda client, clause, index: None)


class TestRunAnalyzeStream:
    async def test_yields_progress_then_done_and_releases_slot(self) -> None:
        events = await analyze.run_analyze_stream(_TEXT)
        collected = [e async for e in events]

        assert collected[-1]["type"] == "done"
        assert any(e["type"] == "progress" for e in collected)

        # 슬롯이 하나뿐이라, 앞선 호출이 반납을 안 했으면 아래가 타임아웃으로 실패한다.
        events2 = await analyze.run_analyze_stream(_TEXT)
        async for _ in events2:
            pass

    async def test_closing_generator_early_releases_slot_immediately(self) -> None:
        """스트림을 끝까지 안 읽고 중간에 닫아도(클라이언트 연결 끊김과 동일)
        세마포어가 즉시 반납돼야 한다 — 반납이 GC 타이밍에 맡겨지면, 이어지는
        요청들이 (남은 슬롯이 없어) 503을 받게 되는 실제 버그였다.
        """
        events = await analyze.run_analyze_stream(_TEXT)
        await events.__anext__()  # 첫 이벤트만 받는다
        await events.aclose()  # 클라이언트 연결 끊김 시 라우터가 하는 것과 동일

        # 슬롯이 즉시 반납됐으면 타임아웃 없이 바로 성공해야 한다.
        events2 = await analyze.run_analyze_stream(_TEXT)
        async for _ in events2:
            pass
