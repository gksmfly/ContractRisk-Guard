# tests/api/test_auth.py
"""backend/api/auth.py의 API 키 검증 단위 테스트.

외부 의존성 없음 — API_KEY를 monkeypatch로 직접 세팅해서 통과/차단 분기를 검증한다.
"""

import pytest
from fastapi import HTTPException

from backend.api import auth


class TestRequireApiKey:
    async def test_no_api_key_configured_skips_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(auth, "API_KEY", "")
        await auth.require_api_key("아무값")  # 예외 없이 통과해야 함
        await auth.require_api_key("")  # 헤더 자체가 없어도 통과

    async def test_correct_key_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(auth, "API_KEY", "secret123")
        await auth.require_api_key("secret123")

    async def test_wrong_key_raises_401(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(auth, "API_KEY", "secret123")
        with pytest.raises(HTTPException) as exc_info:
            await auth.require_api_key("wrong")
        assert exc_info.value.status_code == 401

    async def test_missing_key_raises_401_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(auth, "API_KEY", "secret123")
        with pytest.raises(HTTPException) as exc_info:
            await auth.require_api_key("")
        assert exc_info.value.status_code == 401
