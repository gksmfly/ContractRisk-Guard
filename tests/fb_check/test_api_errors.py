# tests/fb_check/test_api_errors.py
"""치명적 API 오류가 재시도 루프를 뚫고 나오는지 — 08-23 5시간 헛돌기의 재발 방지.

크레딧이 떨어진 뒤 `credit_balance_exhausted`를 평범한 429로 취급해 3번씩 재시도했고,
2,131건 × 3회 = 6,393번의 될 리 없는 호출로 5시간을 태웠다. 429 중 무엇이 풀리고
무엇이 안 풀리는지를 코드가 구분하게 만들었으므로, 그 구분을 고정한다.
"""

import pytest

from backend.fb_check.api_errors import FatalAPIError, is_fatal, raise_if_fatal

# 실제 로그에서 그대로 가져온 메시지 — 문구가 바뀌면 이 테스트가 먼저 깨져야 한다.
CREDITS = ("Error code: 429 - {'error': {'message': 'You have no credits remaining. Add credits "
           "to continue using the API at https://platform.openai.com/settings/organization/"
           "billing/.', 'type': 'insufficient_quota', 'param': None, "
           "'code': 'credit_balance_exhausted'}}")
TPM = ("Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-4o in organization "
       "org-XXXX on tokens per min (TPM): Limit 30000, Used 30000, Requested 2422. Please try "
       "again in 4.844s.', 'type': 'tokens', 'param': None, 'code': 'rate_limit_exceeded'}}")
BAD_KEY = ("Error code: 401 - {'error': {'message': 'Incorrect API key provided.', "
           "'code': 'invalid_api_key'}}")


@pytest.mark.parametrize("msg", [CREDITS, BAD_KEY], ids=["크레딧 소진", "키 오류"])
def test_안_풀리는_오류는_치명적으로_분류한다(msg: str) -> None:
    assert is_fatal(Exception(msg)) is True


@pytest.mark.parametrize("msg", [TPM, "Connection reset by peer", "Request timed out"],
                         ids=["TPM 초과", "네트워크", "타임아웃"])
def test_기다리면_풀리는_오류는_재시도한다(msg: str) -> None:
    """TPM 429는 **반드시** 재시도해야 한다 — 여기서 끊으면 정상 스로틀링에 실행이 죽는다."""
    assert is_fatal(Exception(msg)) is False
    raise_if_fatal(Exception(msg), "Forward Labeling")   # 던지지 않아야 한다


def test_치명적_오류는_재시도_루프를_뚫고_나온다(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_forward가 3번 재시도하지 않고 **첫 호출에서** 중단하는지."""
    from backend.fb_check import forward_labeling

    calls = []

    # OpenAI SDK의 `client.chat.completions.create(...)` 접근 경로를 흉내 낸다.
    # 클래스명은 PascalCase(`Claude.md` 규칙)로 두고, 소문자 접근 경로는 **속성으로** 만든다 —
    # 예전에는 클래스명 자체를 소문자로 두고 린터 예외 지시자로 덮었다.
    class _Fake:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kw) -> None:
                    calls.append(kw)
                    raise RuntimeError(CREDITS)

            completions = Completions      # 접근 경로는 소문자 유지
        chat = Chat

    monkeypatch.setattr(forward_labeling.time, "sleep", lambda *_: pytest.fail("치명적 오류에 백오프하면 안 된다"))
    with pytest.raises(FatalAPIError):
        forward_labeling.run_forward(_Fake(), "제5조 사업자는 언제든지 계약을 변경할 수 있다.")
    assert len(calls) == 1, f"재시도했다 — {len(calls)}회 호출됨 (1회여야 한다)"


def test_일시적_오류는_재시도를_소진한다(monkeypatch: pytest.MonkeyPatch) -> None:
    """반대 방향도 고정한다 — TPM 429에서 조기 중단하면 정상 실행이 죽는다."""
    from backend.fb_check import forward_labeling

    calls = []

    # OpenAI SDK의 `client.chat.completions.create(...)` 접근 경로를 흉내 낸다.
    # 클래스명은 PascalCase(`Claude.md` 규칙)로 두고, 소문자 접근 경로는 **속성으로** 만든다 —
    # 예전에는 클래스명 자체를 소문자로 두고 린터 예외 지시자로 덮었다.
    class _Fake:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kw) -> None:
                    calls.append(kw)
                    raise RuntimeError(TPM)

            completions = Completions      # 접근 경로는 소문자 유지
        chat = Chat

    monkeypatch.setattr(forward_labeling.time, "sleep", lambda *_: None)
    assert forward_labeling.run_forward(_Fake(), "제5조 사업자는 언제든지 계약을 변경할 수 있다.") is None
    assert len(calls) == 3, f"재시도 3회를 소진해야 한다 — {len(calls)}회"
