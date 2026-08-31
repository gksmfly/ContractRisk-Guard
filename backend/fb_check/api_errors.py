# backend/fb_check/api_errors.py
"""재시도해도 소용없는 API 오류를 가려낸다.

## 왜 있는가

2026-08-23 전량 실행이 13:47에 OpenAI 크레딧을 소진했는데, **18:52까지 5시간을 더 돌았다.**

    [1501~2000] CLEAN 71% · ERROR   1
    [2001~2500] CLEAN 36% · ERROR 263
    [2501~3000] CLEAN  0% · ERROR 500     ← 여기서 멈췄어야 했다
    [3001~3500] CLEAN  0% · ERROR 500
    [3501~4000] CLEAN  0% · ERROR 500

구간 지표는 정확히 잡아서 세 번이나 찍었다. **아무도 멈추지 않은 게 문제다.**
`credit_balance_exhausted`를 429로 받아 지수 백오프로 3번씩 재시도했고
(1+2초 대기), 2,131건 × 3회 = **6,393번의 될 리 없는 호출**을 했다.

429는 두 가지를 뜻한다. 하나는 기다리면 풀리고 하나는 절대 안 풀린다:

    rate_limit_exceeded      TPM 초과. 기다리면 풀린다 → 재시도가 맞다 (실제로 14건)
    insufficient_quota       크레딧 없음. 영원히 안 풀린다 → 즉시 중단해야 한다 (6,393건)

HTTP 상태 코드만 보면 둘이 같아서, `code`/`type` 본문을 봐야 구분된다.
"""


class FatalAPIError(RuntimeError):
    """재시도·계속 실행이 무의미한 오류. 호출자는 실행 전체를 중단해야 한다."""


# 응답 본문의 code/type에 이 문자열이 들어 있으면 재시도하지 않는다.
_FATAL_MARKERS = (
    "insufficient_quota",           # 크레딧 소진 — 429로 오지만 절대 안 풀린다
    "credit_balance_exhausted",
    "billing_hard_limit_reached",
    "invalid_api_key",              # 401 — 키가 틀렸다
    "account_deactivated",
    "model_not_found",              # 모델명 오타 — 4,000건을 다 태우고 알 이유가 없다
    "permission_denied",
)


def is_fatal(exc: BaseException) -> bool:
    """이 예외가 '기다려도 안 풀리는' 종류인가."""
    return any(m in str(exc).lower() for m in _FATAL_MARKERS)


def raise_if_fatal(exc: BaseException, stage: str) -> None:
    """치명적이면 FatalAPIError로 승격해 재시도 루프를 뚫고 나간다."""
    if is_fatal(exc):
        raise FatalAPIError(f"{stage}: 재시도해도 풀리지 않는 오류다 — 실행을 중단한다\n  {exc}") from exc
