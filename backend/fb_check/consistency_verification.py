# backend/fb_check/consistency_verification.py
"""
Consistency Verification: 근거 문구 E → 위반 유형 A' + 리스크 라벨 L' 재판정

Forward가 고른 근거 문구만 보고 다시 판정한다. 조항 전문에 섞인 완충 표현
("단, ~한 경우에는", "관련 법령이 정하는 바에 따라")에 희석되지 않은 판정을 얻는 것이
목적이다.

## 왜 forward와 같은 모델을 쓰는가

`FORWARD_MODEL`과 `VERIFY_MODEL`을 다르게 두자는 논의가 있었으나, 실측 결과 두 단계의
차이는 **모델이 아니라 입력에서** 나오고 그 차이가 유용하다:

    두 단계 모두 gpt-4o-mini인데 744건 중 22.0%가 불일치
      verify가 더 위험하게 판정 134건 / 더 안전하게 30건  (상향 4.5배)

      ftc_case 340건(공정위 위반 확정)   High  forward 108 → verify 192
      standard_contract 404건(정부 표준) Low   forward 358 → verify 349

verify는 무차별로 상향하지 않는다 — 위반 확정 조항에서만 High를 84건 더 잡아내고
표준계약서에서는 거의 움직이지 않는다. 여기서 모델까지 다르게 하면 불일치가 생겼을 때
입력 때문인지 모델 때문인지 구분할 수 없게 되므로, **두 단계는 같은 모델로 유지**한다.

교차검증용 독립 투표자가 필요하다면 세 번째 표(`backward_grounding`의 KoELECTRA)를
교체해야 한다 — 그쪽이 상수 투표자 대비 순 기여 +6.1%p에 그치고, 자기 산출물로 학습되는
순환 구조다(`backend/eval/fbcheck_variant_compare.py`).

유형 목록은 forward와 동일하게 `backend.labeling.articles`를 쓴다.
"""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from backend.labeling.articles import (
    ARTICLE_IDS, derive_domain, prompt_block, prompt_block_variant,
)
from backend.fb_check.api_errors import raise_if_fatal
from backend.utils import load_logger

load_dotenv()

logger = load_logger("consistency_verification.log")

VERIFY_MODEL = os.environ["VERIFY_MODEL"]

# forward와 독립적으로 올린다 — 두 단계 프롬프트를 따로 고칠 수 있어야 한다.
PROMPT_VERSION = "ver-v5-ordered-summaryblock"
MAX_VERIFY_CHARS = 3000  # clause 모드 입력 상한 (forward의 MAX_GPT_CHARS와 동일)

_FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "근거 문구:\n사전 통보 없이 언제든지 본 서비스 이용계약을 즉시 해지할 수 있으며",
    },
    {
        "role": "assistant",
        "content": json.dumps({"articles": ["제9조", "제6조"], "risk_level": "High"}, ensure_ascii=False),
    },
    {
        "role": "user",
        "content": "근거 문구:\n이미 납입한 입회비는 반환하지 아니하며, 잔여 회비의 30%를 위약금으로 공제한 후 환급한다",
    },
    {
        "role": "assistant",
        "content": json.dumps({"articles": ["제8조", "제6조"], "risk_level": "High"}, ensure_ascii=False),
    },
    {
        "role": "user",
        "content": "근거 문구:\n회사의 본점 소재지 관할 법원을 전속 관할 법원으로 하며, 이용자는 이에 대하여 이의를 제기할 수 없다",
    },
    {
        "role": "assistant",
        "content": json.dumps({"articles": ["제14조"], "risk_level": "High"}, ensure_ascii=False),
    },
    {
        "role": "user",
        "content": "근거 문구:\n간접손해·특별손해·기대이익 손실에 대해서는 배상하지 아니한다",
    },
    {
        "role": "assistant",
        "content": json.dumps({"articles": ["제7조"], "risk_level": "Medium"}, ensure_ascii=False),
    },
    {
        "role": "user",
        "content": "근거 문구:\n이용자는 언제든지 서비스 해지를 신청할 수 있으며, 회사는 30일 이내에 처리한다",
    },
    {
        "role": "assistant",
        "content": json.dumps({"articles": [], "risk_level": "Low"}, ensure_ascii=False),
    },
]

# 검증 모드가 둘인 이유:
#   span   — forward가 위반 유형을 찾은 경우. 근거 문구만 주고 재판정한다. 조항 전문에
#            섞인 완충 표현에 희석되지 않은 판단을 얻는 것이 목적이다.
#   clause — forward가 "위반 없음"(articles 빈 배열)이라 한 경우. 근거 문구 자체가 없으므로
#            조항 전문을 준다. 부정 판정은 근거 문구로 검증할 수 없기 때문이다.
#            어느 모드로 판정했는지는 결과 레코드에 `verify_mode`로 남긴다.
_HEAD__SYSTEM_SPAN = """당신은 한국 계약법 전문가입니다.
주어진 계약 조항의 근거 문구만을 보고 「약관의 규제에 관한 법률」 위반 유형과 리스크를
판단하세요. 전체 조항 맥락 없이 오직 근거 문구만으로 판단해야 합니다.

## 위반 유형 (해당하는 것을 모두 고르세요 — 복수 선택 가능)

"""

_TAIL__SYSTEM_SPAN = """

## 판단 규칙

- **먼저 제7~14조 중 해당하는 조를 빠짐없이 찾으세요**: 해지·해제→제9조 / 면책·책임제한→제7조 /
  위약금·환급 제한·지연손해금→제8조 / 급부 일방변경→제10조 / 고객 권리 제한→제11조 /
  동의 의제→제12조 / 대리인 책임→제13조 / 관할·부제소·입증책임→제14조.
- **그 다음** 문구가 고객에게 일방적으로 불리하면 제6조를 **추가**하세요.
  **제6조가 구체적 조를 대체하면 안 됩니다** — 제6조를 넣어도 제9조·제8조는 그대로 둡니다.
- 근거 문구만으로 위 조항 중 어디에도 걸린다고 보기 어려우면 `articles`를 빈 배열로 두세요.
  다만 한쪽에만 유리한 구석이 있으면 해당 조를 지목하세요.
- High: 무효로 판단될 소지가 큰 문구 / Medium: 부분적 제한 문구 /
  Low: 통상적이거나 공정한 문구. `articles`가 빈 배열이면 반드시 Low입니다.

반드시 아래 JSON 형식으로만 응답하세요:
{
  "articles": ["제9조"],
  "risk_level": "High" 또는 "Medium" 또는 "Low"
}"""


_HEAD__SYSTEM_CLAUSE = """당신은 한국 계약법 전문가입니다.
주어진 계약 조항이 「약관의 규제에 관한 법률」 제6조~제14조 중 어디에 걸리는지 판단하세요.
다른 검토자가 이 조항을 "위반 없음"으로 봤습니다. 그 판단에 얽매이지 말고 독립적으로
다시 판단하세요.

## 위반 유형 (해당하는 것을 모두 고르세요 — 복수 선택 가능)

"""

_TAIL__SYSTEM_CLAUSE = """

## 판단 규칙

- **먼저 제7~14조 중 해당하는 조를 빠짐없이 찾고**(위약금·환급 제한→제8조,
  급부 일방변경→제10조, 관할·부제소→제14조), **그 다음** 고객에게 일방적으로 불리하면
  제6조를 **추가**하세요. 제6조가 구체적 조를 대체하면 안 됩니다.
- 위 어디에도 해당하지 않으면 `articles`를 빈 배열로 두세요. 계약 조항이 아닌
  제목·목차·설명문도 빈 배열입니다.
- 다만 빈 배열은 신중하게 쓰세요 — 사업자와 고객의 권리·의무가 대등할 때만 해당합니다.
- High: 무효로 판단될 소지가 큰 조항 / Medium: 부분적 제한 조항 /
  Low: 통상적이고 공정한 조항. `articles`가 빈 배열이면 반드시 Low입니다.

반드시 아래 JSON 형식으로만 응답하세요:
{
  "articles": [],
  "risk_level": "High" 또는 "Medium" 또는 "Low"
}"""



def _normalize(out: dict, model: str) -> dict:
    """forward와 같은 규칙으로 다듬는다 — 알 수 없는 조 제거, 빈 배열이면 Low 강제."""
    seen, uniq = set(), []
    for a in (out.get("articles") or []):
        if a in ARTICLE_IDS and a not in seen:
            seen.add(a)
            uniq.append(a)
    return {
        "articles":       uniq,
        "risk_level":     out.get("risk_level", "Low") if uniq else "Low",
        "domain":         derive_domain(uniq),   # 옛 2-도메인 호환용 파생값
        "model":          model,
        "prompt_version": PROMPT_VERSION,
    }


# 조문 블록 기본값 = "summary"(제목 + 각 호 앞 45자 × 2).
# A/B/C 절제 실험 결과(`backend/eval/prompt_block_ablation.md`, 100건 페어드):
#
#   구성            블록토큰   조항당입력   총편차   판정
#   A 전문            1,126     5,026      46     기준선
#   B 제목+요지          498     3,770      48     ← 채택 (Δ+2, 노이즈 범위)
#   C 제목만            111     2,994      61     기각 (Δ+15)
#
# 전문을 빼면 이웃 조의 **경계 문구**가 사라져 갈 곳 잃은 조항이 포괄적 제목
# (제8조 "손해배상액의 예정", 제6조 "일반원칙")으로 쏠린다 — C에서 제8조가 32→43.
# 앞 45자면 그 경계가 유지된다. 전량 라벨링 입력이 12.2M→9.2M 토큰으로 줄어
# TPM 스로틀링 기준 6.8시간 → 5.1시간이 된다.
_DEFAULT_BLOCK = "summary"


def build_system(mode: str = "span", block: str = _DEFAULT_BLOCK) -> str:
    """조문 블록만 갈아끼운 시스템 프롬프트. forward와 같은 규칙(few-shot 불변)."""
    if mode == "span":
        return _HEAD__SYSTEM_SPAN + prompt_block_variant(block) + _TAIL__SYSTEM_SPAN
    if mode == "clause":
        return _HEAD__SYSTEM_CLAUSE + prompt_block_variant(block) + _TAIL__SYSTEM_CLAUSE
    raise ValueError(f"알 수 없는 verify mode: {mode}")


_SYSTEM_SPAN   = build_system("span", _DEFAULT_BLOCK)
_SYSTEM_CLAUSE = build_system("clause", _DEFAULT_BLOCK)


def run_verify(
    client: OpenAI,
    text: str,
    mode: str = "span",
    retries: int = 3,
    model: str = VERIFY_MODEL,
    block: str = _DEFAULT_BLOCK,
) -> dict | None:
    """`mode="span"`이면 근거 문구만, `"clause"`면 조항 전문을 주고 재판정한다."""
    if mode == "span":
        sys_span = build_system("span", block) if block != _DEFAULT_BLOCK else _SYSTEM_SPAN
        messages = [{"role": "system", "content": sys_span}, *_FEW_SHOT_EXAMPLES,
                    {"role": "user", "content": f"근거 문구:\n{text}"}]
    elif mode == "clause":
        # 조항 전문 모드에는 퓨샷을 붙이지 않는다 — 퓨샷이 전부 짧은 문구라
        # 조항 전문 입력과 형식이 어긋나 오히려 판정을 흔든다.
        sys_cl = build_system("clause", block) if block != _DEFAULT_BLOCK else _SYSTEM_CLAUSE
        messages = [{"role": "system", "content": sys_cl},
                    {"role": "user", "content": f"계약 조항:\n{text[:MAX_VERIFY_CHARS]}"}]
    else:
        raise ValueError(f"알 수 없는 verify mode: {mode}")

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
            )
            return _normalize(json.loads(resp.choices[0].message.content), model)
        except Exception as e:
            # 크레딧 소진·키 오류처럼 기다려도 안 풀리는 건 즉시 위로 던진다.
            # 예전에는 429를 전부 같게 보고 3번씩 재시도해서, 크레딧이 떨어진 뒤
            # 5시간 동안 6,393번의 무의미한 호출을 했다 — `api_errors` 참고.
            raise_if_fatal(e, "Consistency Verify")
            logger.warning(f"  Consistency Verify 호출 실패 ({attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    logger.error(f"  Consistency Verify 최종 실패 (재시도 {retries}회 소진)")
    return None
