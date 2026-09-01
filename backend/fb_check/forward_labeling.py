# backend/fb_check/forward_labeling.py
"""
Forward Labeling: 계약 조항 C → 위반 소지 유형 A + 리스크 라벨 L + 근거 문구 E

`FORWARD_MODEL`(환경변수)이 계약 조항 전문을 보고 articles, risk_level, evidence_span,
reasoning을 출력한다. 퓨샷 예시를 포함하여 일관된 판단 기준을 유도한다.

## 왜 `articles`(약관규제법 조항 multi-label)를 쓰는가

이전 버전은 도메인을 `해지_조항 / 책임제한_조항 / 해당없음` 3택으로 물었다. 그 결과
FB-Check 입력 2,218건 중 **1,257건(56.7%)이 `해당없음`으로 판정**됐고, `해당없음`은
evidence_span이 빈 문자열이라 `check_snippet_exists=False`가 되어 **전량 NOISE로 탈락**했다
(`data/fb_check/fb_check_report.json`의 `노이즈_원인.domain_none`). 탈락분에는 공정위가
불공정으로 확정한 ftc_case 조항 349건이 포함돼 있었다.

실제 위반 분포와 맞지 않았던 것이 원인이다 — FTC 의결서 1,092건의 `근거_법령`을 집계하면
제6조 504 · 제9조 406 · 제8조 284 · 제14조 228 · 제11조 196 · 제7조 183 · 제10조 125건으로,
옛 2-도메인이 담당하던 제7·8·9조 밖에 절반 이상이 있다. 케이스당 평균 2.01개 조가 걸리므로
단일 선택이 아니라 multi-label이어야 한다.

유형 목록은 `backend.labeling.articles`(원문 법령 JSON에서 생성)를 단일 출처로 쓴다 —
FTC `근거_법령`과 같은 어휘를 써야 "LLM 판정 vs 공정위 확정"을 채점할 수 있기 때문이다.

옛 `domain` 필드는 `articles`에서 파생시켜 계속 채운다(`articles.derive_domain`) —
`agents/analysis_agent.py` 이하 그래프가 아직 2-도메인을 전제한다.

## 재현성

반환 dict에 `prompt_version`과 `model`을 함께 담는다. 이전 라벨 산출물
(`data/fb_check/clean.jsonl`)에는 이 정보가 없어 **어느 모델이 그 라벨을 만들었는지
확인할 수 없다** — 코드 docstring은 "GPT-4o", `.env`는 `gpt-4o-mini`로 서로 달랐다.
"""

import json
import time

from dotenv import load_dotenv
from openai import OpenAI

from backend.fb_check.api_errors import raise_if_fatal
from backend.labeling.articles import (
    ARTICLE_IDS,
    build_prompt_block_variant,
    derive_domain,
)
from backend.utils import lazy_env, load_logger, require_env

load_dotenv()

logger = load_logger("forward_labeling.log")

MAX_GPT_CHARS = 3000
# **최상단에서 `os.environ["FORWARD_MODEL"]`을 쓰지 않는다.** 이 모듈은 오프라인
# 라벨링용이지만 `agents/analysis_agent.py`가 `run_forward`를 재사용하므로 **서빙
# import 체인이 여기를 지난다.** 예전에는 이 줄 때문에 `FORWARD_MODEL`이 없는 배포에서
# `import backend.api.server` 자체가 raw `KeyError`로 죽었다 — `_validate_required_env()`가
# 돌기도 전이라 "무엇이 없는지"를 알려줄 기회조차 없었다. 검사는 `run_forward()`로 옮겼다.
FORWARD_MODEL = lazy_env("FORWARD_MODEL")

# 프롬프트를 고칠 때마다 올린다 — 라벨 레코드에 함께 기록되어 어느 프롬프트가
# 어느 라벨을 만들었는지 사후에 구분할 수 있게 한다.
PROMPT_VERSION = "fwd-v5-ordered-summaryblock"

_FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": """계약 조항:
제15조(해지) 회사는 이용자에게 사전 통보 없이 언제든지 본 서비스 이용계약을 즉시 해지할 수 있으며, 이로 인해 발생하는 손해에 대하여 회사는 어떠한 책임도 지지 아니한다.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "articles": ["제9조", "제7조", "제6조"],
            "risk_level": "High",
            "evidence_span": "사전 통보 없이 언제든지 본 서비스 이용계약을 즉시 해지할 수 있으며",
            "reasoning": "법률에 없는 해지권을 사업자에게 부여해 고객에게 불이익을 주므로 제9조 제2호에 해당하고, 그로 인한 손해를 전면 면책하므로 제7조 제1호에도 걸린다. 사업자에게만 일방적으로 유리하므로 제6조도 함께 적용된다.",
        }, ensure_ascii=False),
    },
    {
        "role": "user",
        "content": """계약 조항:
제32조(관할) 본 계약과 관련하여 분쟁이 발생한 경우 회사의 본점 소재지 관할 법원을 전속 관할 법원으로 하며, 이용자는 이에 대하여 이의를 제기할 수 없다.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "articles": ["제14조"],
            "risk_level": "High",
            "evidence_span": "회사의 본점 소재지 관할 법원을 전속 관할 법원으로 하며, 이용자는 이에 대하여 이의를 제기할 수 없다",
            "reasoning": "사업자 본점 소재지로 전속관할을 고정해 고객의 재판받을 권리를 제약하므로 제14조 제1호(고객에게 부당하게 불리한 재판관할 합의 조항)에 해당한다. 해지·면책과는 무관한 유형이다.",
        }, ensure_ascii=False),
    },
    {
        "role": "user",
        "content": """계약 조항:
제8조(계약목적물의 변경) ① "갑"은 사업상 필요에 따라 "을"의 사전 동의 없이 분양면적, 구좌의 분할·통합, 동선 변경 등을 할 수 있고, "을"은 이에 동의한 것으로 본다.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "articles": ["제10조", "제12조"],
            "risk_level": "High",
            "evidence_span": "\"을\"의 사전 동의 없이 분양면적, 구좌의 분할·통합, 동선 변경 등을 할 수 있고",
            "reasoning": "급부의 내용을 사업자가 일방적으로 변경할 권한을 두므로 제10조 제1호에 해당하고, 고객의 동의를 의제하므로 제12조 제1호에도 걸린다.",
        }, ensure_ascii=False),
    },
    {
        "role": "user",
        "content": """계약 조항:
제13조(해약환급금) 회원이 중도 해지하는 경우 이미 납입한 입회비는 반환하지 아니하며, 잔여 회비의 30%를 위약금으로 공제한 후 환급한다.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "articles": ["제8조", "제6조"],
            "risk_level": "High",
            "evidence_span": "이미 납입한 입회비는 반환하지 아니하며, 잔여 회비의 30%를 위약금으로 공제한 후 환급한다",
            "reasoning": "입회비 전액 미반환에 더해 잔여 회비의 30%까지 위약금으로 공제하므로 고객에게 부당하게 과중한 손해배상 의무를 지운다(제8조). 고객에게만 일방적으로 불리하므로 제6조도 함께 적용된다.",
        }, ensure_ascii=False),
    },
    {
        "role": "user",
        "content": """계약 조항:
제21조(손해배상) 회사의 귀책사유로 인한 손해에 대해서는 직접 손해에 한하여 배상하며, 간접손해·특별손해·기대이익 손실에 대해서는 배상하지 아니한다.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "articles": ["제7조"],
            "risk_level": "Medium",
            "evidence_span": "간접손해·특별손해·기대이익 손실에 대해서는 배상하지 아니한다",
            "reasoning": "고의·중과실까지 면책하지는 않으나 상당한 이유 없이 손해배상 범위를 제한하므로 제7조 제2호에 해당할 소지가 있다. 전면 면책이 아니어서 Medium으로 본다.",
        }, ensure_ascii=False),
    },
    {
        "role": "user",
        "content": """계약 조항:
제8조(계약 해지) 이용자는 언제든지 서비스 해지를 신청할 수 있으며, 회사는 30일 이내에 처리한다. 단, 이용자가 요금을 미납한 경우 회사는 14일 이상의 유예기간을 부여한 후 해지할 수 있다.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "articles": [],
            "risk_level": "Low",
            "evidence_span": "",
            "reasoning": "고객의 해지권을 보장하고 사업자가 해지할 때에도 유예기간을 부여하므로 제6~14조 어디에도 해당하지 않는다.",
        }, ensure_ascii=False),
    },
]

# 유형 목록은 f-string이 아니라 런타임 결합으로 넣는다 — JSON 예시의 중괄호를
# 이스케이프하지 않아도 되고, 법 개정 시 articles.py만 재생성하면 반영된다.
# 조문 블록만 갈아끼울 수 있게 조립 함수로 둔다 — A/B/C 절제 실험용
# (`backend/eval/prompt_block_ablation.md`). **few-shot은 건드리지 않는다**:
# 한 번에 한 변수만 바꿔야 페어드 비교가 성립한다.
_SYSTEM_HEAD = """당신은 한국 계약법 전문가입니다.
주어진 계약 조항이 「약관의 규제에 관한 법률」 제6조~제14조 중 어디에 걸리는지 판단하세요.

## 위반 유형 (해당하는 것을 모두 고르세요 — 복수 선택 가능)

"""

_SYSTEM_TAIL = """

## 판단 순서 (반드시 이 순서로)

**1단계 — 제7조~제14조 중 해당하는 조를 빠짐없이 찾으세요.** 조항이 무엇을 규정하는지로
판단합니다:

    해지·해제·계약기간          → 제9조
    면책·책임 배제·손해배상 범위 제한 → 제7조
    위약금·환급 제한·지연손해금·"반환하지 아니한다" → 제8조
    급부 내용의 일방적 결정·변경·중지  → 제10조
    항변권·상계권·기한이익·제3자 계약 제한·비밀 누설 → 제11조
    침묵을 동의로 간주·의사표시 형식 제한 → 제12조
    대리인에게 책임 전가          → 제13조
    관할 합의·부제소 특약·입증책임 전가 → 제14조

**2단계 — 1단계에서 찾은 조를 그대로 둔 채, 제6조를 추가할지 판단합니다.** 조항이 고객에게
일방적으로 불리하면 제6조를 **추가**하세요. 공정위는 구체적 조 위반을 인정하면서 제6조를
함께 적용하는 경우가 많습니다.

**제6조는 1단계 결과를 대체하지 않습니다.** 제6조를 넣는다고 해서 제9조·제8조 같은
구체적 조를 빼면 안 됩니다. 제6조 단독은 제7~14조 어디에도 해당하지 않지만 고객에게
부당하게 불리하거나 예상하기 어려운 조항일 때뿐입니다.

## 그 밖의 규칙

- 한 조항이 여러 조에 걸리는 경우가 흔합니다(공정위 의결서 기준 케이스당 평균 2.01개).
- 위 어디에도 해당하지 않으면 `articles`를 빈 배열로 두세요. 계약 조항이 아닌
  제목·목차·설명문도 빈 배열입니다.
- 다만 **빈 배열은 신중하게** 쓰세요. 사업자와 고객의 권리·의무가 대등하고 법령 범위
  안에 있을 때만 해당합니다. 한쪽에만 유리한 구석이 있으면 해당 조를 지목하세요.

## 리스크 라벨

- High: 무효로 판단될 소지가 큰 조항 (일방적 해지, 전면 면책, 전속관할 강제 등)
- Medium: 다툼의 여지가 있으나 전면적이지는 않은 조항 (간접손해 배제, 배상액 상한 등)
- Low: 통상적이고 공정한 조항. `articles`가 빈 배열이면 반드시 Low입니다.

## evidence_span 선택 시 주의

- 조항에 여러 문장이 있으면, 그중 위험도를 가장 잘 드러내는(가장 포괄적이거나 심한)
  문장을 근거로 선택하세요. 앞 문장이 완화된 표현이고 뒤에 더 강한 면책·제한 표현이
  나온다면 뒤 문장을 우선하세요 (첫 문장을 기계적으로 고르지 마세요).
- evidence_span은 다음 단계에서 **이 문구만 단독으로** 재판정에 쓰입니다 — "왜 위험한지"를
  보여주는 구체적 표현("어떠한 책임도 지지 아니한다", "이의를 제기할 수 없다")이 빠지면
  안 됩니다.
- 원문에서 **그대로 복사**하세요. 요약하거나 다듬으면 안 됩니다.

반드시 아래 JSON 형식으로만 응답하세요:
{
  "articles": ["제9조", "제7조"],
  "risk_level": "High" 또는 "Medium" 또는 "Low",
  "evidence_span": "원문에서 그대로 복사한 핵심 근거 문구 (articles가 비면 빈 문자열)",
  "reasoning": "어느 조 몇 호에 왜 해당하는지 (1~2문장)"
}"""


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


def build_system(block: str = _DEFAULT_BLOCK) -> str:
    """`block`은 full(전문) / summary(제목+각 호 앞부분) / title(제목만)."""
    return _SYSTEM_HEAD + build_prompt_block_variant(block) + _SYSTEM_TAIL


_SYSTEM = build_system(_DEFAULT_BLOCK)



def _normalize(out: dict, model: str) -> dict:
    """모델 출력을 다듬는다 — 알 수 없는 조를 버리고, 옛 `domain`과 재현 정보를 채운다.

    `articles`가 비면 `risk_level`은 Low로 강제한다(프롬프트 규칙과 동일). 모델이 빈
    배열과 High를 함께 내는 모순을 막기 위함이다.
    """
    articles = [a for a in (out.get("articles") or []) if a in ARTICLE_IDS]
    seen, uniq = set(), []
    for a in articles:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    risk = out.get("risk_level", "Low")
    if not uniq:
        risk = "Low"
    return {
        "articles":       uniq,
        "risk_level":     risk,
        "evidence_span":  out.get("evidence_span", "") if uniq else "",
        "reasoning":      out.get("reasoning", ""),
        "domain":         derive_domain(uniq),   # 옛 2-도메인 파이프라인 호환용 파생값
        "model":          model,
        "prompt_version": PROMPT_VERSION,
    }


def run_forward(client: OpenAI, clause_text: str, retries: int = 3,
                model: str = FORWARD_MODEL, block: str = _DEFAULT_BLOCK) -> dict | None:
    # 모듈 최상단 대신 **여기서** 검사한다(위 FORWARD_MODEL 주석 참고). `--model`로
    # 명시해 부르면 그 값이 그대로 통과하므로 라벨링 경로 동작은 달라지지 않는다.
    model = require_env(model, "FORWARD_MODEL", "Forward Labeling(라벨링 · Analysis Agent)")
    messages = [{"role": "system", "content": build_system(block) if block != _DEFAULT_BLOCK else _SYSTEM}]
    messages.extend(_FEW_SHOT_EXAMPLES)
    messages.append({"role": "user", "content": f"계약 조항:\n{clause_text[:MAX_GPT_CHARS]}"})

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
            raise_if_fatal(e, "Forward Labeling")
            logger.warning(f"  Forward Labeling 호출 실패 ({attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    logger.error(f"  Forward Labeling 최종 실패 (재시도 {retries}회 소진)")
    return None
