# backend/fb_check/forward_labeling.py
"""
Forward Labeling: 계약 조항 C → 리스크 라벨 L + 근거 문구 E

GPT-4o가 계약 조항 전문을 보고 domain, risk_level, evidence_span, reasoning을 출력한다.
퓨샷 예시 3개를 포함하여 일관된 판단 기준을 유도한다.
"""

import json
import time

from openai import OpenAI

MAX_GPT_CHARS = 3000

_FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": """계약 조항:
제15조(해지) 회사는 이용자에게 사전 통보 없이 언제든지 본 서비스 이용계약을 즉시 해지할 수 있으며, 이로 인해 발생하는 손해에 대하여 회사는 어떠한 책임도 지지 아니한다.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "domain": "해지_조항",
            "risk_level": "High",
            "evidence_span": "사전 통보 없이 언제든지 본 서비스 이용계약을 즉시 해지할 수 있으며",
            "reasoning": "사업자가 사전 통보 없이 일방적으로 즉시 해지할 수 있는 권한을 부여하고 있어 약관규제법 제9조(계약의 해제·해지)에 위반될 소지가 높다.",
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
            "domain": "책임제한_조항",
            "risk_level": "Medium",
            "evidence_span": "간접손해·특별손해·기대이익 손실에 대해서는 배상하지 아니한다",
            "reasoning": "직접손해는 배상하되 간접손해·특별손해를 전면 배제하고 있어 완전 면책은 아니나 소비자에게 불리한 책임 범위 제한에 해당한다.",
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
            "domain": "해지_조항",
            "risk_level": "Low",
            "evidence_span": "이용자는 언제든지 서비스 해지를 신청할 수 있으며, 회사는 30일 이내에 처리한다",
            "reasoning": "이용자의 해지권을 보장하고 회사의 해지 시에도 14일 이상 유예기간을 부여하는 등 쌍방에 공정한 조건을 규정하고 있다.",
        }, ensure_ascii=False),
    },
]

_SYSTEM = """당신은 한국 계약법 전문가입니다.
주어진 텍스트가 계약 조항인지 먼저 판단하고, 조항이면 리스크 라벨을 부여하세요.

도메인 판단:
- 해지_조항: 계약 해지·해제·종료에 관한 조항
- 책임제한_조항: 손해배상 면책·책임 제한에 관한 조항
- 해당없음: 계약 조항이 아닌 제목·목차·설명문이거나, 위 두 도메인에 해당하지 않는 조항

리스크 판단 기준 (해당없음이면 risk_level은 반드시 "Low"):
- High: 사전 통보 없는 즉시 해지, 일방적 해지, 모든 손해 면책, 완전 면책 등 약관규제법 위반 소지가 높은 조항
- Medium: 간접손해·특별손해 배제, 배상액 상한 설정 등 부분적으로 소비자에게 불리한 조항
- Low: 쌍방에 공정하거나 법령 범위 내의 통상적인 조항

반드시 아래 JSON 형식으로만 응답하세요:
{
  "domain": "해지_조항" 또는 "책임제한_조항" 또는 "해당없음",
  "risk_level": "High" 또는 "Medium" 또는 "Low",
  "evidence_span": "원문에서 그대로 복사한 핵심 근거 문구 (해당없음이면 빈 문자열)",
  "reasoning": "판단 근거 설명 (1~2문장)"
}"""


def run_forward(client: OpenAI, clause_text: str, retries: int = 3) -> dict | None:
    messages = [{"role": "system", "content": _SYSTEM}]
    messages.extend(_FEW_SHOT_EXAMPLES)
    messages.append({"role": "user", "content": f"계약 조항:\n{clause_text[:MAX_GPT_CHARS]}"})

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None
