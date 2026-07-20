# backend/fb_check/consistency_verification.py
"""
Consistency Verification: evidence_span E → 재라벨 L'

Forward에서 추출한 evidence_span만 보고 GPT-4o가 독립적으로 재라벨링한다.
L == L' 이면 라벨 일관성 확인, L ≠ L' 이면 NOISE로 판정.
퓨샷 예시 3개를 포함하여 일관된 판단 기준을 유도한다.
"""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from backend.utils import load_logger

load_dotenv()

logger = load_logger("consistency_verification.log")

VERIFY_MODEL = os.environ["VERIFY_MODEL"]

_FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "근거 문구:\n사전 통보 없이 언제든지 본 서비스 이용계약을 즉시 해지할 수 있으며",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "domain": "해지_조항",
            "risk_level": "High",
        }, ensure_ascii=False),
    },
    {
        "role": "user",
        "content": "근거 문구:\n간접손해·특별손해·기대이익 손실에 대해서는 배상하지 아니한다",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "domain": "책임제한_조항",
            "risk_level": "Medium",
        }, ensure_ascii=False),
    },
    {
        "role": "user",
        "content": "근거 문구:\n이용자는 언제든지 서비스 해지를 신청할 수 있으며, 회사는 30일 이내에 처리한다",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "domain": "해지_조항",
            "risk_level": "Low",
        }, ensure_ascii=False),
    },
]

_SYSTEM = """당신은 한국 계약법 전문가입니다.
주어진 계약 조항 근거 문구만을 보고 리스크 라벨을 판단하세요.
전체 조항 맥락 없이 오직 근거 문구만으로 판단해야 합니다.

판단 기준:
- High: 사전 통보 없는 즉시 해지, 일방적 해지, 모든 손해 면책 등 명백히 소비자에게 불리한 문구
- Medium: 간접손해·특별손해 배제, 배상액 상한 등 부분적 제한 문구
- Low: 쌍방에 공정하거나 통상적인 문구

반드시 아래 JSON 형식으로만 응답하세요:
{
  "domain": "해지_조항" 또는 "책임제한_조항",
  "risk_level": "High" 또는 "Medium" 또는 "Low"
}"""


def run_verify(client: OpenAI, evidence_span: str, retries: int = 3, model: str = VERIFY_MODEL) -> dict | None:
    messages = [{"role": "system", "content": _SYSTEM}]
    messages.extend(_FEW_SHOT_EXAMPLES)
    messages.append({"role": "user", "content": f"근거 문구:\n{evidence_span}"})

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            logger.warning(f"  Consistency Verify 호출 실패 ({attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    logger.error(f"  Consistency Verify 최종 실패 (재시도 {retries}회 소진)")
    return None
