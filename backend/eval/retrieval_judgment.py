# backend/eval/retrieval_judgment.py
"""
검색 기반 판단(KoE5 유사사례 top-5 검색 + GPT-4o-mini few-shot) — models/README.md에
서술만 있고 코드가 없던 실험을 재현 가능한 형태로 옮긴 것. Phase B 비교 평가
전용이며, 프로덕션 판단 경로(backend/agents/judgment_agent.py)는 여전히 KoELECTRA다.

v2(현재): Phase B 1차 결과에서 진단된 문제(Low 정답의 63%를 Medium으로 과다예측,
confusion matrix 기준) — 원인을 5건 직접 재현해보니 KoE5 이웃 사례 5개가 전부(5/5)
Low로 만장일치인데도 GPT가 "~일 수 있다" 식으로 이론적 트집을 만들어 Medium을
택하는 패턴이었다(이웃 사례 풀 자체는 Low가 다수라 구성 문제가 아님 — clean_clauses
전체 분포: Low 335 / High 78 / Medium 65). 이건 이 평가셋에 국한된 문제가 아니라
LLM이 애매한 다지선다 척도에서 극단을 피하고 중간으로 헷징하는 잘 알려진 경향이라,
confusion matrix 수치에 맞춘 땜질이 아니라 "제공된 근거를 실제로 따르라"는 일반
원칙으로 프롬프트를 보강했다(_SYSTEM 하단 참고).
"""

import json
import os
import time

from openai import OpenAI

from backend.api.services.retrieval import search_similar_labeled_clauses
from backend.utils import load_logger

logger = load_logger("retrieval_judgment.log")

RETRIEVAL_JUDGE_MODEL = os.environ.get("RETRIEVAL_JUDGE_MODEL", "gpt-4o-mini")
_TOP_K = 5

_SYSTEM = """당신은 한국 계약법 전문가입니다. 아래 검증된 유사 사례들(실제 조항과
확정된 risk_level)을 참고하여 판단 대상 조항의 risk_level을 정하세요.

risk_level: High(약관규제법 위반 소지 높음) / Medium(부분적으로 불리) / Low(공정하거나 통상적)

판단 원칙(중요):
- 유사 사례들의 risk_level이 다수 일치한다면(예: 5개 중 4개 이상 같은 값), 판단
  대상 조항이 그 사례들과 실질적으로 비슷한 내용을 다룬다면 그 다수 판정을 그대로
  따르세요. "이론적으로 다르게 해석될 여지가 있다", "~일 수도 있다"는 식의 추측만으로
  다수 사례와 다른 판정을 내리지 마세요 — 실제 조항 문구에 그 우려를 뒷받침하는
  구체적 근거(예: "일방적으로", "어떠한 경우에도", "책임을 지지 않는다" 같은 명확한
  표현)가 있을 때만 다수 사례와 다르게 판단하세요.
- Medium은 "애매해서 일단 중간"이 아니라, 조항에 소비자에게 불리한 요소가 실제로
  있지만 완전 면책·일방적 조항만큼 심각하지는 않을 때만 선택하세요. 표준계약서·
  공식 약관에서 흔히 보이는 통상적 조항(동의 요건, 절차적 통지의무, 검사·확인 의무
  등)은 Low입니다.

반드시 아래 JSON 형식으로만 응답하세요:
{"risk_level": "High" 또는 "Medium" 또는 "Low"}"""


def _build_examples_block(neighbors: list[dict]) -> str:
    if not neighbors:
        return "(유사 사례 없음)"
    lines = [f"[유사 사례 {i}] (유사도 {n['similarity']:.2f}, risk_level={n['risk_level']})\n{n['text']}" for i, n in enumerate(neighbors, 1)]
    return "\n\n".join(lines)


def retrieval_judge(client: OpenAI, clause_text: str, exclude_chunk_id: str | None = None, retries: int = 3) -> tuple[str | None, int]:
    """KoE5로 유사 검증 사례(clean_clauses) top-5를 찾아 GPT-4o-mini few-shot으로 risk_level을 판단한다.

    (risk_level, neighbor_agreement)를 반환한다 — neighbor_agreement는 최종 판정과 같은
    risk_level을 가진 이웃 사례 개수(0~5). 하이브리드 앙상블이 "GPT가 이웃 합의를 무시하고
    헷징했는지"를 판단하는 신뢰도 신호로 쓴다(backend/eval/hybrid_ensemble.py 참고).
    """
    neighbors = search_similar_labeled_clauses(clause_text, top_k=_TOP_K, exclude_chunk_id=exclude_chunk_id)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"{_build_examples_block(neighbors)}\n\n[판단 대상 조항]\n{clause_text}"},
    ]

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=RETRIEVAL_JUDGE_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
            )
            risk_level = json.loads(resp.choices[0].message.content).get("risk_level")
            agreement = sum(1 for n in neighbors if n["risk_level"] == risk_level)
            return risk_level, agreement
        except Exception as e:
            logger.warning(f"  검색기반 판단 호출 실패 ({attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    logger.error("  검색기반 판단 최종 실패 (재시도 소진)")
    return None, 0
