# backend/agents/red_team_agent.py
"""Red-team Agent — 검증된 유사 사례와 비교해 Judgment의 판단 편향 가능성을 probe하고,
충돌하는 사례가 있으면 LLM으로 반박 근거를 생성한다.

유사 사례 탐색(임베딩 최근접 이웃, 임계값 0.75)은 기존 규칙 그대로 유지한다 — 이
부분은 leave-one-out 실험으로 이미 검증됐다(clean_clauses 478건 기준 탐지율 2.3%,
표본 확인 결과 실제 의미 있는 차이를 잡아내는 것으로 확인됨). 바뀐 건 "충돌 사례를
찾은 뒤 무엇을 보여주는가"뿐이다 — 이전엔 템플릿 문자열 한 줄이었는데, 이제 LLM이
왜 이 판단을 재고할 필요가 있는지 실제 반박 논리를 생성한다.

risk_level 필드는 출력 스키마에 아예 없다 — LLM이 판단 결과 자체를 못 바꾸도록
구조적으로 차단한다(Judgment의 판단은 여기서 절대 안 건드림). 충돌 사례가 없으면
LLM을 호출하지 않는다(비용 절감 — 원래도 대부분 케이스에서 탐지 안 됨, 탐지율 2.3%).
"""

import json
import os
import time

from langchain_core.runnables import RunnableConfig
from openai import OpenAI

from backend.agents.state import ClauseState
from backend.api.services.retrieval import search_similar_labeled_clauses
from backend.utils import load_logger

logger = load_logger("red_team_agent.log")

REDTEAM_MODEL = os.environ.get("REDTEAM_MODEL", "gpt-4o-mini")
_SIMILARITY_THRESHOLD = 0.75
_TOP_K = 5

_SYSTEM = """당신은 한국 계약법 전문가입니다. 방금 어떤 조항에 risk_level 판단이
내려졌는데, 그와 아주 비슷한 조항이 과거에 다른 risk_level로 판정된 사례가
발견됐습니다. 이 판단을 재고할 필요가 있는지, 왜 그런지 반박 근거를 한두 문장으로
작성하세요 — 두 조항이 실질적으로 왜 다르게 봐야 하는지, 혹은 지금 판단이 재검토가
필요한 이유를 구체적으로 설명하세요. risk_level을 다시 정하지는 마세요 — 그건
당신의 역할이 아닙니다.

반드시 아래 JSON 형식으로만 응답하세요:
{"rebuttal": "반박 근거 텍스트"}"""


def _fallback_note(neighbor: dict) -> str:
    """LLM 호출이 실패했을 때 쓰는 안전망 — 이전 버전의 템플릿 문구."""
    return (
        f"유사도 {neighbor['similarity']:.2f}로 매우 비슷한 조항이 '{neighbor['risk_level']}'로 "
        f"판정된 사례가 있습니다: \"{neighbor['text'][:80]}\""
    )


def _generate_rebuttal(client: OpenAI, clause: str, risk_level: str, neighbor: dict, retries: int = 3) -> str | None:
    user_msg = (
        f"[현재 조항] (판정: {risk_level})\n{clause}\n\n"
        f"[상충 사례] (유사도 {neighbor['similarity']:.2f}, 판정: {neighbor['risk_level']})\n{neighbor['text']}"
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=REDTEAM_MODEL,
                messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user_msg}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return json.loads(resp.choices[0].message.content).get("rebuttal")
        except Exception as e:
            logger.warning(f"  Red-team 반박 생성 실패 ({attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def red_team_node(state: ClauseState, config: RunnableConfig) -> dict:
    query = state.get("evidence_span") or state["clause"]
    risk_level = state.get("risk_level")

    neighbors = search_similar_labeled_clauses(query, top_k=_TOP_K)
    for neighbor in neighbors:
        if neighbor["similarity"] >= _SIMILARITY_THRESHOLD and neighbor["risk_level"] != risk_level:
            client = config["configurable"]["client"]
            rebuttal = _generate_rebuttal(client, state["clause"], risk_level, neighbor)
            return {"redteam_note": rebuttal or _fallback_note(neighbor)}

    return {"redteam_note": ""}
