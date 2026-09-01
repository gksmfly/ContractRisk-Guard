# backend/agents/red_team_agent.py
"""Red-team Agent — 검증된 유사 사례와 비교해 Judgment의 판단 편향 가능성을 probe하고,
충돌하는 사례가 있으면 LLM으로 반박 근거를 생성한다.

유사 사례 탐색(임베딩 최근접 이웃, 임계값 0.75)은 기존 규칙 그대로 유지한다 — 이
부분은 leave-one-out 실험으로 이미 검증됐다(clean_clauses 478건 기준 탐지율 2.3%,
표본 확인 결과 실제 의미 있는 차이를 잡아내는 것으로 확인됨). 바뀐 건 "충돌 사례를
찾은 뒤 무엇을 보여주는가"뿐이다 — 이전엔 템플릿 문자열 한 줄이었는데, 이제 LLM이
왜 이 판단을 재고할 필요가 있는지 실제 반박 논리를 생성한다.

판정 필드(조 목록)는 출력 스키마에 아예 없다 — LLM이 판단 결과 자체를 못 바꾸도록
구조적으로 차단한다(Judgment의 판단은 여기서 절대 안 건드림). 충돌 사례가 없으면
LLM을 호출하지 않는다(비용 절감 — 원래도 대부분 케이스에서 탐지 안 됨, 탐지율 2.3%).

## ⚠️ 지금은 사실상 비활성이다 (2026-08-31)

비교 축이 `risk_level`에서 **조 목록**으로 바뀌었는데 이웃 데이터(`clean_clauses` 테이블)는
아직 옛 라벨로 적재돼 있다. 조 라벨이 없는 이웃은 건너뛰므로 실제로는 아무것도 발동하지
않는다. 되살리려면 새 `clean.jsonl`(조 multi-label)로 재적재할 것 — `red_team_node`
docstring 참고.
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

# 프롬프트가 말하는 축과 `_generate_rebuttal`이 실제로 넘기는 값(조 목록)이 같아야 한다.
# 2026-08-31 taxonomy 전환 때 user 메시지만 조 목록으로 바뀌고 이 시스템 프롬프트는
# risk_level을 말하고 있었다 — 모델에게 존재하지 않는 필드를 두고 논하라고 시키는 셈이었다.
_SYSTEM = """당신은 한국 계약법 전문가입니다. 방금 어떤 조항에 대해 「약관의 규제에 관한
법률」 제6~14조 중 어디에 걸리는지 판정이 내려졌는데, 그와 아주 비슷한 조항이 과거에
**다른 조**로 판정된 사례가 발견됐습니다. 이 판정을 재고할 필요가 있는지, 왜 그런지
반박 근거를 한두 문장으로 작성하세요 — 두 조항이 실질적으로 왜 다르게 봐야 하는지,
혹은 지금 판정이 재검토가 필요한 이유를 구체적으로 설명하세요. 어느 조인지를 다시
정하지는 마세요 — 그건 당신의 역할이 아닙니다.

반드시 아래 JSON 형식으로만 응답하세요:
{"rebuttal": "반박 근거 텍스트"}"""


def _build_fallback_note(neighbor: dict) -> str:
    """LLM 호출이 실패했을 때 쓰는 안전망 — 이전 버전의 템플릿 문구."""
    return (
        f"유사도 {neighbor['similarity']:.2f}로 매우 비슷한 조항이 "
        f"'{', '.join(neighbor.get('articles') or []) or neighbor.get('risk_level', '?')}'로 "
        f"판정된 사례가 있습니다: \"{neighbor['text'][:80]}\""
    )


def _generate_rebuttal(client: OpenAI, clause: str, verdict: object, neighbor: dict, retries: int = 3) -> str | None:
    user_msg = (
        f"[현재 조항] (판정: {verdict})\n{clause}\n\n"
        f"[상충 사례] (유사도 {neighbor['similarity']:.2f}, "
        f"판정: {neighbor.get('articles') or neighbor.get('risk_level', '?')})\n{neighbor['text']}"
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


# `clean_clauses`가 조 라벨을 갖고 있는가. **첫 검색 결과에서 배운다** —
# 별도 프로브 질의를 던지면 아끼려던 DB 왕복이 그대로 든다.
# `None`=아직 모름, `False`=옛 라벨만 있음(이후 검색 자체를 건너뜀).
_NEIGHBORS_LABELED: bool | None = None


def reset_neighbor_cache() -> None:
    """테스트 전용 — 모듈 캐시를 비운다. `clean_clauses` 재적재 후에도 쓸 수 있다."""
    global _NEIGHBORS_LABELED
    _NEIGHBORS_LABELED = None


def red_team_node(state: ClauseState, config: RunnableConfig) -> dict:
    """비슷한 조항이 **다르게** 판정된 적이 있으면 반박 메모를 붙인다.

    ## 조 taxonomy 전환 중에는 비활성이다 (2026-08-31)

    judgment_agent가 `risk_level`을 더 이상 내지 않는데, 이웃 데이터(`clean_clauses`
    테이블)는 아직 **옛 라벨(risk_level)** 로 적재돼 있다. 그대로 두면
    `neighbor["risk_level"] != None`이 **항상 참**이라 유사도만 넘으면 무조건 반박이
    발동한다 — 근거 없는 경고를 매번 붙이는 셈이다.

    그래서 이웃에 조 라벨(`articles`)이 있을 때만 비교하고, 없으면 **침묵한다.**
    틀린 근거로 말하는 것보다 아무 말도 안 하는 편이 낫다.

    되살리려면 `clean_clauses`를 새 `clean.jsonl`(조 multi-label)로 재적재해야 한다.
    """
    global _NEIGHBORS_LABELED
    mine = set(state.get("model_articles") or [])
    if not mine:
        return {"redteam_note": ""}

    # 이웃이 옛 라벨만 갖고 있다는 걸 한 번 알았으면 **검색 자체를 건너뛴다.**
    # 어차피 산출이 0인데 조항마다 임베딩 최근접 검색이 돈다(30조항이면 30번).
    if _NEIGHBORS_LABELED is False:
        return {"redteam_note": ""}

    query = state.get("evidence_span") or state["clause"]
    neighbors = search_similar_labeled_clauses(query, top_k=_TOP_K)
    if neighbors and all(n.get("articles") is None for n in neighbors):
        _NEIGHBORS_LABELED = False
        logger.info("  Red-team 비활성 — clean_clauses에 조 라벨이 없다. "
                    "새 clean.jsonl로 재적재하고 프로세스를 재시작하면 되살아난다")
        return {"redteam_note": ""}
    _NEIGHBORS_LABELED = True
    for neighbor in neighbors:
        theirs = neighbor.get("articles")
        if theirs is None:
            continue          # 옛 라벨만 있는 이웃 — 비교 불가. 발동시키지 않는다
        if neighbor["similarity"] >= _SIMILARITY_THRESHOLD and set(theirs) != mine:
            client = config["configurable"]["client"]
            rebuttal = _generate_rebuttal(client, state["clause"], sorted(mine), neighbor)
            return {"redteam_note": rebuttal or _build_fallback_note(neighbor)}

    return {"redteam_note": ""}
