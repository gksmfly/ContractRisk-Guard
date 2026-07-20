# backend/agents/analysis_agent.py
"""Analysis Agent — GPT-4o Forward Labeling으로 조항 유형·근거 문구를 1차 분석한다.

실제 GPT 호출은 backend.fb_check.forward_labeling.run_forward()를 그대로
재사용한다(FB-Check 파이프라인과 동일한 프롬프트로 일관성을 유지하기 위함) —
이 노드는 그래프 상태를 읽고/쓰는 얇은 래퍼 역할만 한다.
"""

from langchain_core.runnables import RunnableConfig

from backend.agents.state import ClauseState
from backend.fb_check.forward_labeling import run_forward


def analysis_node(state: ClauseState, config: RunnableConfig) -> dict:
    client = config["configurable"]["client"]
    fwd = run_forward(client, state["clause"])
    if fwd is None:
        fwd = {"domain": "해당없음", "risk_level": "Low", "evidence_span": "", "reasoning": ""}

    domain = fwd.get("domain", "해당없음")
    if domain not in ("해지_조항", "책임제한_조항"):
        domain = "해당없음"

    return {
        "domain": domain,
        "evidence_span": fwd.get("evidence_span", ""),
        "reasoning": fwd.get("reasoning", ""),
    }
