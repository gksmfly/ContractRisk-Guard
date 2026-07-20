# backend/agents/graph.py
"""6-agent 파이프라인의 LangGraph 배선.

Analysis → Retrieval Strategy → Evidence Selection → Judgment → Red-team →
Evidence Verification 순서로 6개 노드가 모두 연결되어 있다. domain이
"해당없음"이면 검색·판단 단계를 거치지 않고 바로 끝나고, Evidence Verification이
근거가 부족하다고 판단하면 Retrieval Strategy로 최대 3회 되돌아간다(매회 다른
전략으로 검색 범위를 넓힘 — retrieval_strategy_agent.py 참고). 이 재검색 루프가
애초에 LangGraph를 선택한 이유다.
"""

from langgraph.graph import END, START, StateGraph

from backend.agents.analysis_agent import analysis_node
from backend.agents.evidence_selection_agent import evidence_selection_node
from backend.agents.evidence_verification_agent import evidence_verification_node
from backend.agents.judgment_agent import judgment_node
from backend.agents.red_team_agent import red_team_node
from backend.agents.retrieval_strategy_agent import retrieval_strategy_node
from backend.agents.state import ClauseState

_compiled_graph = None


def _route_after_analysis(state: ClauseState) -> str:
    return END if state.get("domain") == "해당없음" else "retrieval_strategy"


def _route_after_verification(state: ClauseState) -> str:
    return "retrieval_strategy" if state.get("should_retry") else END


def build_graph():
    graph = StateGraph(ClauseState)
    graph.add_node("analysis", analysis_node)
    graph.add_node("retrieval_strategy", retrieval_strategy_node)
    graph.add_node("evidence_selection", evidence_selection_node)
    graph.add_node("judgment", judgment_node)
    graph.add_node("red_team", red_team_node)
    graph.add_node("evidence_verification", evidence_verification_node)

    graph.add_edge(START, "analysis")
    graph.add_conditional_edges(
        "analysis",
        _route_after_analysis,
        {"retrieval_strategy": "retrieval_strategy", END: END},
    )
    graph.add_edge("retrieval_strategy", "evidence_selection")
    graph.add_edge("evidence_selection", "judgment")
    graph.add_edge("judgment", "red_team")
    graph.add_edge("red_team", "evidence_verification")
    graph.add_conditional_edges(
        "evidence_verification",
        _route_after_verification,
        {"retrieval_strategy": "retrieval_strategy", END: END},
    )

    return graph.compile()


def get_graph():
    """컴파일된 그래프를 모듈 레벨에 캐싱해서 반환한다 (요청마다 재컴파일하지 않도록)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
