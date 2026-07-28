# backend/agents/graph.py
"""6-agent 파이프라인의 LangGraph 배선.

Analysis 이후 두 브랜치가 병렬로 실행된다:
  - 판단 브랜치: Judgment → Red-team
  - 근거 브랜치: Retrieval Strategy → Evidence Selection → Evidence Verification
                (근거 부족 시 Retrieval Strategy로 최대 3회 재검색)

이 둘은 서로 상태 의존성이 없다 — Judgment는 evidence_span/clause만 읽고
retrieval_candidates·legal_basis를 쓰지 않고, Evidence Verification은
legal_basis·evidence_agreement만 읽고 risk_level·redteam_note를 쓰지 않는다
(state.py 필드별 read/write 주석 참고). 원래는 이 둘을 한 줄로 억지로 이어붙여서
직렬 실행했는데, 실제 의존성이 없으므로 fan-out(Analysis에서 두 브랜치로 분기)
시켜도 결과가 달라지지 않는다 — KoELECTRA GPU 추론과 DB 벡터 검색이 동시에
돌아가 지연시간이 줄어든다. 두 브랜치는 길이가 달라도(판단 브랜치는 고정 2단계,
근거 브랜치는 재검색 루프로 가변) 문제없다 — LangGraph는 각 브랜치가 독립적으로
END에 도달하는 걸 허용하고, 전체 invoke()는 모든 브랜치가 끝날 때까지 기다린다.

domain이 "해당없음"이면 두 브랜치 모두 건너뛰고 바로 끝난다.
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


def _route_after_analysis(state: ClauseState) -> list[str]:
    if state.get("domain") == "해당없음":
        return [END]
    return ["judgment", "retrieval_strategy"]  # 판단·근거 브랜치로 동시 분기(fan-out)


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
        ["judgment", "retrieval_strategy", END],
    )

    # 판단 브랜치 — 근거 브랜치와 독립적으로 끝까지 실행됨
    graph.add_edge("judgment", "red_team")
    graph.add_edge("red_team", END)

    # 근거 브랜치 — 재검색 루프 후 종료
    graph.add_edge("retrieval_strategy", "evidence_selection")
    graph.add_edge("evidence_selection", "evidence_verification")
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
