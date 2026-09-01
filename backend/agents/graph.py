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


def _route_after_judgment(state: ClauseState) -> list[str]:
    """**판단 주체가 모델이다** — 조를 하나도 못 지목하면 근거 수집으로 가지 않는다.

    예전에는 `analysis` 직후 GPT의 2-도메인 값(`domain == "해당없음"`)으로 끊었다.
    조 multi-label로 서빙을 바꾸면서 `analyze.py`의 게이트는 모델 출력으로 옮겼는데
    **그래프가 그보다 먼저 GPT 기준으로 끊고 있었다** — judgment가 아예 실행되지 않아
    게이트 이전이 반쪽이었다.

    지금은 judgment를 항상 돌리고 그 결과로 가른다. 비용은 거의 없다 — 110M 인코더
    forward 한 번(로컬)이고, 아낀 것은 검색·근거 브랜치 쪽이다.
    """
    if not state.get("model_articles"):
        return [END]
    return ["red_team", "retrieval_strategy"]  # 반박·근거 브랜치로 동시 분기(fan-out)


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

    # analysis(GPT) → judgment(모델) 순서다. 판단을 낸 뒤 그 결과로 분기하므로
    # `evidence_selection`에서 `model_articles`를 읽을 수 있다(예전 병렬 구조에서는
    # 두 브랜치가 서로의 상태를 못 봤다).
    graph.add_edge(START, "analysis")
    graph.add_edge("analysis", "judgment")
    graph.add_conditional_edges(
        "judgment",
        _route_after_judgment,
        ["red_team", "retrieval_strategy", END],
    )
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
