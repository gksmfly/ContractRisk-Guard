# backend/agents/graph.py
"""LangGraph 배선 — Analysis → Judgment → (근거·반박 브랜치).

    START → analysis(GPT) → judgment(모델) → ┬→ red_team → END
                                              └→ retrieval_strategy → evidence_selection
                                                 → evidence_verification → (재검색 최대 3회) → END

## 왜 judgment가 앞에 오나 (2026-08-31, 이전에는 병렬이었다)

**게이트 주체가 GPT에서 모델로 바뀌었다.** 예전에는 `analysis` 직후 GPT의 2-도메인 값
(`domain == "해당없음"`)으로 분기했는데, 조 multi-label 서빙으로 옮기면서 판단 주체가
분류 모델이 됐다. 라우팅을 안 고치면 **judgment가 실행되기도 전에 GPT 기준으로 끊긴다** —
게이트 이전이 반쪽이 된다.

## 병렬을 포기한 이유 — 이제 의존성이 **있다**

이전 docstring은 "두 브랜치는 서로 상태 의존성이 없다"고 적혀 있었다. **더 이상 아니다.**
`evidence_selection`이 `model_articles`를 읽어 조문 원문을 매핑한다(법령 검색을 안 쓴다).
병렬로 되돌리면 LangGraph에서 두 브랜치가 서로의 상태를 못 보므로 **`legal_basis`가
조용히 빈 목록이 된다** — 화면에 근거가 사라지는데 예외는 안 난다.

    되돌리지 말 것. 되돌리려면 evidence_selection이 model_articles를 안 읽게 먼저 바꿀 것.

### 직렬화 비용

KoELECTRA 추론이 검색의 임계 경로에 들어왔다. 다만 조항당 총 지연은 GPT 왕복(~10초)이
지배하고 인코더 forward는 로컬 110M 한 번(수십 ms)이라 실질 영향은 작다. 그리고 조를
하나도 못 지목한 조항은 **검색 브랜치를 아예 안 타므로** 오히려 아끼는 쪽이다.

## 분기 조건

모델이 조를 하나도 지목하지 않으면(`model_articles`가 비면) 두 브랜치 모두 건너뛰고
바로 끝난다 — `analyze.py`가 그 조항을 `OutOfScopeClause`로 돌려준다.
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
