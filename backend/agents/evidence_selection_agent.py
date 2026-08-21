# backend/agents/evidence_selection_agent.py
"""Evidence Selection Agent — Retrieval Strategy가 넘긴 후보 풀을 재랭킹해 최종 근거(top-2)를 고른다.

법령·판례 모두 **RRF 융합 순위를 그대로** 쓴다. 재랭킹을 얹는 시도는 두 번 다 실패했다:

1. Cross-Encoder(BAAI/bge-reranker-v2-m3) — 사전 실험(clean_clauses 478건)에서 RRF-only
   보다 낮았다(10.7% vs 20.1%).
2. **법원 심급 가중치 — 2026-08-16 측정으로 폐지**(아래).

## 법원 심급 가중치를 뺀 이유 (2026-08-16)

원래는 "대법원 판례가 더 권위 있는 근거"라는 도메인 지식으로 `{대법원: 0.10, 고등법원: 0.05}`를
RRF 점수에 가산했다. 판례 검색 정확도를 처음 측정하면서
(`backend/eval/precedent_retrieval_compare.py` — FTC 근거_법령 ↔ 판례 참조조문 조인
ground truth, n=298) 두 가지가 드러났다:

- **`"고등법원"` 키는 한 번도 매칭된 적이 없다.** DB의 `court` 값은 58종이고
  (`서울고법`·`부산고등법원`·`대전고법(청주)` …) 정확히 `"고등법원"`인 값은 없다.
  완전일치 `.get()`이라 고등법원급 3,452청크(11.4%)가 전부 가산 0이었다.
- **대법원 가산 0.10은 "소폭"이 아니었다.** RRF 1위~51위의 점수 폭 전체가 0.00738인데
  가산이 0.10이라 **13.5배**다. 51위 대법원 판례가 1위 고법 판례를 무조건 이겼고,
  대법원이 판례의 68.5%(20,646청크)라 사실상 "대법원 순 정렬"이었다.

실측에서 가중치가 있는 쪽이 **모든 구간에서 더 나빴다**(폐지 vs 현행: @2 9.4% vs 6.4%,
@5 17.1% vs 14.8%, @10 23.8% vs 20.8%, @20 31.2% vs 27.5%). 운영이 실제 노출하는 @2에서
McNemar p=0.035. 작은 가중치(0.0003·0.001)도 시도했으나 폐지 대비 이득이 없었고,
큰 값(0.01·0.1)은 현행과 똑같이 나빴다.

"권위 있는 판례"는 법적 설득력의 기준이지 **검색 관련도의 기준이 아니다** — 관련도 높은
하급심을 밀어내는 부작용이 이득을 넘었다. 도메인 지식이라도 측정에서 지면 뺀다
(Cross-Encoder를 기각한 것과 같은 기준). 전체 수치·재현 명령어는
`backend/eval/measurement_findings_2026-08-16.md` 참고.
"""

from backend.agents.state import ClauseState
from backend.api.schemas import LegalBasis
from backend.api.services.retrieval import candidate_to_legal_basis

# 최종 노출 개수. 2였다가 5로 올렸다 — 약관규제법 고정 검색에서 정답 조문의 첫 등장
# 순위 중앙값이 7위라, top-2로 자르면 검색이 찾아온 정답을 대부분 버린다.
#
#   FTC 의결서 100건, 약관규제법 고정 검색 기준 적중률
#     top-1  35%   top-2  43%   top-3  54%   top-5  66%   top-10  80%   top-20  88%
#
# 5로 둔 것은 화면에 근거를 몇 개까지 보여줄지의 절충이다 — 재현율만 보면 10이 더 낫다.
# 재현: `backend/eval/law_router_compare.py`
_FINAL_K = 5

# 후보가 하나도 없을 때(DB 미가동·미적재 등)만 쓰는 최종 fallback.
LEGAL_BASIS_FALLBACK: dict[str, list[dict]] = {
    "해지_조항": [
        {"law": "약관규제법", "article": "제9조",      "description": "고객에게 부당하게 불리한 해제·해지권 부여 조항 무효"},
        {"law": "민법",      "article": "제543~553조", "description": "계약 해제·해지의 일반 요건 및 효과"},
    ],
    "책임제한_조항": [
        {"law": "약관규제법", "article": "제7조",      "description": "사업자 책임 부당 면제·제한 조항 무효"},
        {"law": "민법",      "article": "제750~766조", "description": "불법행위 손해배상 책임"},
    ],
    "해당없음": [],
}


def evidence_selection_node(state: ClauseState) -> dict:
    # 법령·판례 모두 RRF 순서 그대로 top-K (재랭킹 없음 — 위 docstring 참고)
    candidates = state.get("retrieval_candidates", {}) or {}
    law_top       = candidates.get("law", [])[:_FINAL_K]
    precedent_top = candidates.get("precedent", [])[:_FINAL_K]
    selected = law_top + precedent_top

    if not selected:
        fallback = LEGAL_BASIS_FALLBACK.get(state.get("domain", "해당없음"), [])
        return {"legal_basis": [LegalBasis(**b) for b in fallback], "evidence_agreement": False}

    legal_basis = [candidate_to_legal_basis(c) for c in selected]
    evidence_agreement = any(c.get("in_both") for c in selected)
    return {"legal_basis": legal_basis, "evidence_agreement": evidence_agreement}
