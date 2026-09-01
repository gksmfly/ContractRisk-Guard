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
from backend.labeling.articles import ARTICLES, LAW_NAME
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
def _fallback_for(articles: list[str]) -> list[dict]:
    """검색이 아무것도 못 건졌을 때 붙이는 **참고 법령 안내.**

    ## 판정이 아니라 안내다

    배포 임계값에서 조 단위 성능은 상수와 구분되지 않는다(38.2% vs 상수 40.3%,
    CI [-7.7,+3.4] 미판정). **그래서 "이 조항은 제9조 위반"이라고 말하지 않는다** —
    사용자가 조문을 찾아볼 출발점만 제공한다. 정확도 주장이 아니므로 미판정이
    문제가 되지 않는 형태다.

    제품이 주장하는 것은 조항 지목이고 그건 측정돼 있다(조항 단위 재현 78.0% ·
    오경보 2.6%). 조는 그 위에 얹는 참고값이다.

    ## 법령 문구를 손으로 쓰지 않는다

    조문 설명은 `labeling/articles.py`에서 가져온다 — 그건 원문 법령 JSON에서 생성된다.
    예전 판본은 2-도메인 키에 설명을 직접 타이핑해 넣었는데, 법령을 손으로 옮겨 적으면
    틀려도 아무도 모른다(이 프로젝트에서 실제로 겪은 실패다).
    """
    out: list[dict] = []
    for a in articles:
        meta = ARTICLES.get(a)
        if not meta:
            continue
        out.append({
            "law": LAW_NAME,
            "article": a,
            # 조 제목 + 첫 호. 원문에서 그대로 온다.
            "description": meta["title"] + (f" — {meta['items'][0]}" if meta.get("items") else ""),
        })
    return out


def evidence_selection_node(state: ClauseState) -> dict:
    """화면에 붙일 근거를 고른다.

    ## 법령은 검색하지 않고 **예측한 조에서 바로 매핑한다** (2026-08-31)

    예전에는 법령도 RRF 검색 결과 top-K를 그대로 붙였다. 그 결과 서비스 이용약관
    해지 조항에 **민법 제658조(노무의 내용과 해지권)·제674조의3(여행 계약 해제)·
    상법 제168조의5(금융리스)** 가 "적용 법령"으로 나왔다. 화면에서 직접 확인했다.

    조를 예측하는 순간 조문 원문은 결정된다 — `articles.py`가 원문 법령 JSON에서
    생성하므로 검색이 개입할 이유가 없다. 법령 검색은 top-10에서 80%지만
    **top-K로 자르면 무관한 법이 섞여 들어온다.**

    ## 판례는 "참고 사례"로 격하한다

    판례 검색은 hit@5 = 14%다(무작위 5.3%). 화면에 5건을 붙이면 그중 관련 있는 게
    들어 있을 확률이 14%라는 뜻이다. 같은 화면에서 조 표시는 38.2%(상수와 미판정)라는
    이유로 판정형에서 안내형으로 낮췄는데, **14%짜리를 "적용 법령"으로 두는 것은
    그 신중함과 모순된다.** 근거가 아니라 참고 사례로 표시한다.

    `retrieval_candidates`는 계속 채워진다 — `evidence_verification`의 재검색 루프가
    그것을 본다. 바뀐 것은 **화면에 무엇을 근거로 내보내는가**다.
    """
    # 법령: 예측한 조에서 직접 매핑 (검색 미사용)
    law_basis = [LegalBasis(**b) for b in _fallback_for(state.get("model_articles") or [])]

    candidates = state.get("retrieval_candidates", {}) or {}
    law_top       = candidates.get("law", [])[:_FINAL_K]
    precedent_top = candidates.get("precedent", [])[:_FINAL_K]

    return {
        "legal_basis": law_basis,
        # 참고 사례 — **근거가 아니다**(hit@5 14%). 화면에서 그렇게 표시할 것.
        "precedent_refs": [candidate_to_legal_basis(c) for c in precedent_top],
        # **화면에 뭘 내보내는지와 별개다.** Dense·Sparse 양쪽에서 나온 후보가 있는지는
        # 검색 품질 신호이고 `evidence_verification`의 재검색 판단에 쓰인다. 법령을 표시에서
        # 뺐다고 해서, 또 판례가 하나도 없다고 해서 그 신호까지 버리면 재검색 루프가 눈이 먼다.
        "evidence_agreement": any(c.get("in_both") for c in law_top + precedent_top),
    }
