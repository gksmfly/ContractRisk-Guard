# backend/agents/

6-agent LangGraph 파이프라인 — 조항 하나를 받아 domain·risk_level 판단, 법적 근거, 편향 점검까지 마친 결과를 반환합니다.

---

## 배선 구조 (`graph.py`)

```
                    ┌─→ Judgment → Red-team ──────────────────┐
Analysis(GPT-4o) ───┤                                          ├─→ (병합된 최종 state)
                    └─→ Retrieval Strategy → Evidence Selection → Evidence Verification
                                    ↑______________재검색(최대 3회)_______________|
```

- Analysis 이후 **판단 브랜치**(Judgment→Red-team)와 **근거 브랜치**(Retrieval Strategy→Evidence Selection→Evidence Verification)가 병렬로 실행됩니다 — 서로 상태 의존성이 없어서(판단 브랜치는 검색 결과를 안 쓰고, 근거 브랜치는 판단 결과를 안 씀) fan-out해도 결과가 달라지지 않습니다. 실측(타임스탬프 비교)으로 실제 동시 실행 확인됨 — 직렬이면 걸릴 시간의 절반 정도로 단축.
- domain이 "해당없음"이면 두 브랜치 모두 건너뛰고 즉시 종료.
- 근거 브랜치는 Evidence Verification이 "근거 불충분"이라 판단하면 Retrieval Strategy로 최대 3회 되돌아갑니다(매회 다른 전략으로 검색 범위 확대). **이 재검색 루프가 LangGraph를 선택한 근본 이유**입니다 — 병렬 실행은 부수적으로 고친 설계 부채이지, 존재 이유가 아닙니다.
- 6개 노드 중 실제로 LLM을 호출하는 건 Analysis뿐입니다. 나머지(Retrieval Strategy/Evidence Selection/Judgment/Red-team/Evidence Verification)는 전부 결정론적 알고리즘(Dense+Sparse 검색, RRF 재랭킹, KoELECTRA 추론, 임베딩 최근접 이웃, 임계값 체크)입니다 — "멀티에이전트"라는 이름의 정당성은 병렬성이나 각 노드의 자율적 LLM 추론이 아니라 조건부 동적 제어 흐름(재검색 루프)에 있습니다.

---

## 파일 목록

### `state.py` — `ClauseState`
그래프 전체가 공유하는 상태(TypedDict). 각 필드를 어느 노드가 읽고/쓰는지가 병렬화 가능 여부를 결정합니다:

| 필드 | 쓰는 노드 | 읽는 노드 |
|---|---|---|
| `clause`, `domain`, `evidence_span`, `reasoning` | Analysis | 전체 |
| `retrieval_candidates` | Retrieval Strategy | Evidence Selection |
| `legal_basis`, `evidence_agreement` | Evidence Selection | Evidence Verification |
| `risk_level`, `verified` | Judgment | Red-team, (최종 응답) |
| `redteam_note` | Red-team | (최종 응답) |
| `evidence_verified`, `retry_count`, `should_retry` | Evidence Verification | (라우팅 전용) |

### `analysis_node` (`analysis_agent.py`)
GPT-4o Forward Labeling(`backend.fb_check.forward_labeling.run_forward()`) 재사용 — 조항 유형(domain)·근거 문구(evidence_span)·판단 이유(reasoning) 1차 분석. FB-Check 파이프라인과 동일 프롬프트로 일관성 유지. `config["configurable"]["client"]`로 OpenAI 클라이언트를 주입받는다.

### `retrieval_strategy_node` (`retrieval_strategy_agent.py`)
`backend.api.services.retrieval.fetch_candidates()` 호출부. 최초 시도(`retry_count=0`)에서는
검색 전에 `query_router.route_law_names()`(로컬 EXAONE-3.5-7.8B, API 비용 0)로 조항이
어느 법령에 해당할지 먼저 예측해 법령 검색을 그 법령들로 좁힌다 — 법령 코퍼스가
43청크(약관규제법)~1,305청크(민법)로 불균형해서 필터 없이 전체를 경쟁시키면 소수
법령이 밀리는 문제를 완화한다(`backend/eval/retrieval_alternatives_survey.md`의
RAPTOR-lite 실측: RRF 8%→33%, McNemar p<0.0001). 재검색 시도마다(`retry_count`) 다른
전략 사용:
| 회차 | top_k | pg_trgm 임계값 | 검색 범위 | 법령 라우팅 |
|---|---|---|---|---|
| 0(최초) | 6 | 0.10 | evidence_span, law/precedent 분리 | 있음 |
| 1 | 10 | 0.10 | 조항 전체로 확대 | 없음(필터 해제 — 1회차 라우팅이 틀렸을 수 있어 넓게 재검색) |
| 2 | 16 | 0.05 | Sparse 재현율 확보 | 없음 |
| 3(마지막) | 24 | 0.05 | law/precedent 통합 검색 | 없음 |

### `evidence_selection_node` (`evidence_selection_agent.py`)
후보 풀을 top-2로 재랭킹. RRF 순위를 그대로 쓰되 판례에 한해 법원 심급 가중치(대법원 +0.10, 고등법원 +0.05) 가산. Cross-Encoder 재랭킹은 사전 실험(10.7% vs 20.1% 정답 적중률)에서 성능 저하로 미채택. 후보가 없으면 `LEGAL_BASIS_FALLBACK`(도메인별 하드코딩 법조문) 사용.

### `judgment_node` (`judgment_agent.py`)
`models/v4` KoELECTRA 로드 → domain·risk_level 최종 판단. `evidence_span`(없으면 `clause`로 폴백)을 입력으로 씀 — v4가 evidence_span 길이(평균 40자대) 위주로 학습됐기 때문. 검색 기반 방식(KoE5+GPT-4o-mini)으로 교체할지는 미결정 — `models/README.md` 참고.

### `red_team_node` (`red_team_agent.py`)
LLM 미호출. `clean_clauses`(FB-Check 검증 478건) 임베딩 최근접 이웃 중 유사도 0.75 이상인데 risk_level이 다른 사례가 있으면 편향 의심 노트를 남긴다. 임계값은 leave-one-out 실험으로 검증(탐지율 2.3%, 표본 확인 결과 실제 의미 있는 차이 포착).

### `evidence_verification_node` (`evidence_verification_agent.py`)
`legal_basis`가 있고 `evidence_agreement`(Dense·Sparse 양쪽 다 찾은 근거인지)가 True면 충분하다고 판단. 아니면 `retry_count`를 올리고 재검색 라우팅(`MAX_RETRIES=3`). 코사인 유사도 기반 신뢰도 신호는 사전 실험에서 정답 적중률과 상관관계가 거의 없어(hit 0.567 vs miss 0.559) 채택 안 함.

---

## 테스트

노드별 유닛 테스트: `tests/agents/test_retrieval_strategy_agent.py`, `test_evidence_selection_agent.py`, `test_evidence_verification_agent.py`, `test_red_team_agent.py`. 그래프 전체(실제 GPT+GPU+DB) 통합 테스트는 `tests/analyze/test_analyze_integration.py`(`-m integration`). 자세한 실행법은 `tests/README.md` 참고.
