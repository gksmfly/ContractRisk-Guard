# backend/agents/

고정 6단계 LangGraph 파이프라인 — 조항 하나를 받아 **약관규제법 제6~14조 위반 소지 판단**, 관련 조문, 참고 판례, 편향 점검까지 마친 결과를 반환합니다.

> **"6-agent 멀티에이전트"가 아닙니다.** LLM이 흐름을 제어하지 않습니다 — 노드 순서는 고정이고, 유일한 동적 분기는 재검색 루프(임계값 체크)입니다.

---

## 배선 구조 (`graph.py`)

```
                                        ┌─→ Red-team ──────────────────────────┐
Analysis(GPT-4o) → Judgment(KoELECTRA) ─┤                                       ├─→ (최종 state)
                          │             └─→ Retrieval Strategy → Evidence Selection
                          │                   → Evidence Verification
                          │                        ↑____재검색(최대 3회)____|
                          └─ model_articles가 비면 두 브랜치 모두 건너뛰고 즉시 종료
```

- **게이트 주체가 모델입니다** (2026-08-31 변경). 예전에는 Analysis 직후 GPT의 2-도메인 값(`domain == "해당없음"`)으로 끊었는데, 조 multi-label 서빙으로 옮기면서 판단 주체가 분류 모델이 됐습니다. 라우팅을 안 고치면 **judgment가 실행되기도 전에 GPT 기준으로 끊깁니다**.
- **병렬을 포기했습니다 — 이제 의존성이 있습니다.** `evidence_selection`이 `model_articles`를 읽어 조문 원문을 매핑합니다(법령 검색을 안 씁니다). 병렬로 되돌리면 두 브랜치가 서로의 상태를 못 봐서 **`legal_basis`가 조용히 빈 목록이 됩니다** — 화면에 근거가 사라지는데 예외는 안 납니다. 되돌리려면 `evidence_selection`이 `model_articles`를 안 읽게 먼저 바꾸세요.
- 직렬화 비용은 작습니다 — 조항당 지연은 GPT 왕복(~10초)이 지배하고 인코더 forward는 로컬 110M 한 번(수십 ms)입니다. 조를 못 지목한 조항은 검색 브랜치를 아예 안 타므로 오히려 아끼는 쪽입니다.
- 근거 브랜치는 Evidence Verification이 "근거 불충분"이라 판단하면 Retrieval Strategy로 최대 3회 되돌아갑니다(매회 다른 전략으로 검색 범위 확대). **이 재검색 루프가 LangGraph를 선택한 근본 이유**입니다.
- 6개 노드 중 LLM을 호출하는 건 Analysis와 (충돌이 탐지된 경우에만) Red-team뿐입니다. 나머지는 전부 결정론적입니다.

---

## 파일 목록

### `state.py` — `ClauseState`
그래프 전체가 공유하는 상태(TypedDict). 각 필드를 어느 노드가 읽고/쓰는지가 병렬화 가능 여부를 결정합니다:

| 필드 | 쓰는 노드 | 읽는 노드 |
|---|---|---|
| `clause`, `articles`, `domain`, `evidence_span`, `reasoning` | Analysis | 전체 |
| `model_articles`, `needs_review`, `verified` | Judgment | 그래프 라우팅, Evidence Selection, Red-team, (최종 응답) |
| `retrieval_candidates` | Retrieval Strategy | Evidence Selection, Evidence Verification |
| `legal_basis`, `precedent_refs`, `evidence_agreement` | Evidence Selection | Evidence Verification, (최종 응답) |
| `redteam_note` | Red-team | (최종 응답) |
| `evidence_verified`, `retry_count`, `should_retry` | Evidence Verification | (라우팅 전용) |

`articles`(GPT 1차 라벨)와 `model_articles`(모델 판단)는 **별도 필드입니다** — 같은 이름을 쓰면 judgment 노드가 GPT 값을 덮어써서 `verified`가 자기 자신과의 비교가 됩니다. `domain`은 옛 2-도메인 파생값이고 **판단에 쓰이지 않습니다**(저장된 옛 결과와의 응답 호환용).

### `analysis_node` (`analysis_agent.py`)
GPT-4o Forward Labeling(`backend.fb_check.forward_labeling.run_forward()`) 재사용 — 조항 유형(domain)·근거 문구(evidence_span)·판단 이유(reasoning) 1차 분석. FB-Check 파이프라인과 동일 프롬프트로 일관성 유지. `config["configurable"]["client"]`로 OpenAI 클라이언트를 주입받는다.

### `retrieval_strategy_node` (`retrieval_strategy_agent.py`)
`backend.api.services.retrieval.fetch_candidates()` 호출부. 최초 시도(`retry_count=0`)에서는
법령 검색을 **약관규제법 제6~14조로 고정**한다 — 46청크 중 실질 규범은 이 9개뿐이고 나머지
37개는 심사청구·분쟁조정 같은 절차 조문이라 필터가 없으면 상위를 차지한다(top-5 66% → 81%).

**LLM 라우팅(EXAONE)은 제거됐다.** FTC 의결서 100건 실측에서 상수에 0-100으로 졌다:

    RRF(필터 없음)              18/100
    EXAONE top-2 라우팅          37/100
    약관규제법 + 민법 고정        23/100   ← 파티션을 하나만 더해도 폭락
    약관규제법 + EXAONE 예측 추가  40/100   ← 추가만 해도 해롭다
    약관규제법 고정              65/100   ← EXAONE만 맞은 케이스 0건, McNemar p<0.00001

이득의 정체는 "똑똑하게 고르기"가 아니라 **"좁히기"** 였다. 재현: `backend/eval/law_router_compare.py`.
한계 — 이 평가셋은 전부 FTC 약관 사건이라 약관규제법이 100% 정답이다. 재검색 시도마다 다른 전략:

| 회차 | top_k | pg_trgm 임계값 | 검색 범위 | 법령 필터 |
|---|---|---|---|---|
| 0(최초) | 6 | 0.10 | evidence_span, law/precedent 분리 | 약관규제법 제6~14조 고정 |
| 1 | 10 | 0.10 | 조항 전체로 확대 | 해제(재검색은 넓히는 게 취지라 좁히는 필터와 상충) |
| 2 | 16 | 0.05 | Sparse 재현율 확보 | 해제 |
| 3(마지막) | 24 | 0.05 | law/precedent 통합 검색 | 해제 |

### `evidence_selection_node` (`evidence_selection_agent.py`)
**법령은 검색하지 않고 `model_articles`에서 직접 매핑한다**(2026-08-31). 조가 정해지는 순간 조문 원문도 정해지므로 검색이 개입할 이유가 없다 — 검색으로 붙이던 시절에는 서비스 이용약관 해지 조항에 민법 제658조·제674조의3, 상법 제168조의5가 "적용 법령"으로 떴다.

**판례는 "참고 사례"로 격하한다.** RRF 순위 top-5를 그대로 쓰되(재랭킹 없음) `precedent_refs`로 분리해 내보낸다 — hit@5가 14%(무작위 5.3%)라 "적용 법령"과 같은 위계에 둘 수 없다.

재랭킹을 얹는 시도는 두 번 다 측정에서 졌다: Cross-Encoder(10.7% vs RRF 20.1%), 법원 심급 가중치(@2 6.4% vs 폐지 9.4%, McNemar p=0.035 — `"고등법원"` 키는 DB에 없어 한 번도 매칭된 적이 없었고, 대법원 가산 0.10은 RRF 점수 폭 전체(0.00738)의 13.5배라 사실상 "대법원 순 정렬"이었다).

### `judgment_node` (`judgment_agent.py`)
`models/article_v2`(max_len **512**) KoELECTRA 로드 → **위반 소지 조 multi-label** 판단(약관규제법 제6·7·8·9·10·11·12·14조. 제13조는 support 부족으로 접힘). 임계값은 학습 때 dev에서 확정한 조별 값(`thresholds.npy`)을 그대로 쓴다 — 여기서 다시 고르면 그게 평가셋 오염이다. **조항 원문(`clause`)**을 입력으로 씀 — 학습·채점과 같은 입력이다. 2026-09-01 이전에는 `evidence_span`(없으면 `clause`)을 넣었는데, 그건 span 증강으로 학습한 `models/v4` 시절 규칙이었다. 페어드 실측(136건)에서 조각 입력은 조항 재현을 **81.6% → 72.1% (-9.6%p [-16.9,-2.9])** 로 유의하게 떨어뜨린다. 게다가 span은 위반 조항의 53%·비위반 조항의 2%에만 있어 **입력이 정답과 상관한다** — 되돌리지 말 것 (`backend/eval/input_parity_eval.py`).

**위험도 3단계(High/Medium/Low)는 더 이상 내지 않는다**(2026-08-31). 조 taxonomy로 바꾸면서 risk의 gold를 정의할 방법이 없어 헤드를 일부러 뺐고, 그 위에 얹혀 있던 `confidence_band` 실측치도 v4 전용이라 옮길 수 없다. 지금 내는 것은 `model_articles`(빈 리스트 가능)와 `needs_review`(지목 여부 = 이진)다.

실측(배포 임계값, `backend/eval/article_gold_eval.py`, clean gold n=255): 조항 단위 재현 **78.4%**(`article_v2`. `article_v1`은 78.0%). 조 단위 per-sample F1은 38.5%로 상수 기준선 36.1% 대비 **미판정** — 그래서 화면에서 **조 이름은 단정하지 않고 참고로만** 붙인다.

**오경보율은 아직 없다.** 예전에 여기 "오경보 2.6%"가 함께 적혀 있었는데, 그 값은 음성 풀의 정답이 `agreed_articles`(= GPT 라벨) 그 자체라 **순환**이었다 — 모델이 GPT보다 잘 찾아 짚은 것도 오경보로 세어진다. `disagree_with_gpt`로 개명했다. **이 연구 범위에서는 측정하지 않는다.** 독립 준거를 만들려면 법률 비전문가가 약관 조항 수백 건을 이진 판단해야 하는데, 그 판단 자체의 신뢰도를 담보할 방법이 없어 **한계로 남긴다.** 평가셋(`data/eval/prevalence/evalset_v1.json`, 149건)은 얼려 두었으므로 나중에 판단자가 생기면 그대로 쓸 수 있다. 상세는 `models/README.md`.

### `red_team_node` (`red_team_agent.py`)
충돌 사례 탐색은 LLM 미호출 — `clean_clauses` 임베딩 최근접 이웃 중 유사도 0.75 이상인데 **다른 조**로 판정된 사례가 있으면 편향 의심으로 간주한다. 임계값은 leave-one-out 실험으로 검증(탐지율 2.3%). 탐지된 경우에만 LLM(`gpt-4o-mini`)을 호출해 반박 근거를 생성 — 출력 스키마에 판정 필드가 없어 LLM이 판단 자체를 못 바꾸도록 구조적으로 막아둠. 호출 실패 시 템플릿 문구로 폴백.

> ⚠️ **현재 사실상 비활성입니다.** 비교 축이 `risk_level`에서 조 목록으로 바뀌었는데 `clean_clauses` 테이블은 아직 옛 라벨로 적재돼 있습니다. 조 라벨이 없는 이웃은 건너뛰므로 아무것도 발동하지 않고, 한 번 확인하면 이후 조항에서는 검색 자체를 건너뜁니다(조항마다 임베딩 검색이 도는 낭비를 막기 위함). 되살리려면 새 `clean.jsonl`(조 multi-label)로 재적재하고 프로세스를 재시작하세요. **틀린 축으로 경고하느니 침묵하는 편이 낫다**는 판단입니다.

### `evidence_verification_node` (`evidence_verification_agent.py`)
`legal_basis`가 있고 `evidence_agreement`(Dense·Sparse 양쪽 다 찾은 근거인지)가 True면 충분하다고 판단. 아니면 `retry_count`를 올리고 재검색 라우팅(`MAX_RETRIES=3`). 코사인 유사도 기반 신뢰도 신호는 사전 실험에서 정답 적중률과 상관관계가 거의 없어(hit 0.567 vs miss 0.559) 채택 안 함.

---

## 테스트

노드별 유닛 테스트: `tests/agents/test_retrieval_strategy_agent.py`, `test_evidence_selection_agent.py`, `test_evidence_verification_agent.py`, `test_red_team_agent.py`. 그래프 전체(실제 GPT+GPU+DB) 통합 테스트는 `tests/analyze/test_analyze_integration.py`(`-m integration`). 자세한 실행법은 `tests/README.md` 참고.
