# 법조문 검색 아키텍처 대안 조사 — 최종 결과 (2026-08-06)

## 배경

`backend/eval/lightrag_compare_final.py` 최종 비교(법령 전체 3,323청크)에서 LightRAG가
RRF보다 수치상 우세했지만(20% vs 12%, McNemar p=0.115 유의성 미확보) 근본원인을 더
파보니, **LightRAG 자체가 코퍼스 확장에 따라 성능이 퇴화**한다는 게 확인됐다(76개
공유 쿼리 직접비교, 9:0 flip, 이항검정 p=0.004 — 상세는 memory
`project_lightrag_vs_rrf.md` 참고).

이 진단에 따라 그래프 없는 대안들과 그래프 스코핑 대안까지 **전부 같은 ground
truth(FTC 근거_법령, 법령 전체 3,323청크, seed=42, 동일 100건)로 실측**했다. "판단만
하고 넘어가지 말고 다 해보자"는 방침으로, 처음엔 API 제약·개념 중복을 이유로 스킵했던
GraphRAG 스코핑·SEAL-RAG도 실제 구현해서 넣었고, 최고 성능 조합·다른 로컬 모델(Qwen2.5-14B)·
다른 임베딩 모델(LBOX 법률특화)까지 전부 테스트했다. 총 12개 실험.

## 종합 결과표 (적중률 높은 순)

| # | 방법 | 결과 | 베이스라인 | 비고 |
|---|---|---|---|---|
| 1 | **RAPTOR-lite**(EXAONE 라우팅) | **33%** | RRF 8% | p<0.0001 |
| 1 | **LegalMALR-lite**(EXAONE 쿼리 재구성) | **33%** | RRF 8% | p<0.0001 |
| 3 | GraphRAG 스코핑 근사(EXAONE 라우팅 + chunks_vdb 필터) | 31% | LightRAG raw 20% | OpenAI 임베딩 사용(confound) |
| 4 | 종합 콤보(EXAONE 재구성+라우팅+재랭킹) | 28% | RRF 8% | **개별 최고(33%)보다 낮음 — 의외의 결과** |
| 4 | LegalMALR-lite(Qwen2.5-14B) | 28% | RRF 8% | EXAONE(7.8B,한국어특화)보다 낮음 |
| 6 | RAPTOR-lite(Qwen2.5-14B) | 26% | RRF 8% | 마찬가지로 EXAONE보다 낮음 |
| 7 | LightRAG(참고, 전체 인덱스) | 20% | RRF 12%(pool5) | 코퍼스 커질수록 자체 퇴화 |
| 8 | 1+2 조합(도메인필터+재랭킹) | 17% | RRF 8% | p=0.0225 — 그래프·LLM 없는 방법 중 유일하게 유의 |
| 8 | Contextual Retrieval(KoE5, 결정론적 프리픽스) | 17% | Dense 12% | p=0.125 |
| 10 | Cross-Encoder 재랭킹 | 16% | RRF(pool20) 11% | p=0.125 |
| 11 | SEAL-RAG-lite(1회 스왑) | 15% | RRF(pool20) 11% | 전체 재정렬(16%)보다 보수적이라 약간 낮음 |
| 12 | 도메인 필터링 Hybrid | 13% | RRF 8% | p=0.0625, 손해 케이스 0건 |
| 13 | LBOX 법률 임베딩(구조인식 헤드 미지원) | 5% | KoE5 12% | 아키텍처 호환 문제로 핵심 기능 미검증 |

## 5가지 핵심 발견

**1) 로컬 LLM 쿼리 이해가 압도적으로 우수하다.** EXAONE-3.5-7.8B(API 비용 0)로 쿼리를
재구성하거나 법령을 먼저 라우팅하는 두 방법이 RRF 8%→33%로 그래프 없이도 LightRAG(20%)를
크게 앞선다(p<0.0001).

**2) 다 합친다고 더 좋아지지 않는다.** 재구성+라우팅+재랭킹을 한 파이프라인에 다 넣은
"종합 콤보"는 28%로 개별 최고(33%)보다 오히려 낮았다. 가능한 원인 두 가지를 확인했다:
(a) 한 번의 EXAONE 호출에 두 작업(재구성+라우팅)을 동시에 요청해서 각각의 품질이
단일 작업 전용 호출보다 떨어졌을 가능성, (b) Cross-Encoder 재랭킹이 이미 좋은 순서를
오히려 흐트러뜨렸을 가능성 — SEAL-RAG-lite(보수적 1회 스왑, 15%)가 전체 재정렬(16%)보다
낮았던 것과 같은 패턴(재랭킹이 무조건 이득은 아님)이 반복됐다. "요소를 더 넣으면 항상
좋아진다"는 가정이 이 코퍼스에서는 성립하지 않는다.

**3) 모델 크기보다 한국어 특화가 더 중요했다.** Qwen2.5-14B(14B, 다국어 범용)로 같은
프롬프트를 재실행하니 재구성 28%/라우팅 26%로 EXAONE-3.5-7.8B(7.8B, 한국어 특화,
33%/33%)보다 둘 다 낮았다. 거의 2배 큰 모델인데도 졌다 — 이 태스크에서는 한국어 법률
텍스트에 대한 언어 특화가 파라미터 수보다 중요하다는 뜻.

**4) "그래프 자체"보다 "스코핑 여부"가 핵심이었다.** LightRAG의 원래 그래프 순회는 못
바꿨지만, 그래프가 만든 최종 청크 벡터스토어를 EXAONE 라우팅으로 필터링했더니 20%→31%로
크게 올랐다. LightRAG이 나쁜 게 아니라 "코퍼스 전체를 무차별로 검색하는 것"이 나쁘다는
진단(그래프 희석)이 다시 한번 확인됨. 단 이 실험은 LightRAG 설정상 OpenAI 임베딩을 쓰기
때문에 다른 옵션(KoE5)과 임베딩 모델 자체가 다르다는 confound가 있다.

**5) 법률 도메인 특화 모델이라고 항상 이기는 건 아니다.** LBOX(한국 리걸테크 회사)의
법률 특화 임베딩 모델을 테스트하려 했으나, 체크포인트 아키텍처(`RobertaForSAILER`)가
표준 transformers에 없는 커스텀 클래스라 핵심 기능(구조인식 판결/이유 헤드)이 로드 시
버려지고 pooler도 무작위 초기화됐다 — 남은 공유 인코더 바디만으로 테스트한 결과 5%로
범용 KoE5(12%)보다도 낮았다. "법률 특화"라는 이름만으로 성능을 가정하면 안 된다는 사례.

---

## 방법별 상세

### RAPTOR-lite / LegalMALR-lite (EXAONE) — `raptor_lite_compare.py`, `legalmalr_lite_compare.py`
로컬 EXAONE-3.5-7.8B-Instruct(few-shot 2건, API 비용 0)로 각각 법령 라우팅/쿼리 재구성.
RRF 8%→33%, 두 방법 다 p<0.0001. **caveat**: few-shot이 "약관의 규제에 관한 법률"을
출력하도록 유도했는데 이 FTC 평가셋 100/100건이 전부 그 법을 정답에 포함 — 블라인드
편향인지 확인한 결과, 이 문자열을 명시한 그룹은 원문 그대로의 RRF 기준으로도 이미 더
쉬운 케이스였음(모델이 실제 신호를 읽고 있다는 근거). 절대 수치는 이 FTC-약관법 특화
벤치마크의 산물일 수 있어 방향성만 신뢰.

### GraphRAG 스코핑 근사 — `lightrag_scoped_compare.py`
LightRAG의 공개 API엔 소스별 그래프 순회 제한이 없어(이미 확인됨) 진짜 서브그래프
스코핑은 여전히 안 함. 대신 그래프가 만든 청크 벡터스토어(`rag.chunks_vdb`)를 top-30으로
넉넉히 조회한 뒤 EXAONE 라우팅 예측 법령에 속하는 청크만 남겨 top-5. 스코핑 없음 20%
(기존 LightRAG 최종 결과와 거의 일치 — 일관성 확인됨) → 스코핑 31%.

### 종합 콤보 — `combo_best_compare.py`
EXAONE 1회 호출로 쿼리 재구성+법령 라우팅 동시 예측 → 예측 법령 파티션만 재구성 쿼리로
검색 → Cross-Encoder 재랭킹. RRF 8%→28% — 개별 최고 방법(33%)보다 낮음(핵심 발견 2번).

### Qwen2.5-14B 변형 — `qwen_variant_compare.py`
`legalmalr_lite_compare.py`/`raptor_lite_compare.py`와 프롬프트·few-shot·ground truth
완전 동일, 모델만 EXAONE→Qwen2.5-14B 교체. 재구성 28%, 라우팅 26% — 둘 다 EXAONE보다 낮음.

### 1+2 조합 — `combo_1_2_compare.py`
도메인 파티션별 후보 확보(옵션2) + Cross-Encoder 재랭킹(옵션1). RRF 8%→17%,
**그래프도 LLM도 없는 방법 중 유일하게 p<0.05 확보(p=0.0225)**.

### Contextual Retrieval — `contextual_retrieval_compare.py`
청크 본문 앞에 "[법령: {law_name} 제{article_no}조]" 프리픽스(결정론적, LLM 불필요).
Dense-only: 프리픽스 없음 12% → 있음 17%, p=0.125.

### Cross-Encoder 재랭킹 / SEAL-RAG-lite — `rerank_compare.py`, `seal_rag_lite_compare.py`
같은 후보 풀(top-20)·같은 모델(BAAI/bge-reranker-v2-m3)로 "전체 재정렬"(16%) vs "RRF
순서 보존 + 최저점 1개만 교체"(15%) 비교 — 전체 재정렬이 근소하게 우세.

### 도메인 필터링 Hybrid — `domain_filter_compare.py`
16개 law_name 파티션(약관규제법 43청크 vs 민법 1,305청크 등 극단적 불균형)별 top-8을
**실제 유사도 점수**로 재병합. RRF 8%→13%, 손해 케이스 0건.
> 첫 구현은 파티션별 **순위**로 병합해서 0/50이라는 명백히 잘못된 결과가 나왔다 — 모든
> 파티션 1위가 관련성과 무관하게 동일 점수를 받는 버그. 실제 점수 기반으로 고쳐 재실행.

### LBOX 법률 임베딩 — `lbox_embedding_compare.py`
"핵심 발견 5" 참고. 5% (KoE5 12%보다 낮음), 구조인식 헤드 미검증.

---

## 다음 결정 (사용자 확인 필요)

최고 성능은 여전히 RAPTOR-lite/LegalMALR-lite(EXAONE, 33%)다. 운영 반영 시 쿼리마다
로컬 LLM 추론(EXAONE 7.8B, GPU 상주 필요)이 추가되는 트레이드오프는 그대로 남아있다.
1+2 조합(17%, LLM 없음)이 유일하게 통계적으로 유의하면서 운영 부담이 거의 없는 대안이다.
아키텍처 전환 여부는 아직 결정되지 않았다 — memory `project_lightrag_vs_rrf.md`,
`project_retrieval_alternatives_eval.md` 참고.
