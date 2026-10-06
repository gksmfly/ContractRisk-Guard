# backend/eval/

모델·검색·라벨 품질을 **측정**하는 스크립트 모음입니다. Ground-truth 데이터셋 구축은 `backend/scripts/rebuild_ftc_ground_truth.py`·`extract_precedent_ground_truth.py`(수집 성격이라 `scripts/`에 위치)가 담당하고, 이 디렉토리는 그 위에서 가설을 하나씩 재는 코드만 둡니다.

## 먼저 읽을 것

- **확정 수치는 스크립트가 아니라 [`reported_numbers.md`](reported_numbers.md) 한 곳에만 있습니다.** 스크립트 docstring이나 옛 문서의 숫자와 다르면 `reported_numbers.md`가 맞습니다.
- 각 스크립트는 특정 시점의 가설 하나를 재려고 쓴 **일회성 실험 기록**입니다. 다시 돌리기 전에 `reported_numbers.md` §4(축 상태)에서 그 축이 이미 닫혔는지 확인하세요.
- 각 파일 맨 위 docstring에 "왜 재는가 / 판정 기준 / 실행법"이 있습니다. 아래 표는 그 첫 줄을 요약한 것입니다.
- 날짜는 git에 처음 들어간 날입니다(파일 수정 시각은 일괄 이동 때문에 믿을 수 없음).

## 폴더 구조

```
backend/eval/
├── *.py           지금 쓰는 스크립트(1~7번 섹션)
├── common.py      여러 스크립트가 공유하는 FTC 근거_법령 정답셋 로더
└── archive/       결론이 났거나 수치가 무효가 된 스크립트(8~9번 섹션) — 기록용
```

`archive/`의 스크립트도 import와 실행은 됩니다(`python -m backend.eval.archive.<이름>`). 결과 파일은 여전히 `data/eval/`에 씁니다.

---

## 주제별 파일 목록

### 1. 실서비스 검증·오경보 통제 — 최근 작업

| 파일 | 날짜 | 무엇을 재나 |
|---|---|---|
| `real_tos_eval.py` | 미커밋 | 실제 기업 이용약관(`tos_targets.tsv`)을 `article_v2` 인코더로 판정해 조항 지목률 측정 |
| `style_transfer_fpr.py` | 미커밋 | B4 — 문체 교락 통제: 오경보가 내용이 아니라 문체 때문인가 |
| `dual_channel_recall.py` | 미커밋 | B5 — 두 채널 결합 재현율을 운영 모델(gpt-4o-mini) 기준으로 재측정 |
| `gpt_channel_fpr.py` | 09-12 | GPT(forward) 채널이 표준계약서를 얼마나 짚는가 — 두 채널 게이트의 전제 검사 |

### 2. 디코더 sLLM(LoRA) vs 인코더

| 파일 | 날짜 | 무엇을 재나 |
|---|---|---|
| `sllm_lora_eval.py` | 09-12 | 디코더 sLLM에 LoRA를 붙여 조 multi-label을 `article_v2`와 같은 잣대로 채점 |
| `sllm_vs_encoder.py` | 09-12 | EXAONE LoRA와 `article_v2`를 같은 255건에서 짝지어 비교 |
| `threshold_match_sweep.py` | 09-12 | `article_v2` 임계값을 낮춰 sLLM과 같은 재현율로 맞춘 뒤 오경보 비교 |
| `serving_cost_compare.py` | 09-19 | 인코더와 sLLM의 서빙 비용(지연·메모리)을 같은 입력으로 측정 |

### 3. 조 multi-label 모델 채점·학습 검증 (`article_v2` 계열)

| 파일 | 날짜 | 무엇을 재나 |
|---|---|---|
| `article_gold_eval.py` | 08-31 | 외부 정답(FTC 근거_법령)으로 채점하는 **최종 성적표** |
| `record_f1_eval.py` | 08-31 | 건별 집합 F1로 모델을 상수 기준선과 비교(학습 로그 macro F1과 단위가 다름) |
| `teacher_on_gold.py` | 08-31 | teacher(gpt-4o)를 gold 327건에 돌려 천장 측정 |
| `confound_articles.py` | 08-31 | 출처 교락 측정(조 multi-label 판본) — 학습 전에 반드시 실행 |
| `confound_analysis.py` | 08-21 | 라벨-출처 교락 정량화 — 점수 중 "코퍼스 판별" 몫 |
| `source_signal_probe.py` | 08-21 | 출처 판별이 서식 때문인가 내용 때문인가(정규화 단계별) |
| `fixedsplit_verify.py` | 08-21 | 누수를 고친 분할로 재학습한 모델이 지름길 없이도 작동하는지 |
| `risk_scheme_compare.py` | 08-21 | 3-class vs 2-class — Medium을 빼면 High/Low 판별이 나아지는가 |
| `ensemble_compare.py` | 08-21 | KoELECTRA 시드 앙상블(기각됨 — 재시도 금지) |
| `confidence_calibration.py` | 08-21 | KoELECTRA softmax 확률이 실제 정답률과 상관있는가 |
| `maxlen_harm_check.py` | 09-01 | `max_len` 변경이 해를 끼치지 않는지 사전 등록 규칙으로 판정 |
| `prompt_block_compare.py` | 08-31 | 조문 블록 절제 실험 A/A'/B/C 페어드 채점(기준: `prompt_block_ablation.md`) |

### 4. 라벨 품질·FB-Check

| 파일 | 날짜 | 무엇을 재나 |
|---|---|---|
| `label_pilot.py` | 08-21 | LLM이 지목한 약관규제법 조가 공정위 확정과 맞는가(⚠ gpt-4o 사용 — 운영 `.env`는 mini) |
| `smoke_check.py` | 08-31 | 전량 라벨링 go/no-go — `fb_check --limit 300` 결과 채점 |
| `fbcheck_variant_compare.py` | 08-21 | FB-Check 세 번째 투표자(KoELECTRA)가 검증인가 노이즈인가 |
| `fbcheck_external_eval.py` | 09-07 | FB-Check CLEAN vs NOISE를 외부 준거(FTC 근거_법령)로 채점 |
| `noise_reason_audit.py` | 08-31 | NOISE 사유별 감사 — 버려진 레코드가 정말 버릴 값이었나 |
| `flywheel_v3.py` | 09-07 | 모델이 세 번째 투표자였다면 나쁜 라벨을 골라냈을까(철회·닫힘) |

### 5. 입력 분포·파이프라인 정합성

| 파일 | 날짜 | 무엇을 재나 |
|---|---|---|
| `input_parity_eval.py` | 09-01 | 학습은 조항 원문, 운영은 evidence_span — 그 격차 측정 |
| `input_distribution_check.py` | 09-01 | 학습 텍스트와 운영 텍스트가 같은 종류의 문자열인가(라벨 불필요) |

### 6. 배포 임계값·유병률

| 파일 | 날짜 | 무엇을 재나 |
|---|---|---|
| `threshold_r_sweep.py` | 08-31 | 배포 임계값 — 위반 유병률 r을 고르지 않고 축으로 냄 |
| `prevalence_worksheet.py` | 09-01 | 유병률 r 측정용 블라인드 워크시트 생성 |
| `comparison_table_survey.py` | 08-31 | 의결서 대조표로 보정셋을 만들 수 있는지 조사(닫힘 — 재료 부재, `reported_numbers.md` §4) |

### 7. 판례 검색

| 파일 | 날짜 | 무엇을 재나 |
|---|---|---|
| `precedent_ground_truth.py` | 08-21 | 판례 `참조조문`을 FTC 근거_법령과 조인해 평가용 정답 생성 |
| `precedent_retrieval_compare.py` | 08-21 | 판례 검색 정확도 + 법원 심급 가중치 스윕 |
| `neighbor_judgment_compare.py` | 08-21 | "비슷한 사례 보여주기"가 실제로 판단에 도움이 되는가 |

---

## archive/ — 결론 난 실험

### 8. 법령 조문 검색 대안 — ⚠ 08-21 정정으로 절대 수치 전부 무효

법령 청크에 각 호가 빠져 있던 상태에서 잰 값입니다. 현재 운영은 약관규제법 제6~14조 고정이고, 남은 교훈은 "좁힐수록 좋다 / 합치면 나빠진다"뿐입니다. 요약은 `retrieval_alternatives_survey.md` 참고.

| 파일 | 날짜 | 무엇을 재나 |
|---|---|---|
| `lightrag_pilot.py` | 08-14 | LightRAG vs Hybrid RRF 파일럿(FTC 근거_법령을 정답으로) |
| `lightrag_compare.py` | 08-14 | 같은 비교를 법령 전체 3,323청크로 맞춘 본비교 |
| `lightrag_resume_indexing.py` | 08-14 | 잔액 부족으로 끊긴 인덱싱 이어 넣기 |
| `lightrag_compare_partial.py` | 08-14 | 1,500청크 인덱싱 상태로 돌린 부분 비교 |
| `lightrag_compare_final.py` | 08-14 | 전체 인덱싱 완료 후 최종 비교 |
| `lightrag_scoped_compare.py` | 08-14 | GraphRAG 서브그래프 스코핑(옵션3) |
| `rerank_compare.py` | 08-14 | 그래프 없이 Cross-Encoder 재랭킹만(옵션1) |
| `domain_filter_compare.py` | 08-14 | 법령 도메인 필터링 Hybrid(옵션2) |
| `contextual_retrieval_compare.py` | 08-14 | 청크 앞에 법령명·조문번호 붙여 임베딩(옵션4) |
| `lbox_embedding_compare.py` | 08-14 | 옵션4의 임베딩을 LBOX 법률 특화 모델로 교체 |
| `seal_rag_lite_compare.py` | 08-14 | SEAL-RAG(옵션5) vs 옵션1 |
| `combo_1_2_compare.py` | 08-14 | 옵션1+옵션2 조합 |
| `legalmalr_lite_compare.py` | 08-14 | LegalMALR식 쿼리 재구성 경량 근사 |
| `raptor_lite_compare.py` | 08-14 | RAPTOR식 법령명 1단계 라우팅(로컬 EXAONE) |
| `raptor_lite_openai_compare.py` | 08-21 | 같은 라우팅을 gpt-4o-mini로 |
| `qwen_variant_compare.py` | 08-14 | 쿼리 재구성·라우팅을 Qwen2.5-14B로 교체 |
| `combo_best_compare.py` | 08-14 | 재구성+라우팅+재랭킹 종합 콤보 |
| `stage2_recall_diagnosis.py` | 08-21 | 라우팅 이후 조문 검색 실패가 순위 문제인가 검색 문제인가 |
| `stage2_improvement_compare.py` | 08-21 | 2단계 개선안 3종 비교 |
| `law_router_compare.py` | 08-21 | LLM 라우터가 상수 기준선을 이기는가(닫힘 — `reported_numbers.md` §4) |

### 9. 판단 방식 비교 (Phase B) — KoELECTRA v4 vs 검색 기반

가장 초기 실험입니다. 검색 기반의 우위는 문서 단위 누수였던 것으로 확인됐습니다(통제 후 사실상 동점).

| 파일 | 날짜 | 무엇을 재나 |
|---|---|---|
| `retrieval_judgment.py` | 07-28 | 검색 기반 판단(KoE5 top-5 + GPT-4o-mini few-shot) — 비교 전용, 운영 경로 아님 |
| `compare_judgment.py` | 07-28 | 위 두 방식을 `data/eval/ground_truth_3class.jsonl` 1,052건에서 class별 P/R/F1 비교 |
| `hybrid_ensemble.py` | 07-28 | KoELECTRA + 검색 기반 앙상블이 단독보다 나은가 |

```bash
python -m backend.eval.archive.compare_judgment
```

```bash
python -m backend.eval.archive.compare_judgment --sample 100
```

`ground_truth_3class.jsonl`의 High/Medium 라벨은 FTC 위반유형 개수 프록시이고, 기존 883/534건 재현율 평가와는 High의 정의가 달라 직접 비교할 수 없습니다.

---

## 공용 헬퍼

| 파일 | 날짜 | 역할 |
|---|---|---|
| `common.py` | 미커밋 | FTC 근거_법령 정답셋 로더(`build_ground_truth`, `LAWS_PATH`, `N_QUERIES`) — 원래 `archive/lightrag_compare.py`에 있던 것 |
| `local_llm.py` | 08-14 | 짧은 분류·재구성 작업을 로컬 오픈모델(EXAONE 등)로 처리하는 공용 헬퍼 |

---

## 스크립트가 아닌 파일

| 파일 | 내용 |
|---|---|
| `reported_numbers.md` | **보고 수치 단일 출처.** 성능·오경보·축 상태·철회된 값 |
| `measurement_findings_2026-08-16.md` | 08-16 측정 세션 기록(판례 평가셋 구축, 법원 가중치 버그 등) |
| `retrieval_alternatives_survey.md` | 8번 섹션 검색 대안 조사 기록(수치는 08-21 정정으로 무효) |
| `prompt_block_ablation.md` | `prompt_block_compare.py`의 사전 등록 판정 기준과 결과 |
| `tos_targets.tsv` | `real_tos_eval.py` 입력 — 측정할 실제 이용약관 목록 |
