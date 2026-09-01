# backend/eval/

KoELECTRA(v4) vs 검색 기반(KoE5+GPT-4o-mini few-shot) 판단 방식 비교 평가. Ground-truth 데이터셋 구축은 `backend/scripts/rebuild_ftc_ground_truth.py`·`extract_precedent_ground_truth.py`(수집 성격이라 `scripts/`에 위치)가 담당하고, 이 디렉토리는 그 데이터셋으로 두 판단 방식을 실제로 돌려 비교하는 로직만 담당합니다.

---

## 배경

기존 883/534건 평가(`models/README.md`)는 FTC 시정조치 High 케이스의 **재현율만** 측정했다 — Low/Medium 정답 데이터가 없었기 때문이다. `data/eval/ground_truth_3class.jsonl`(Phase A, `backend/scripts/rebuild_ftc_ground_truth.py` 참고)이 처음으로 3-class(High/Medium/Low) 정답셋을 만들었고, 이 디렉토리가 그 위에서 두 판단 메커니즘의 class별 precision/recall/F1을 비교한다.

---

## 파일 목록

### `retrieval_judgment.py`
검색 기반 판단 — `backend.api.services.retrieval.search_similar_labeled_clauses()`로 `clean_clauses`에서 KoE5 top-5 유사 사례를 찾고, GPT-4o-mini(`RETRIEVAL_JUDGE_MODEL`, 기본 `gpt-4o-mini`)에게 few-shot으로 제공해 risk_level만 판단시킨다. `models/README.md`에 설명만 있고 코드가 없던 실험을 재현 가능한 형태로 옮긴 것 — **프로덕션 판단 경로(`backend.agents.judgment_agent`)는 여전히 KoELECTRA**, 이 모듈은 비교 평가 전용.

### `compare_judgment.py`
메인 비교 스크립트.
1. **evidence_span 추출**(KoELECTRA용): `backend.fb_check.forward_labeling.run_forward()`로 원문에서 evidence_span만 뽑는다(domain·risk_level 출력은 버림 — 비교 대상 오염 방지). `data/eval/evidence_span_cache.jsonl`에 한 건씩 즉시 append — 1,052건 중간에 API 오류로 끊겨도 재실행 시 이어서 진행
2. **KoELECTRA 평가**: evidence_span을 `backend.agents.judgment_agent.predict_articles()`에 입력(프로덕션과 동일 입력 분포)
3. **검색기반 평가**: 원문 그대로 `retrieval_judge()`에 입력 — 기존 실험에서 검색기반은 원문이 evidence_span보다 오히려 살짝 나았기 때문에, 각 방식이 실제로 쓰는 입력 형태로 공정하게 비교
4. `sklearn.metrics`로 class별 precision/recall/F1 + confusion matrix 산출 → `data/eval/compare_judgment_report.json`

```bash
python -m backend.eval.compare_judgment              # 전체 1,052건
python -m backend.eval.compare_judgment --sample 100  # 비용/시간 가늠용 샘플
```

**비용**: GPT-4o-mini 호출만 과금 대상(KoELECTRA·KoE5 임베딩은 로컬). 1,052건 기준 evidence_span 추출 + 검색기반 판단 합쳐 약 $0.3~0.5.

---

## 결과 해석 시 주의

- `data/eval/ground_truth_3class.jsonl`의 High/Medium 라벨은 FTC 위반유형 개수 프록시(외부 권위 기관이 직접 부여한 등급이 아님) — 상세 한계는 `data/eval/candidates/rebuild_report.json`의 `note` 필드와 `backend/scripts/README.md` 참고
- 이 평가의 숫자는 기존 883/534건 재현율 평가와 **정의가 달라 직접 비교 불가**(High의 의미 자체가 다름) — 별도의 새 벤치마크로 취급할 것
