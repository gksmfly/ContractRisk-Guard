# backend/training/

Step 5/8: KoELECTRA 파인튜닝. `backend/model/electra.py`의 `DualHeadElectra`(domain + risk_level 동시 분류)를 학습시켜 `models/v*/`에 저장합니다.

---

## 파일 목록

### `train.py`

**데이터 소스** (`--data-source`)
| 값 | 대상 | 특징 |
|---|---|---|
| `seed` | `data/labels/seed_labeled.jsonl` | FB-Check 검증 이전 원본. 텍스트가 FTC 판례 발췌문 전체(평균 600자+, 최대 2만자)라 실서비스 입력(100~150자)과 길이 분포가 다르고, 길이와 risk_level이 상관돼 있어 모델이 "내용"이 아니라 "길이"로 지름길 학습할 위험 |
| `clean` | `data/fb_check/clean.jsonl` | FB-Check CLEAN. `evidence_span`(평균 49자, Consistency Verification으로 "이 근거만으로도 같은 라벨이 재현됨"이 검증된 짧은 인용구)을 학습 텍스트로 사용 |

- `load_records()`: `clean` 소스일 때 `evidence_span`이 없는 레코드는 제외, `forward_domain`이 `DOMAIN_MAP`에 없는 값(GPT가 드물게 도메인 문자열을 깨뜨려 반환, 예: "해 책임제한_조항")이면 제외. `risk_level`은 `final_label`(forward/verify/backward 다수결) 우선, 없으면 `forward_label` 폴백(옛 clean.jsonl 호환)
- `--no-fulltext-augment`가 없으면(기본) evidence_span과 별도로 원문 전체도 같은 라벨로 학습 예시에 추가 — evidence_span만 학습하면 원문이 그대로 들어오는 실서비스 폴백 케이스에서 정확도가 폭락하기 때문(ground truth 평가: evidence_span 69.6% vs 원문 대체 23.0%)
- 학습: `DualHeadElectra` + 클래스 불균형 보정(`compute_class_weight`) + `train_test_split`(stratify=risk_level) + 매 epoch 검증, `risk_macro_f1` 최고 갱신 시에만 체크포인트 저장
- 출력: `models/{v1,v2,...}/`(가중치+토크나이저+`heads.pt`) + `metrics.json`(epoch별 domain/risk 정밀도·재현율·F1)

**실행**:
```bash
python -m backend.training.train --data-source seed                                    # 1차(v1)
python -m backend.training.train --data-source clean --epochs 10 --batch-size 32 --gpu 1  # 2차 이후(v2+)
python -m backend.training.train --data-source clean --seed 123 --no-fulltext-augment    # 버전 비교 실험용
```
기본 저장 경로는 `seed`→`models/v1`, `clean`→`models/v2`, `--model-dir`로 임의 지정 가능(v3 이후 전부 이렇게 지정해서 씀).

### `train_article.py` (현행 — 조 multi-label)

운영 모델 `models/article_v2`를 만든 학습 스크립트. 규칙은 파일 머리 docstring에 있다(held-out gold 미사용, 문서 단위 분할, 상수 기준선 병기, 체크포인트는 dev로 선택).

- `--negative-holdout-docs FILE`: 음성 홀드아웃을 시드 추첨 대신 **고정 문서 목록**으로 뗀다. 데이터가 늘면 추첨 결과가 바뀌어 이전 세대가 학습한 문서가 평가셋이 되기 때문
- `metrics.json`에 `labels_sha256`(학습 데이터 지문), `negative_holdout_doc_ids`, `train_config.seed`를 남긴다(2026-09-28부터. `article_v2`에는 없다)

### `retrain.py` — 새 데이터가 들어왔을 때

데이터만 바꿔 재학습하고, 사전 등록 규칙으로 무해 확인한 뒤, 이력을 남긴다. **서빙 전환은 하지 않는다.**

```
① 스냅숏 → ② 변경 확인 → ③ 라벨러 확인(다르면 멈춤) → ④ 분포 이동 경고
→ ⑤ 현행 설정 그대로 학습 → ⑥ 무해 확인 → ⑦ 이력 기록 → ⑧ (--promote-as) 승격
```

| 판정 | 기준 | 기각 조건 |
|---|---|---|
| 주 ① | gold clean 255 조항 단위 재현 | 후보−현행 페어드 CI 상한 < 0 |
| 주 ② | 표준계약서 홀드아웃 12문서 오경보 | 후보−현행 문서 부트스트랩 CI 하한 > 0 |
| 기록 | 조 F1 차이, 후보의 상수 대비, 첫 기록 대비 누적 이동 | 판정에 안 씀 |

- **통과는 "나빠지지 않았다"이지 개선의 증거가 아니다.** gold 255로는 조 F1 +3%p도 가릴 수 없다
- 홀드아웃 문서는 [`negative_holdout_docs.json`](negative_holdout_docs.json)에 고정돼 있다
- 이력: `data/eval/retrain_history.jsonl`(환경변수 `RETRAIN_HISTORY`). 세대별 고정 평가셋 지표가 쌓여 누적 하락을 본다
- 모델 붕괴 모니터링을 따로 두지 않은 이유와 대신 막는 위험은 `retrain.py` docstring 참고

```bash
python -m backend.training.retrain --check-only                          # 데이터 확인만(GPU 불필요)
python -m backend.training.retrain --gpu 1                               # 학습·판정·기록
python -m backend.training.retrain --gpu 1 --promote-as models/article_v3
MODEL_DIR=models/article_v3 uvicorn backend.api.server:app               # 서빙 전환은 사람이
```

---

## 참고

- 버전별(v1~v9) 성능 비교, 시드 변동성 실험, 검색 기반 대안과의 비교는 전부 `models/README.md` 참고 — 이 디렉토리엔 학습 스크립트만 있고 실험 결과·의사결정 기록은 없음
- 재현성을 위해 `--seed`로 Python/NumPy/PyTorch 난수를 전부 고정하지만, 데이터가 작을수록(특히 High 클래스 72~186건) 시드 하나 바꾸는 것만으로 최종 정확도가 20%p 이상 흔들릴 수 있다는 게 실험으로 확인됨 — 단일 시드 학습 결과로 파이프라인 버전을 비교하지 말 것
