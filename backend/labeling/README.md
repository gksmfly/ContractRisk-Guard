# backend/labeling/

Step 3: Seed 라벨 생성 — FB-Check(GPT Forward/KoELECTRA Backward)가 정제할 원본 학습 데이터를 만듭니다. 전문가 없이 두 소스에서 자동으로 라벨을 부여합니다.

---

## 파일 목록

### `seed.py`
두 소스를 처리해 `data/labels/seed_labeled.jsonl`(2,219건) + `data/labels/seed_label_report.json`을 만든다.

**FTC 시정조치** (`extract_ftc_records()`)
- `data/raw/ftc_cases/ftc_cases_parsed.json`의 `조항_원문`에서 조항 추출
- `risk_level`은 케이스 자체의 `risk_level`(기본 `"High"` 하드코딩 — 공정위가 확정한 위반이므로 별도 판정 없이 High) 그대로 사용
- **알려진 한계**: 이의제기로 뒤집힌 `재결기각`(4건)이나 위반 심각도가 제각각인 `시정권고`/`시정명령`을 구분 없이 전부 High로 취급 — `backend/scripts/rebuild_ftc_ground_truth.py`가 평가용으로 이 한계를 별도로 보정함(학습 데이터인 이 파일 자체는 아직 안 고침)

**표준계약서** (`extract_contract_records()`)
- `data/raw/contract/contracts_표준*.json`의 `추출_텍스트`를 `제N조(제목)` 패턴(`split_articles()`)으로 조문 분리
- `classify_domain()`: 해지/책임제한 키워드 등장 횟수로 도메인 판정(둘 다 0이면 제외)
- `assess_risk()`: 도메인별 정규식 패턴으로 risk_level 판정
  - High: "즉시 해지", "사전 통보 없이", "일방적", "어떠한 경우에도 해지" 등 / 책임제한은 "모든 손해에 대하여 책임을 지지 않", "완전 면책" 등
  - Medium: "7일 이내", "정당한 이유 없이" 등 / 책임제한은 "간접 손해", "배상액 상한" 등
  - 둘 다 안 걸리면 Low

**실행**:
```bash
python -m backend.labeling.seed
```

---

## 참고

`split_articles()`(조문 분리 정규식)와 `classify_domain()`은 `backend/scripts/rebuild_ftc_ground_truth.py`(평가용 ground-truth 재구축)에서도 재사용된다 — 원래 `_split_articles`처럼 비공개(밑줄 접두) 함수였으나 이 재사용을 위해 공개 함수로 이름을 바꿨다.

`assess_risk()`의 정규식 판정은 표준계약서 Low/Medium/High 라벨의 유일한 근거라 — 이 라벨을 그대로 "정답"으로 평가에 쓰면 순환논리가 된다(모델이 이 정규식보다 똑똑하게 판단해도 "틀렸다"고 채점됨). 평가용 ground truth는 이 파일이 아니라 `data/eval/ground_truth_3class.jsonl`(외부 권위 소스 기반)을 쓴다.
