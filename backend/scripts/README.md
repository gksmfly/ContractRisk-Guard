# backend/scripts/

계약 리스크 분석에 필요한 원천 데이터를 수집·파싱하는 스크립트 모음입니다.

---

## 파일 목록

### `crawl_law_api.py`
**국가법령정보 API — 법령·판례·법령해석례 수집**

`--source` 인자로 수집 대상을 선택하는 통합 스크립트입니다.

| `--source` | 대상 | 출력 디렉터리 | 목록 파일 |
|---|---|---|---|
| `law` | 법령 (조문) | `data/raw/law/` | `mst_list.json` |
| `precedent` | 판례 (판시사항·판결요지) | `data/raw/case/` | `prec_list.json` |
| `interpretation` | 법령해석례 (회신내용) | `data/raw/commentary/` | `expc_list.json` |

- **방식**: 목록 API 전 페이지 순회 → 항목별 상세 API 호출 → 필요 필드만 추출하여 개별 JSON 저장
  - 이미 저장된 파일은 건너뜀
- **실행**:
  ```bash
  python backend/scripts/crawl_law_api.py --source law --key <API인증키>
  python backend/scripts/crawl_law_api.py --source precedent --key <API인증키>
  python backend/scripts/crawl_law_api.py --source interpretation --key <API인증키>
  python backend/scripts/crawl_law_api.py --source law --key <API인증키> --query 계약
  python backend/scripts/crawl_law_api.py --source law --key <API인증키> --delay 2.0
  ```

---

### `crawl_ftc_cases.py`
**공정거래위원회 불공정약관 시정조치 사례 목록 수집 (Seed 데이터)**

- **대상**: 공정거래위원회 심결례 중 사건번호에 "약관"이 포함된 케이스 전체 (`reprsntViolTy=10`이 실제로 필터링되지 않아 `caseNo=약관`으로 대체)
- **방식**: Playwright로 전체 페이지 순회 → 사건 메타데이터 + PDF 식별자(docId, docSn) 수집
- **출력**: `data/raw/ftc_cases/ftc_cases_raw.json`
- **실행**:
  ```bash
  python backend/scripts/crawl_ftc_cases.py
  python backend/scripts/crawl_ftc_cases.py --no-headless
  python backend/scripts/crawl_ftc_cases.py --delay 2.0 --max-pages 100
  ```
- **참고**: `parse_ftc_case_pdf.py` 실행 전 반드시 먼저 실행해야 합니다.

---

### `parse_ftc_case_pdf.py`
**공정거래위원회 시정조치 PDF 다운로드 및 텍스트 파싱**

- **사전 조건**: `crawl_ftc_cases.py` 실행 후 `data/raw/ftc_cases/ftc_cases_raw.json` 존재
- **방식**: pdf_info(docId, docSn)로 Playwright PDF 다운로드 → pdfplumber 텍스트 추출
  - 조항 원문 / 위반 유형 / 근거 법령 / 시정 내용 / risk_level 구조화 (High = 공정위 시정조치 대상, 위반 확정)
- **출력**: `data/raw/ftc_cases/ftc_cases_parsed.json`
- **실행**:
  ```bash
  python backend/scripts/parse_ftc_case_pdf.py
  python backend/scripts/parse_ftc_case_pdf.py --skip-download   # 기존 PDF로 파싱만
  python backend/scripts/parse_ftc_case_pdf.py --no-headless
  python backend/scripts/parse_ftc_case_pdf.py --delay 2.0
  ```

#### 알려진 버그와 수정 (2026-07-13)

`extract_clause_text()`의 조항 추출 정규식이 "제N조(제목)"이면 무조건 매칭해서, 피심인의
실제 약관뿐 아니라 **약관규제법 조문 자체(예: 제7조 면책조항의 금지)나 심결문의 증거
인용·법조문 설명까지 조항 원문으로 잘못 추출**하고 있었다. 게다가 다음 "제N조"가 나올
때까지 무한정 이어붙여서 7,000자 넘는 청크도 생겼다.

이로 인해 FB-Check Forward Labeling(GPT)이 이런 오염된 텍스트를 받으면 "이게 실제
계약 조항인지" 판단 자체를 포기하는 경우가 많았다 — FTC 시정조치 확정 사례 499건 중
195건(39%)이 evidence_span도 못 뽑고 "해당없음" 처리됐는데, 원인을 추적해보니 상당수가
이 파싱 오염 때문이었다 (`backend/fb_check/README.md`의 정량 평가 섹션 참고).

수정: ① 조항 뒤 600자로 컷, ② "이러한 사실은", "적용 법조", "정하고 있는 조항은" 같은
증거·법조문 인용 경계에서 잘라냄, ③ 헤더가 약관규제법 제6~17조 실제 조문명과 정확히
일치하면(정규화 비교) 통째로 제외. 60건 재검증 결과 추출 실패 0건, 평균 조항 길이
703자 → 410자로 감소.

**주의**: 이 수정은 `parse_ftc_case_pdf.py`에만 반영됐고, 기존에 이미 생성된
`data/raw/ftc_cases/ftc_cases_parsed.json` → `seed_labeled.jsonl` → FB-Check 결과에는
아직 반영 안 됐다 (PDF 재파싱부터 FB-Check 재실행까지 전체 파이프라인을 다시 돌려야
반영됨 — 시간이 걸려서 보류 중).

---

### `crawl_standard_contract.py`
**공정거래위원회 표준계약서 HWP 다운로드 및 텍스트 추출**

- **대상**: 공정거래위원회 표준계약서 게시판 (6개 카테고리)
  - 표준약관 / 표준하도급계약서 / 표준가맹계약서
  - 표준유통거래계약서 / 표준대리점거래계약서 / 표준비밀유지계약서
- **방식**: Playwright로 게시판 순회 → HWP 파일 다운로드 → olefile + zlib 텍스트 추출
- **출력**: `data/raw/contract/contracts_parsed.json` (카테고리별·전체 통합)
- **실행**:
  ```bash
  python backend/scripts/crawl_standard_contract.py
  python backend/scripts/crawl_standard_contract.py --skip-download      # 기존 HWP로 파싱만
  python backend/scripts/crawl_standard_contract.py --category 표준약관  # 특정 카테고리만
  python backend/scripts/crawl_standard_contract.py --no-headless       # 브라우저 화면 표시
  python backend/scripts/crawl_standard_contract.py --delay 2.0 --max-pages 30
  ```

---

### `rebuild_ftc_ground_truth.py`
**Low/Medium 정답 데이터가 없는 문제를 수작업 라벨링 없이 해결하는 evaluation ground-truth 재구축**

- **배경**: 기존 883/534건 평가는 FTC High 케이스 재현율만 측정 — Low/Medium 정답이 없었음. `seed_labeled.jsonl`의 Medium/Low는 정규식(`seed.py`의 `assess_risk()`)이라 그대로 정답으로 쓰면 순환논리(모델이 정규식보다 똑똑해도 "틀렸다"고 채점됨).
- **1차 시도(기각)**: FTC `대표조치유형`(시정명령→High, 시정권고→Medium) 가설 — 스팟체크 결과 시정권고 쪽이 오히려 위반유형·근거법령 수가 더 많아(평균 2.2 vs 1.0) 기각.
- **채택**: 케이스별 `위반_유형` 개수(2개+ → High, 1개 → Medium)를 심각도 프록시로 사용. FTC가 공식 부여한 등급이 아니라 이 프로젝트가 만든 추정치임에 유의(스팟체크 중 반례 1건 확인됨: "사전통지 없이 해지" 표현이 있는데도 위반유형 1개라 Medium 분류).
- **Low**: `data/raw/contract/`(공정위 공식 표준계약서 6종)에서 조문 패턴으로 시작하는(=크롤링 노이즈 아닌) 문서만 추출 — "정부가 공식 발행한 공정 계약 템플릿"이라는 외부 권위를 근거로 삼음(정규식 위험도 판정 안 씀)
- **재결기각**(이의제기로 뒤집힌 4건) 등 미확정 케이스는 제외
- `data/fb_check/clean.jsonl`(학습 데이터)과 겹치는 `chunk_id`는 제외해 평가 누출 방지
- `extract_precedent_ground_truth.py`(아래) 결과와 병합해 최종 산출
- **출력**: `data/eval/ground_truth_3class.jsonl`(High 460 / Medium 98 / Low 485 + 판례 9건 = 총 1,052건), `data/eval/candidates/rebuild_report.json`
- **실행**:
  ```bash
  python -m backend.scripts.rebuild_ftc_ground_truth
  ```

---

### `extract_precedent_ground_truth.py`
**법원 판례의 무효/유효 판결 — 제3의(사법부) ground-truth 보강**

- **배경**: FTC(행정부, 시정조치·표준계약서)만으로는 Medium 신호가 약함. 완전히 독립된 권위 있는 기관(법원)의 판결을 추가로 활용.
- **방식**: `data/domain/case/`(1,995건) 중 약관규제법을 직접 다루고, 해지·책임제한 도메인 키워드가 있고, "무효" 여부가 쟁점인 판례 23건을 사람이 직접 읽고 판별 — 자동 정규식 판별을 시도했으나(무효/유효 부정 표현 탐지) 23건 중 1건만 잡혀서, 이 표본 크기에서는 일반화된 파서보다 수작업이 낫다고 판단
- **수작업 기준**: 파기환송(최종 확정 아님)이나 한 판례에 조항이 여러 개라 결론이 갈리는 케이스는 제외. 무효 확정 → Medium(개별 사건 판단이라 FTC 확정 위반만큼 광범위하지 않다고 보수적으로 잡음, 연 60% 연체료처럼 심각해 보이는 케이스도 예외 없이 적용), 유효 확정(무효 아님) → Low
- **재현성**: 수작업 판별 결과를 `_CURATED_VERDICTS` 딕셔너리(case_id → risk_level, 조항 텍스트, 비고)에 하드코딩 — 코드만 재실행해도 같은 결과가 나옴
- **출력**: `data/eval/candidates/precedent_candidates.jsonl` (Low 3 / Medium 6, 총 9건)
- **실행**:
  ```bash
  python -m backend.scripts.extract_precedent_ground_truth
  ```
- **참고**: `rebuild_ftc_ground_truth.py`를 나중에 실행하면 이 출력 파일을 자동으로 읽어 최종 ground truth에 합침 — 실행 순서는 이 스크립트 먼저.

---

### `utils.py`
**공통 유틸리티**

모든 스크립트에서 공유하는 함수 모음입니다. `load_dotenv()`도 이 모듈에서 한 번만 호출됩니다.

- `save_json(data, filepath)`: JSON 직렬화 후 파일 저장 (부모 디렉터리 자동 생성)
- `setup_logger(log_filename)`: 콘솔 + 파일 동시 출력하는 named logger 반환
- `PROJECT_ROOT`: 프로젝트 루트 Path (`backend/scripts/utils.py` 기준 2단계 상위)

---

## 참고

이 디렉터리는 2026-06-21 히스토리 정리 과정에서 잠시 최상위 `scripts/`로 옮겨졌다가, 실제로 `backend/` 밖에 둘 architectural한 이유가 없어 다시 `backend/scripts/`로 복원했습니다.
