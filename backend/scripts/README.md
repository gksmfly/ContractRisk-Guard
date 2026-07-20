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

### `utils.py`
**공통 유틸리티**

모든 스크립트에서 공유하는 함수 모음입니다. `load_dotenv()`도 이 모듈에서 한 번만 호출됩니다.

- `save_json(data, filepath)`: JSON 직렬화 후 파일 저장 (부모 디렉터리 자동 생성)
- `setup_logger(log_filename)`: 콘솔 + 파일 동시 출력하는 named logger 반환
- `PROJECT_ROOT`: 프로젝트 루트 Path (`backend/scripts/utils.py` 기준 2단계 상위)

---

## 참고

이 디렉터리는 2026-06-21 히스토리 정리 과정에서 잠시 최상위 `scripts/`로 옮겨졌다가, 실제로 `backend/` 밖에 둘 architectural한 이유가 없어 다시 `backend/scripts/`로 복원했습니다.
