# data/

파이프라인 각 단계의 산출물. `.gitignore`에 등록되어 있어 저장소에는 커밋되지 않는다(민감·대용량 데이터). 스냅샷 기준 수치이며 크롤링/재학습을 거치며 계속 바뀐다 — 정확한 현재 값은 각 파일을 직접 확인할 것.

## 파이프라인 단계별 흐름

```
raw/ → domain/ → processed/chunks(laws/precedents/interpretations) ┐
                                                                      ├→ fb_check/ → models/
labels/(seed_labeled.jsonl, FTC 시정조치 Seed) ─────────────────────┘
```

## 디렉토리

| 경로 | 내용 | 생성 스크립트 |
|---|---|---|
| `raw/law/` | 국가법령정보 API 원천 법령 | `backend/scripts/crawl_law_api.py` |
| `raw/case/` | 국가법령정보 API 원천 판례 (정상+빈응답 파일 혼재 — 빈 응답도 "미존재 확인 기록"이므로 삭제 금지) | 〃 |
| `raw/commentary/` | 국가법령정보 API 원천 해석례 | 〃 |
| `raw/contract/` | 공정위 표준계약서(HWP) 원문·추출 텍스트, Low/Medium 라벨 소스 | `backend/scripts/crawl_standard_contract.py` |
| `raw/ftc_cases/` | 공정위 시정조치 심결례 목록 JSON + PDF(`pdfs/`, 3,290건) + 파싱 결과(`ftc_cases_parsed.json`) | `backend/scripts/crawl_ftc_cases.py`, `backend/scripts/parse_ftc_case_pdf.py` |
| `domain/{law,case,commentary}/` | 도메인(해지·책임제한) 필터링 통과 문서만 별도 보관 | `backend/domain/filter_*.py` |
| `processed/chunks/`(→ 실제 파일은 `processed/*.jsonl`) | 청킹·정제 결과: `laws.jsonl`, `precedents.jsonl`, `interpretations.jsonl` | `backend/preprocess/` |
| `labels/` | `seed_labeled.jsonl` — FTC 시정조치(High) + 표준계약서(Low/Medium) Seed 라벨, Ground Truth로 사용 | `backend/labeling/seed.py` |
| `fb_check/` | FB-Check(Forward-Backward Consistency Check) 결과: `clean.jsonl`(CLEAN), `noise.jsonl`(NOISE), `fb_check_results.jsonl`(전체), `rescued_domain_none.jsonl`(domain_none 구제 데이터) | `backend/fb_check/` |
| `logs/` | 각 단계 실행 로그 | 공통 (`load_logger`/`setup_logger`) |

`fb_check/`, `data/labels/`, `data/raw/ftc_cases/`에는 과거 파싱 버그 수정 전후를 비교하기 위한 `_backup_before_*` 백업 폴더가 각 파일 안에 보관되어 있다(폴더별로 독립 백업 — 전체를 한 곳에 모으지 않음, 자세한 배경은 [models/README.md](../models/README.md) 버전 히스토리 참고).

## 데이터 주의사항 (Claude.md 발췌)

- `raw/case/`의 `{"Law": "일치하는 판례가 없습니다..."}` 파일은 삭제 금지 — API 미존재 ID 확인 기록, 삭제 시 무한 재호출 발생.
- FB-Check CLEAN 판정은 원 논문의 2-way(forward==verify) 대신 3-way 다수결(forward/verify/backward-KoELECTRA 중 2개 이상 일치)로 확장 운영 중 — 방법론적 편차, 논문에 명시 필요([models/README.md](../models/README.md) 참고).
