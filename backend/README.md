# backend/

파이프라인(데이터 수집 → 도메인 필터링 → 전처리 → Seed 라벨링 → FB-Check → 모델 학습)과 FastAPI 서빙 코드. 전체 아키텍처·코드 규칙은 [../Claude.md](../Claude.md) 참고.

## 디렉토리

| 경로 | 역할 |
|---|---|
| `scripts/` | 원천 데이터 수집: 법령/판례/해석례 API(`crawl_law_api.py`), 공정위 시정조치 목록·PDF(`crawl_ftc_cases.py`, `parse_ftc_case_pdf.py`), 표준계약서 HWP(`crawl_standard_contract.py`) |
| `domain/` | Step 1: 도메인(해지·책임제한) 필터링 — `filter_law.py`(메타데이터), `filter_precedent.py`(키워드∪Dense), `filter_interpretation.py`(Dense 단독), 설정은 `config.py` |
| `preprocess/` | Step 2: 청킹·정제 — `extractor.py`, `cleaner.py` → `data/processed/*.jsonl` |
| `labeling/` | Step 3: `seed.py` — FTC 시정조치(High) + 표준계약서(Low/Medium) Seed 라벨링, `classify_domain()` 등 |
| `fb_check/` | Step 6: Forward-Backward Consistency Check — `forward_labeling.py`(GPT Forward), `backward_grounding.py`(근거 추출+KoELECTRA), `consistency_verification.py`(재라벨링). `llm_benchmark.py`/`oss_experiment/`는 오픈소스 LLM 대체 가능성 실험(프로덕션 경로 아님) |
| `model/` | `electra.py` — `DualHeadElectra`(domain + risk_level 동시 분류), `DOMAIN_MAP`/`RISK_MAP` |
| `training/` | Step 5/8: `train.py` — KoELECTRA 파인튜닝 CLI (`--gpu`, `--epochs`, `--seed`, `--no-fulltext-augment` 등) |
| `db/` | `loader.py` — PostgreSQL+pgvector 적재(`chunks`/`seed_clauses`/`clean_clauses`, OpenAI `text-embedding-3-large`). `embedding_benchmark.py`는 임베딩 모델 비교 실험 |
| `api/` | FastAPI 서빙 — `server.py`(앱 진입점), `routers/analyze.py`(엔드포인트), `schemas.py`(요청/응답 스키마), `services/analyze.py`(`run_analyze`, 조항 분리+판단 파이프라인 핵심 로직), `services/retrieval.py`(근거 법령·판례 Dense Retrieval), `services/latency_benchmark.py`(처리 속도 측정) |
| `utils.py` | 공통 유틸 — `save_json`/`save_jsonl`/`load_jsonl`/`load_logger`/`PROJECT_ROOT` |

`scripts/utils.py`는 크롤 스크립트 전용 별도 유틸(`save_json`/`setup_logger`/`PROJECT_ROOT`)로, 위 `backend/utils.py`와 의도적으로 분리되어 있다(크롤러는 `backend/` 나머지 모듈에 의존하지 않고 독립 실행 가능해야 함).

## 파이프라인 실행 순서

```
scripts/ → domain/ → preprocess/ → labeling/seed.py → db/loader.py
                                                       → fb_check/ (forward → backward → verify)
                                                       → training/train.py → models/v*/
api/ (server.py)는 models/v4를 로드해 서빙 — 현재 프로덕션 버전 및 판단 근거는 ../models/README.md 참고
```

## 실행 환경

```bash
source .venv/bin/activate
```
모든 명령은 저장소 루트에서 실행. GPU는 `cuda:1` 고정. FastAPI 서버: `uvicorn backend.api.server:app --reload`.

## 테스트

`api/services/analyze.py`에 대한 pytest는 [../tests/README.md](../tests/README.md) 참고.
