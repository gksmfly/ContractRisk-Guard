# backend/api/

FastAPI 서빙 레이어 — 계약서 텍스트/PDF를 받아 고정 6단계 파이프라인(`backend.agents`)을 실행하고 **조항별 "확인 필요" 판단 + 관련 약관규제법 조**를 반환합니다.

> **위험도 등급을 반환하지 않습니다** (2026-08-31). 조 multi-label 모델에 risk 헤드가 없고, 옛 `confidence_band` 실측치는 `models/v4` 전용이라 옮길 수 없습니다.

---

## 파일 목록

### `server.py`
앱 진입점. CORS 허용 origin은 `CORS_ORIGIN` 환경변수(기본 `http://localhost:3000`, 프론트엔드 dev 서버). 실행: `uvicorn backend.api.server:app --reload`.

### `schemas.py`
요청/응답 Pydantic 스키마.
- `AnalyzeRequest`: `text`(계약서 원문) 하나뿐.
- `ClauseResult`: 조항 하나의 분석 결과 — `articles`(모델이 지목한 조, **참고값**), `needs_review`(이진), `evidence_spans`, `legal_basis`(예측 조에서 매핑한 조문), `precedent_refs`(유사 판례, **참고**), `reasoning`, `verified`, `redteam_note`, `evidence_verified`. `domain`은 옛 저장분 호환용 빈 문자열.
- `OutOfScopeClause`: 모델이 조를 지목하지 않은 조항 — **어떤 등급도 붙이지 않습니다.** "확인되지 않았다"이지 "안전하다"가 아닙니다(조 단위 재현 78%이므로 약 5건 중 1건은 여기 잘못 들어와 있습니다).
- `AnalyzeResponse`: `total_clauses` + `review_count` + `clauses` + `input_clauses`/`truncated_clauses`/`out_of_scope` + `model_version`. `high_count`/`medium_count`/`low_count`는 옛 저장분 호환용으로 남아 있고 **새 응답에서는 항상 0**입니다.

### `routers/analyze.py`
- `GET /health`: DB까지 확인하는 readiness 체크(`SELECT 1` 실패 시 503).
- `POST /api/analyze`: 원문 텍스트 직접 분석. 20자 미만이면 400.
- `POST /api/analyze/stream`: 같은 분석을 SSE로 스트리밍(조항이 끝날 때마다 진행률).
- `POST /api/analyze-pdf` / `POST /api/analyze-pdf/stream`: PDF 업로드(`pypdf` 추출, 10MB 상한) 후 동일 경로 재사용.

전부 `require_api_key` + `enforce_rate_limit` 의존성이 걸립니다. 스트리밍 경로는 **검증·세마포어 획득을 첫 바이트 전에** 끝냅니다 — `StreamingResponse`가 시작되면 상태 코드를 바꿀 수 없습니다.

### `services/analyze.py`
핵심 로직 `run_analyze()` / `run_analyze_stream()`:
1. `split_clauses()` — 정규식(`제N조`, 번호 목록, 원 문자 ①②③, 연속 줄바꿈)으로 조항 분리, 최대 `MAX_CLAUSES`(기본 60). 넘긴 개수는 `truncated_clauses`로 **응답에 반드시 남깁니다** — 예전 상한 20이 실제 계약서(30조항대)를 조용히 잘라냈습니다.
2. 조항을 **병렬로** 그래프에 던집니다(`asyncio.gather`). 동시 호출은 모듈 전역 세마포어 두 개가 묶습니다 — 요청별로 만들면 `요청 4 × 조항 30 = 120`으로 곱해져 OpenAI rate limit에 그대로 부딪힙니다.
3. 모델이 조를 하나도 지목하지 않은 조항은 **버리지 않고** `OutOfScopeClause`로 목록에 남깁니다 — 예전에는 `None`을 반환하고 호출부가 버려서 조항이 응답에서 통째로 사라졌습니다(입력 20 → 결과 10건).
4. `_extract_spans()`로 evidence_span의 원문 내 위치(start/end) 계산 — 완전일치 → 공백 정규화 → 퍼지(0.85). FB-Check와 **같은 기준**을 씁니다(검증이 통과시킨 근거를 서빙이 버리면 하이라이트가 조용히 사라집니다).

### `services/retrieval.py`
법령·판례 Hybrid(Dense+Sparse) 검색 — `backend.agents`의 Retrieval Strategy/Evidence Selection/Red-team이 공통으로 재사용하는 DB 서비스 레이어. 상세는 코드 상단 docstring 참고(RRF 융합, KoE5 임베딩, pg_trgm Sparse, 법원 심급 가중치 등 설계 근거가 정리돼 있음).

### `services/latency_benchmark.py`
`run_analyze()` 처리 속도 측정 — 조항 수(1/5/10/20)별로 총 소요·조항당 평균 시간을 잰다.
```bash
python -m backend.api.services.latency_benchmark
python -m backend.api.services.latency_benchmark --sizes 1,5,10,20
```
실측 결과: 조항당 약 8.6~8.8초 — 병목은 판단(Judgment, GPU 추론이라 빠름)이 아니라 근거 브랜치의 재검색 루프(KoE5 임베딩+DB 쿼리가 재시도마다 순차로 쌓임)로 확인됨.

---

## 실행

```bash
uvicorn backend.api.server:app --reload --port 8000
```

환경변수: `OPENAI_API_KEY`, `DATABASE_URL`, `FORWARD_MODEL`, `CORS_ORIGIN`, `MODEL_DIR`(KoELECTRA 체크포인트 경로, 기본 `models/article_v2`. 읽는 곳은 `backend/agents/judgment_agent.py`).
