# backend/api/

FastAPI 서빙 레이어 — 계약서 텍스트/PDF를 받아 6-agent 파이프라인(`backend.agents`)을 실행하고 조항별 위험도 분석 결과를 반환합니다.

---

## 파일 목록

### `server.py`
앱 진입점. CORS 허용 origin은 `CORS_ORIGIN` 환경변수(기본 `http://localhost:3000`, 프론트엔드 dev 서버). 실행: `uvicorn backend.api.server:app --reload`.

### `schemas.py`
요청/응답 Pydantic 스키마.
- `AnalyzeRequest`: `text`(계약서 원문) 하나뿐.
- `ClauseResult`: 조항 하나의 분석 결과 — `domain`, `risk_level`, `confidence`(verified 여부에 따라 1.0/0.7), `evidence_spans`, `legal_basis`, `reasoning`, `redteam_note`, `evidence_verified`.
- `AnalyzeResponse`: `total_clauses` + High/Medium/Low 카운트 + `clauses` 목록.

### `routers/analyze.py`
- `GET /health`: 헬스체크.
- `POST /api/analyze`: 원문 텍스트 직접 분석. 20자 미만이면 400.
- `POST /api/analyze-pdf`: PDF 업로드(`pypdf`로 텍스트 추출) 후 동일 분석 경로 재사용.

### `services/analyze.py`
핵심 로직 `run_analyze()`:
1. `split_clauses()` — 정규식(`제N조`, 번호 목록, 원 문자 ①②③, 연속 줄바꿈)으로 조항 분리, 최대 20개
2. 조항마다 `backend.agents.graph.get_graph()`로 6-agent 그래프 실행(`_process_clause`)
3. domain이 "해당없음"인 조항은 결과에서 제외
4. `_extract_spans()`로 evidence_span의 원문 내 위치(start/end) 계산

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

환경변수: `OPENAI_API_KEY`, `DATABASE_URL`, `FORWARD_MODEL`, `CORS_ORIGIN`, `MODEL_DIR`(KoELECTRA 체크포인트 경로, 기본 `models/v4`).
