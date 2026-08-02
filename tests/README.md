# tests/

백엔드(`backend/`) pytest 테스트. 테스트 대상 영역별로 하위 디렉터리를 나눈다.

## 구성

```
tests/
├── api/         API 서버 인프라 (인증, rate limit, health check, DB 커넥션, 업로드)
├── analyze/     analyze 파이프라인 진입점(backend/api/services/analyze.py)
└── agents/      개별 LangGraph 에이전트 노드
```

### `tests/api/`

| 파일 | 범위 | 외부 의존성 |
|---|---|---|
| `test_auth.py` | `backend/api/auth.py`의 API 키 검증 | 없음 |
| `test_rate_limit.py` | `backend/api/rate_limit.py`의 요청 제한 로직 | 없음 |
| `test_health.py` | `/health` 라우터, DB 연결 성공/실패 분기 | 없음 (DB 연결은 mock) |
| `test_db_connection.py` | `backend/db/connection.py`의 커넥션 풀 획득/반납/롤백 | 없음 (psycopg2 mock) |
| `test_pdf_upload.py` | `/api/analyze-pdf` 업로드 검증(확장자, 용량, 파싱 실패) | 없음 |

### `tests/analyze/`

| 파일 | 범위 | 외부 의존성 |
|---|---|---|
| `test_analyze_unit.py` | `split_clauses`, `_extract_spans` 등 순수 함수 단위 테스트 | 없음 — API 키/GPU 없이 항상 실행 가능 |
| `test_analyze_concurrency.py` | 동시 요청 제한(semaphore) 및 타임아웃 처리 | 없음 |
| `test_analyze_integration.py` | `run_analyze()` 전체 파이프라인(GPT-4o + KoELECTRA + DB 검색) | OpenAI API(비용 발생), GPU, `.env` 설정, DB 필요 |

### `tests/agents/`

| 파일 | 범위 | 외부 의존성 |
|---|---|---|
| `test_retrieval_strategy_agent.py` | `backend/api/services/retrieval.py`의 RRF(reciprocal rank fusion) 병합 로직 | 없음 |
| `test_evidence_selection_agent.py` | `backend/agents/evidence_selection_agent.py` | 없음 |
| `test_evidence_verification_agent.py` | `backend/agents/evidence_verification_agent.py`의 재시도 로직 | 없음 |
| `test_red_team_agent.py` | `backend/agents/red_team_agent.py` | 없음 |

## 실행

```bash
# 평소: 빠른 단위 테스트만 (integration 마커 제외)
pytest tests/ -m "not integration"

# 특정 영역만
pytest tests/api/
pytest tests/agents/

# 배포 전 등: 통합 테스트 포함 (비용 발생, .env 필요)
pytest tests/ -m integration
```

`pytest.ini`에 `integration` 마커가 등록되어 있고, `asyncio_mode = auto`로 설정되어 있어 `async def test_...` 함수를 별도 데코레이터 없이 바로 사용할 수 있다.

## 현재 커버리지

- `backend/fb_check/`, `backend/training/`, `backend/scripts/` 등 데이터/모델 파이프라인 단계는 아직 자동화된 테스트가 없음(주로 1회성 스크립트 실행 후 결과 파일을 수동 검증하는 방식으로 운영 중).
- `backend/api/services/latency_benchmark.py`는 pytest가 아니라 별도 스크립트로 지연시간을 측정한다(테스트가 아닌 벤치마크).
- `backend/agents/graph.py`(전체 그래프 조립)와 `backend/agents/judgment_agent.py`는 개별 노드 테스트가 아직 없음 — `tests/analyze/test_analyze_integration.py`가 전체 파이프라인을 통째로 검증하는 걸로 간접 커버.
