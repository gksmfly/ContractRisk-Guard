# tests/

`backend/api/services/analyze.py` (분석 파이프라인 진입점)에 대한 pytest 테스트.

## 구성

| 파일 | 범위 | 외부 의존성 |
|---|---|---|
| `test_analyze_unit.py` | `split_clauses`, `_extract_spans` 등 순수 함수 단위 테스트 | 없음 — API 키/GPU 없이 항상 실행 가능 |
| `test_analyze_integration.py` | `run_analyze()` 전체 파이프라인(GPT-4o + KoELECTRA + DB 검색) | OpenAI API(비용 발생), GPU, `.env` 설정, DB 필요 |

## 실행

```bash
# 평소: 빠른 단위 테스트만
pytest -m "not integration"

# 배포 전 등: 통합 테스트 포함 (비용 발생, .env 필요)
pytest -m integration
```

`pytest.ini`에 `integration` 마커가 등록되어 있고, `asyncio_mode = auto`로 설정되어 있어 `async def test_...` 함수를 별도 데코레이터 없이 바로 사용할 수 있다.

## 현재 커버리지

- `backend/api/services/analyze.py`만 테스트 대상. `backend/fb_check/`, `backend/training/`, `backend/scripts/` 등 나머지 파이프라인 단계는 아직 자동화된 테스트가 없음(주로 1회성 스크립트 실행 후 결과 파일을 수동 검증하는 방식으로 운영 중).
- `backend/api/services/latency_benchmark.py`는 pytest가 아니라 별도 스크립트로 지연시간을 측정한다(테스트가 아닌 벤치마크).
