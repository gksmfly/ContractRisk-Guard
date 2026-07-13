# backend/db/

PostgreSQL(pgvector) 적재와 임베딩 모델 검증을 담당하는 모듈입니다.

---

## 파일 목록

### `loader.py`
**data/processed·labels·fb_check → PostgreSQL+pgvector 적재**

- **대상**:
  - `data/processed/chunks/*.jsonl` (법령·판례·해석례 청크) → `chunks` 테이블
  - `data/labels/seed_labeled.jsonl` (Seed 라벨) → `seed_clauses` 테이블
  - `data/fb_check/clean.jsonl` (FB-Check CLEAN) → `clean_clauses` 테이블
  - `data/fb_check/noise.jsonl` (FB-Check NOISE) → `noise_clauses` 테이블
- **임베딩 모델**: `nlpai-lab/KoE5` (dim=1024, 로컬 GPU) — 아래 벤치마크 결과에 따라 선정
  - E5 컨벤션에 따라 저장 문서는 `"passage: "` 프리픽스를 붙여 인코딩함
  - 검색 쿼리 임베딩 시에는 `"query: "` 프리픽스를 붙여야 함 (Retrieval Strategy Agent 구현 시 반영 필요)
- **방식**: 배치 단위(64건)로 임베딩 즉시 upsert — 중간에 프로세스가 끊겨도 이미 처리된 배치는 보존되며, 재실행 시 남은 레코드만 이어서 처리
- **실행**:
  ```bash
  python -m backend.db.loader              # 전체 적재
  python -m backend.db.loader --source chunks
  python -m backend.db.loader --source seed
  python -m backend.db.loader --source clean
  python -m backend.db.loader --source noise
  ```
- **환경변수 (.env)**: `DATABASE_URL`, `EMBED_DEVICE`(기본 `cuda:1`)

---

### `embedding_benchmark.py`
**Dense Retrieval 임베딩 모델 비교 — `loader.py`가 사용할 모델 선정용**

- **배경**: `loader.py`가 기존에 사용하던 `openai/text-embedding-3-large`가 한국어 법률 도메인 검색에 적합한지 검증된 적이 없어서, 실제 판례 데이터로 후보 모델들을 비교함
- **방식**: 해지·책임제한 도메인 판례(`data/domain/case`, 관련) vs 비도메인 무작위 판례(`data/raw/case`, 비관련)를 계약 리스크 검색 쿼리 6개로 얼마나 잘 구분하는지 측정
  - 평가 기준: 분리도(관련 평균 유사도 − 비관련 평균 유사도), threshold별 통과율, 처리 속도
- **실행**:
  ```bash
  python -m backend.db.embedding_benchmark
  ```
- **결과 저장**: `data/embedding_benchmark_result.json`

#### 비교 결과 (2026-07-04, 관련 500건 / 랜덤 436건, 쿼리 6개)

| 모델 | 분리도 | 관련 평균 | 랜덤 평균 | 속도(docs/sec) |
|---|---|---|---|---|
| bge-m3 | 0.0261 | 0.5220 | 0.4958 | 45.3 |
| KURE-v1 | 0.0289 | 0.4966 | 0.4677 | 42.3 |
| **KoE5** | **0.1041** | 0.3642 | 0.2602 | **101.1** |
| openai/text-embedding-3-large (기존) | 0.0463 | 0.4295 | 0.3832 | 2.9 |

**threshold 0.40 기준 통과율** (관련문서는 많이, 랜덤문서는 적게 통과해야 좋음):

| 모델 | 관련 통과 | 랜덤 통과 |
|---|---|---|
| bge-m3 | 99.2% | 95.0% |
| KURE-v1 | 97.2% | 83.5% |
| KoE5 | 32.0% | 1.6% |
| openai | 64.8% | 39.2% |

**결론**: bge-m3·KURE-v1은 절대 유사도값은 높지만 관련/랜덤이 거의 안 갈려서(threshold를 어디에 둬도 랜덤 문서가 대량 통과) 변별력이 없었다. **KoE5**는 절대값은 낮지만 관련/랜덤 분리가 뚜렷하고(threshold 0.40에서 관련 32% vs 랜덤 1.6%), 로컬 GPU 추론이라 API 레이트리밋 없이 속도도 가장 빠르며 비용이 없다. 이에 따라 `loader.py`의 운영 임베딩 모델을 `text-embedding-3-large` → `KoE5`로 교체함.

**주의**: 임베딩 모델 교체로 벡터 차원이 1536 → 1024로 바뀌므로, 기존에 적재된 테이블은 재사용할 수 없다. 재적재 전 반드시 기존 테이블을 삭제해야 한다:
```sql
DROP TABLE IF EXISTS chunks, seed_clauses, clean_clauses, noise_clauses CASCADE;
```
