# backend/retrieval/

법령·판례 Hybrid(Dense+Sparse) 검색 — `backend.agents`의 Retrieval Strategy/Evidence Selection/Red-team이 공통으로 재사용하는 DB 서비스 레이어. 원래 `backend/api/services/`에 있었으나, agents가 api를 거꾸로 import하게 만들어 여기로 분리했습니다(2026-09-26). 이 패키지는 `backend.api`·`backend.agents`를 import하지 않습니다 — 의존 방향은 `api → agents → retrieval → db` 한 방향입니다.

---

## 파일 목록

### `search.py`
`chunks` 테이블(법령 조문 + 판례) 대상 Hybrid 검색과 `clean_clauses` 대상 유사 라벨 조항 검색. 상세는 코드 상단 docstring 참고(RRF 융합, KoE5 임베딩, pg_trgm Sparse, 법원 심급 가중치 등 설계 근거가 정리돼 있음).

### `schemas.py`
`LegalBasis` — 검색 결과를 근거 한 건으로 바꾼 형태. agents 상태와 API 응답(`backend.api.schemas.ClauseResult`)이 같이 씁니다.
