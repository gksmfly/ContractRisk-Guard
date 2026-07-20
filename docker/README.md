# docker/

로컬 개발용 PostgreSQL + pgvector 컨테이너 정의.

## 구성

| 파일 | 역할 |
|---|---|
| `docker-compose.yml` | `pgvector/pgvector:pg16` 이미지 기반 단일 `postgres` 서비스 정의 |

## 실행

```bash
cd docker
docker compose up -d
```

- 포트: `5432:5432`
- 데이터 볼륨: `pgdata` (named volume, 컨테이너 재생성해도 데이터 유지)
- 연결 대상: `backend/db/loader.py`가 `.env`의 `DATABASE_URL`로 접속 (`chunks`, `seed_clauses`, `clean_clauses` 3개 테이블 적재)

## 환경변수

`POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB`는 프로젝트 루트의 `.env`(`env_file: ../.env`로 로드)에서 오버라이드 가능. 지정하지 않으면 `contractrisk-guard-db` / `1234` / `contractrisk-guard-db` 기본값 사용(로컬 개발 전용값 — 운영 배포 시에는 반드시 `.env`에 별도 값을 지정할 것).

## 참고

- 이미지에 `pgvector` 확장이 포함되어 있어 별도 `CREATE EXTENSION vector` 외 추가 설치 불필요.
- 아직 `Dockerfile`(백엔드/프론트엔드 앱 컨테이너화)은 없음 — Claude.md 파이프라인 12단계("프론트엔드 + PDF 리포트")와 함께 추후 추가 예정.
