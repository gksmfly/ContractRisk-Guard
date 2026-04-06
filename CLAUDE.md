# CLAUDE.md

이 파일은 Claude Code(claude.ai/code)가 이 저장소에서 작업할 때 참고하는 지침입니다.

## 언어 규칙

모든 응답, 코드 주석, 출력 메시지, 실행 결과는 반드시 **한국어**로 작성한다.

## 프로젝트 개요

**ContractRisk-Guard**
: AI Agent 기반 검증 루프 + Red-team 기업 법무 플랫폼 (계약 해지 + 책임 제한 조항 리스크 예측 도메인)
→ 법률 판단의 구조화 + 근거 검증 자동화 + **Forward-Backward Consistency Check 기반 데이터 자동 정제 및 학습 루프**

### 핵심 Contribution
- 공정위 시정조치 사례(Seed)를 기준점으로 부트스트래핑
- GPT-4o Forward 라벨링 + 학습된 모델 Backward 검증으로 노이즈 자동 탐지·제거
- 전문가 없이 고품질 법률 데이터셋 자동 구축 → 정제 데이터로 분류 모델 학습 → 에이전트 엔진으로 사용

## 분석 도메인

### 1. 해지 조항
: 일방적 해지, 해지 통보 기간, 위약금, 해지 사유 모호성
- 관련 법령: 약관규제법(§9), 민법(§543~553), 전자상거래법(§17)

### 2. 책임 제한 조항
: 전면 면책, 간접손해 배제, 손해배상 상한, 고의·중과실 배제 여부
- 관련 법령: 약관규제법(§7), 민법(§750~766), 소비자기본법(§19)

## 아키텍처

```
scripts/       데이터 수집 스크립트 (법령, 판례, 심결례, 공정위 시정조치)
data/          원본 및 가공 데이터 (raw/, processed/, seed/)
models/        Forward-Backward 파이프라인, 분류 모델 학습
backend/       FastAPI 기반 API 레이어, LangGraph Agent 오케스트레이션
frontend/      사용자 인터페이스 (대시보드, 리포트)
experiments/   실험용 노트북
configs/       설정 파일
tests/         테스트 모음
docs/          설계 문서, 파이프라인 명세
```

## 데이터 구조

### 수집 데이터
- 법령 데이터: 국가법령정보 공동활용 API → `data/raw/laws/`
- 판례 데이터: 국가법령정보 공동활용 API → `data/raw/precedents/`
- 법령해석례: 국가법령정보센터 → `data/raw/interpretations/`
- 표준계약서 20종: 공정거래위원회 → `data/raw/contracts/`
- **공정위 시정조치 사례 (Seed)**: case.ftc.go.kr → `data/seed/` ← Ground Truth

### 데이터 3층 구조
1. 패턴층: 계약서 조항 데이터 (형태 분석)
2. 규범층: 법령 + 해설 데이터 (법적 기준)
3. 근거층: 판례 데이터 (실제 증거)

## 핵심 알고리즘: Forward-Backward Consistency Check

```
Forward  : C → L         (GPT-4o가 조항 텍스트로 라벨 생성)
Backward : (C, L) → E    (학습된 모델이 근거 문구 추출, 단 E ⊂ C)
Verify   : E → L'        (문구만 보고 재라벨링)
Decision : L == L' → CLEAN / L ≠ L' → NOISE
```

### 부트스트래핑 파이프라인
1. Seed 데이터 구축: 공정위 시정조치 사례 → Ground Truth
2. 1차 모델 학습: Seed 데이터로 소형 분류 모델 학습
3. Weak-to-Strong Verification: GPT-4o(Forward) + 1차 모델(Backward+Verify)
4. Data Flywheel: CLEAN 데이터 누적 → 2차 모델 → 반복

## 기술 스택

- Backend: FastAPI
- LLM: GPT-4o (설명 생성, 검색 쿼리)
- 분류 모델: KoELECTRA 또는 KoBERT (파인튜닝)
- Embedding: text-embedding-3-large
- DB: PostgreSQL + pgvector + JSONB
- Orchestration: LangGraph
- Retrieval: Hybrid (Dense + Keyword) + Metadata filtering + Reranking

## 코드 작성 규칙

- Python 3.11 이상, 타입 힌트 필수, f-string 사용
- 파일 첫 줄에 파일 경로 주석 필수
- 함수명은 동사로 시작 (`fetch_`, `crawl_`, `save_`, `build_`)
- 모든 함수에 docstring 작성
- `print()` 사용 금지 → `logger` 사용
- 모든 API 호출은 `try-except`로 감싸기
- 경로 하드코딩 금지 → `.env` 또는 `Path()` 사용
- 모든 민감 정보는 `.env` 관리, Git 업로드 금지
- 실제 데이터는 서버에만 저장, 로컬은 샘플만 사용

## 데이터 수집 사용법

```bash
# 법령 수집
python scripts/crawl_law.py --key <인증키>

# 판례 수집
python scripts/crawl_precedent.py --key <인증키>

# 법령해석례 수집
python scripts/crawl_interpretation.py --key <인증키>

# 공정위 시정조치 사례 수집 (Seed)
python scripts/crawl_ftc_seed.py
```

## Git 규칙

- `main`: 안정 버전
- `dev`: 개발용
- 서버에서는 `pull`만 수행, `commit` 금지
- 대용량 데이터 Git 업로드 금지
