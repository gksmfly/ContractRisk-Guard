# CLAUDE.md

이 파일은 Claude Code(claude.ai/code)가 이 저장소에서 작업할 때 참고하는 지침입니다.

## 언어 규칙

모든 응답, 코드 주석, 출력 메시지, 실행 결과는 반드시 **한국어**로 작성한다.

## 코드 작성 규칙

### 기본 원칙
- 코드의 기준(Source of Truth)은 로컬(Mac)이다
- 서버에서 직접 코드 수정 금지
- 모든 코드는 Git을 통해서만 서버에 반영한다
- Jupyter Notebook에서 코드 수정 금지
- 모든 파일 첫 줄에는 반드시 해당 파일 경로를 주석으로 작성한다
```python
# scripts/crawl_law.py
```

### 언어 및 버전
- Python 3.11 이상
- 타입 힌트 필수 (`def func(x: str) -> dict:`)
- f-string 사용 (`%` 포맷, `.format()` 금지)

### 파일 및 디렉토리 네이밍
- 파일명: `snake_case` (crawl_law.py, hybrid_retriever.py)
- 클래스명: `PascalCase` (AnalysisAgent, RiskPolicy)
- 상수명: `UPPER_SNAKE_CASE` (DOMAIN_LAWS, BASE_URL)
- 경로 하드코딩 금지, 반드시 `.env` 또는 `Path()` 사용

### 환경변수
- 모든 민감 정보는 `.env`에서 관리
- `.env`는 절대 Git에 올리지 않는다
- 로컬: `.env.local` → `.env`로 복사해서 사용
- 서버: `.env.server` → `.env`로 복사해서 사용

### 함수 작성
- 함수 하나는 하나의 역할만
- 함수명은 동사로 시작 (`fetch_`, `crawl_`, `save_`, `build_`)
- 모든 함수에 docstring 작성
```python
def fetch_law_detail(auth_key: str, mst_id: str) -> dict | None:
    """
    MST 번호로 법령 상세 내용을 조회합니다.

    Args:
        auth_key: API 인증키
        mst_id: 법령일련번호

    Returns:
        법령 상세 내용 딕셔너리, 실패 시 None
    """
```

### 예외 처리
- 모든 API 호출은 `try-except`로 감싼다
- 에러는 `print()` 대신 `logger` 사용
- 실패해도 전체 프로세스가 멈추지 않도록 처리

### 로깅
- `print()` 사용 금지, 반드시 `logger` 사용
- 수집 스크립트는 `logs/` 폴더에 파일 로그 저장
- 로그 파일은 Git에 올리지 않는다

### 데이터 관리
- 실제 데이터는 서버에만 저장
- 로컬에는 샘플 데이터만 사용
- 대용량 데이터 Git 업로드 금지
- 데이터 저장 경로는 `.env`로 관리

### Agent 작성 규칙
- 모든 Agent는 `base_agent.py` 상속
- Agent 하나는 하나의 역할만
- 입력/출력 타입 명시 필수
- 실패 조건 반드시 정의

### Git 규칙
- `main`: 안정 버전
- `dev`: 개발용
- 서버에서는 `pull`만 수행, `commit` 금지

## 프로젝트 개요

ContractRisk-Guard는 한국 법령 문서를 활용하여 **기업-개인(B2C) 서비스 계약서**의 위험도를 분석하는 초기 단계 Python 프로젝트입니다. 국가법령정보 API에서 법령을 수집·처리하여 계약 위험 요소를 평가하는 것을 목표로 합니다.

## 분석 도메인

현재 2개 도메인을 기준으로 개발 진행:

### 1. 해지 조항
- 일방적 해지, 해지 통보 기간, 위약금, 해지 사유 모호성
- 관련 법령: 약관규제법(§9), 민법(§543~553), 전자상거래법(§17), 상법(§64~69)

### 2. 책임 제한 조항
- 전면 면책, 간접손해 배제, 손해배상 상한, 고의·중과실 배제 여부
- 관련 법령: 약관규제법(§7), 민법(§750~766), 전자상거래법(§21), 소비자기본법(§19), 제조물책임법(§3~4)

> 추후 자동갱신 조항, 청약철회 조항, 개인정보 조항 등으로 도메인 확장 예정

## 아키텍처

프로젝트는 다음 구조로 구성됩니다:

- **scripts/**: 데이터 수집 스크립트 (법령 크롤링 등)
- **data/**: 원본 및 가공된 법령 데이터 저장소
- **models/**: 위험 평가 ML 모델
- **backend/**: API/서비스 레이어
- **frontend/**: 사용자 인터페이스
- **experiments/**: 실험용 노트북/스크립트
- **configs/**: 설정 파일
- **docker/**: 컨테이너화 설정
- **tests/**: 테스트 모음

## 국가법령정보 API

주요 데이터 출처는 국가법령정보 시스템(`http://www.law.go.kr/DRF/lawService.do`)입니다. 주요 파라미터:
- `OC`: API 인증키
- `target`: `"law"` (법령 대상)
- `MST`: 법령일련번호 (예: `253527` → 약관법)
- `type`: `"JSON"` (응답 형식)

수집된 원본 JSON은 `data/raw/` 경로에 저장됩니다.

## 데이터 수집 사용법

전체 수집 후 파싱·필터링 단계에서 도메인별 분류를 진행합니다.

```bash
# 법령 전체 수집 → data/raw/laws/
python scripts/crawl_law.py --key <인증키>

# 판례 전체 수집 → data/raw/precedents/
python scripts/crawl_precedent.py --key <인증키>

# 법령해석례 전체 수집 → data/raw/interpretations/
python scripts/crawl_interpretation.py --key <인증키>
```

## 환경 설정

- Python 버전: **3.11.10** (`.python-version` 참조)
- 가상환경: `venv/`
```bash
source venv/bin/activate
pip install -r requirements.txt
```