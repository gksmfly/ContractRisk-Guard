# CLAUDE.md

이 파일은 Claude Code(claude.ai/code)가 이 저장소에서 작업할 때 참고하는 지침입니다.

## 언어 규칙

모든 응답, 코드 주석, 출력 메시지, 실행 결과는 반드시 **한국어**로 작성한다.

## 코드 작성 규칙



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

```bash
# 전체 도메인 법령 수집
python scripts/crawl_law.py --key <인증키>

# 해지 조항 관련 법령만 수집
python scripts/crawl_law.py --key <인증키> --domain termination

# 책임 제한 관련 법령만 수집
python scripts/crawl_law.py --key <인증키> --domain liability

# 키워드로 법령 검색 (새 법령 탐색용)
python scripts/crawl_law.py --key <인증키> --query 약관

# 특정 MST 법령 수집
python scripts/crawl_law.py --key <인증키> --mst 253527
```

## 환경 설정

- Python 버전: **3.11.10** (`.python-version` 참조)
- 가상환경: `venv/`

```bash
source venv/bin/activate
pip install -r requirements.txt
```
