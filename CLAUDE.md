# CLAUDE.md

이 파일은 Claude Code(claude.ai/code)가 이 저장소에서 작업할 때 참고하는 지침입니다.

## 언어 규칙

모든 응답, 코드 주석, 출력 메시지, 실행 결과는 반드시 **한국어**로 작성한다.

## 프로젝트 개요

ContractRisk-Guard는 한국 법령 문서를 활용하여 계약서의 위험도를 분석하는 초기 단계 Python 프로젝트입니다. 국가법령정보 API에서 법령을 수집·처리하여 계약 위험 요소를 평가하는 것을 목표로 합니다.

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

## 환경 설정

의존성 파일이 아직 없으므로 필요한 패키지를 직접 설치합니다:

```bash
pip install requests
```
