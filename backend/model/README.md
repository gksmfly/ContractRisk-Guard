# backend/model/

모델 클래스 정의. `backend.training.train`(학습)과 `backend.fb_check.backward_grounding`(FB-Check 역방향 검증), `backend.agents.judgment_agent`(프로덕션 추론) 세 곳에서 공유합니다.

---

## 파일 목록

### `electra.py`

**`DualHeadElectra`** — `monologg/koelectra-base-v3-discriminator` 인코더 위에 도메인 분류 헤드와 위험도 분류 헤드를 나란히 얹은 듀얼 헤드 모델.
- `forward()`: `[CLS]` 토큰 임베딩(dropout 0.1) → 두 개의 `nn.Linear` 헤드로 각각 도메인/위험도 로짓 반환
- `save()`: 인코더는 `save_pretrained()`(HuggingFace 표준), 두 헤드는 별도로 `heads.pt`에 `state_dict` 저장 — `judgment_agent.py`가 로드할 때 인코더와 헤드를 따로 복원하는 이유

**레이블 매핑 상수** (양방향 dict 전부 존재 — 학습 땐 정방향, 추론 땐 역방향 사용)
```python
DOMAIN_MAP = {"해지_조항": 0, "책임제한_조항": 1}
RISK_MAP   = {"High": 0, "Medium": 1, "Low": 2}
INV_DOMAIN_MAP = {v: k for k, v in DOMAIN_MAP.items()}
INV_RISK_MAP   = {v: k for k, v in RISK_MAP.items()}
```

---

## 참고

이 모듈 자체는 체크포인트를 만들지도, 로드하지도 않습니다 — 순수 아키텍처 정의입니다. 실제 체크포인트(`models/v1`~`v9`)와 버전별 성능·의사결정은 `models/README.md`, 학습은 `backend/training/README.md`, 프로덕션 추론은 `backend/agents/README.md`의 `judgment_node` 항목 참고.
