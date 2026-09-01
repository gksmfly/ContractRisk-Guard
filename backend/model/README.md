# backend/model/

모델 클래스 정의. 헤드가 둘이고 쓰임이 갈립니다 — `ArticleMultiLabelElectra`는 프로덕션 판단(`backend.agents.judgment_agent`)과 그 학습(`backend.training.train_article`)이, `DualHeadElectra`는 전량 라벨링 경로(`backend.fb_check.backward_grounding`)와 옛 학습(`backend.training.train`)이 씁니다.

---

## 파일 목록

### `electra.py`

**`ArticleMultiLabelElectra`** (프로덕션) — 같은 인코더 위에 **조 multi-label 헤드** 하나. 한 조항이 여러 조에 걸리는 게 정상이라(공정위 의결서 기준 케이스당 평균 2.08개) softmax가 아니라 sigmoid + `BCEWithLogitsLoss`를 쓴다.
- 출력 차원이 고정 상수가 아니라 **데이터가 정한 조 목록** 길이 — `article_labels(counts, min_support=5)`가 support 기준으로 고르고, 접힌 조(제13조)는 예측 대상에서 빠진다
- `risk_level` 헤드는 **일부러 없다**. 조에서 risk를 유도하는 규칙을 지금 정하면 또 근거 없는 상수가 되고, risk의 gold 자체가 미정이다
- `save()`/`load()`가 짝이다 — 조 이름·임계값까지 함께 저장해 출력 차원과 조 이름의 대응이 깨지지 않게 한다

**`DualHeadElectra`** (라벨링 경로 전용) — `monologg/koelectra-base-v3-discriminator` 인코더 위에 도메인 분류 헤드와 위험도 분류 헤드를 나란히 얹은 듀얼 헤드 모델.
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

`DualHeadElectra`는 체크포인트를 만들지도 로드하지도 않는 순수 아키텍처 정의지만, `ArticleMultiLabelElectra`는 `save()`/`load()`를 직접 갖습니다(저장 형식을 아는 곳을 한 군데로 묶기 위함). 실제 체크포인트(프로덕션 `models/article_v1`, 옛 세대 `models/v1`~`v9`)와 버전별 성능·의사결정은 `models/README.md`, 학습은 `backend/training/README.md`, 프로덕션 추론은 `backend/agents/README.md`의 `judgment_node` 항목 참고.
