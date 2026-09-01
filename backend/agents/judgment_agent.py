# backend/agents/judgment_agent.py
"""Judgment Agent — 파인튜닝된 KoELECTRA로 **위반 소지 조**를 판단한다.

GPT-4o(Analysis Agent)는 조항 유형 1차 분석·근거 문구 추출·설명 생성에만 쓰고,
최종 판단은 이 모듈(분류 모델)이 맡는다.

## 위험도 3단계를 더 이상 내지 않는다 (2026-08-31)

이전에는 `models/v4`(dual-head)가 domain 2종 + risk_level 3단계를 냈고, 화면이
High/Medium/Low를 표시했다. 두 가지가 그 설계를 무너뜨렸다:

1. **risk_level에 외부 준거가 없다.** 조 multi-label로 taxonomy를 바꾸면서 risk 헤드를
   일부러 뺐다 — gold를 정의할 방법이 없었다. 검증되지 않은 3단계를 화면에 띄우는 것은
   `out_of_scope`에서 막으려던 것과 같은 종류의 거짓 확신이다.
2. **`confidence_band`의 실측 정확도(57.5% / 46.8% / 38.2%)는 v4 전용이다.**
   n=453, 옛 라벨 정의 위에서 잰 값이라 새 모델로 옮길 수 없다. 옮기면 숫자만 남고
   근거는 사라진다.

## 지금 내는 것과, 그것이 서는 근거

    출력      위반 소지 조 목록(빈 리스트 가능)
    주장      "이 조항을 확인해 보세요 — 제N조 관련으로 보입니다"
    근거      조항 단위 재현 78.0% · 오경보 2.6%  (배포 임계값, `article_gold_eval`)
              ← r(계약서 내 위반 비율)과 무관한 모델 속성이라 r 없이도 보고할 수 있다

조 단위 정밀도는 그보다 낮다(44%대). 그래서 **조 이름은 단정하지 않고 참고로 붙인다** —
"제9조 위반입니다"는 44% 위에 서지만 "이 조항을 확인해 보세요"는 78% 위에 선다.
같은 모델·같은 임계값인데 무엇을 주장하느냐가 어느 지표로 채점되는지를 정한다.

임계값은 학습 때 dev에서 확정한 조별 값(`thresholds.npy`)을 그대로 쓴다. 배포 유병률 r로
재최적화하면 최대 10% 비용 개선이 있으나, r을 신뢰성 있게 측정할 방법이 없어 하지 않는다
(`backend/eval/threshold_r_sweep.py`, `prevalence_worksheet.py` 참고).
"""

import os
from pathlib import Path
from typing import Any

import torch
from transformers import ElectraTokenizerFast

from backend.agents.state import ClauseState
from backend.model.electra import ArticleMultiLabelElectra
from backend.utils import PROJECT_ROOT

MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(PROJECT_ROOT / "models/article_v1")))


def model_version() -> str:
    """판단에 쓰인 체크포인트 이름. 저장된 분석 결과에 함께 남긴다.

    기본값은 `models/article_v1`(조 multi-label)이다. 옛 `models/v4`(domain 2종 + risk
    3단계)는 **서빙에서 빠졌다** — 누수 있는 분할이었고, 학습 라벨의 16.4%를 정확도
    45%짜리 이전 세대가 결정했다.

    **이 필드가 있어야 taxonomy가 다른 두 세대의 저장분을 나중에 구분할 수 있다.**
    v4 시절 결과에는 `risk_level`·`confidence_band`가 있고 `articles`가 없다 —
    `analyses.result`가 JSONB라 마이그레이션 없이 함께 저장되므로
    `WHERE result->>'model_version' = 'v4'`로 옛 판정을 골라낼 수 있다.
    """
    return MODEL_DIR.name
# 프로젝트 규칙상 GPU는 cuda:1 고정(`Claude.md`). 이전에는 `torch.device("cuda")`라
# 인덱스 없이 잡아 항상 cuda:0으로 갔다 — EXAONE(16GB)과 같은 GPU에 몰리는 원인이었다.
JUDGMENT_DEVICE = os.environ.get("JUDGMENT_DEVICE", "cuda:1")

_electra_model: ArticleMultiLabelElectra | None = None
_thresholds: Any = None
_electra_tokenizer: ElectraTokenizerFast | None = None
_electra_device: Any = None


def _get_electra() -> tuple[ArticleMultiLabelElectra, ElectraTokenizerFast, Any]:
    """`ArticleMultiLabelElectra.load()`로 읽는다 — 직접 조립하면 지문 검사를 건너뛴다.

    임계값은 학습 때 dev에서 확정한 조별 값을 그대로 쓴다(`thresholds.npy`).
    **여기서 다시 고르지 않는다** — 그게 곧 평가셋 오염이다.
    """
    global _electra_model, _electra_tokenizer, _electra_device, _thresholds
    if _electra_model is None:
        import numpy as np
        _electra_device = torch.device(JUDGMENT_DEVICE if torch.cuda.is_available() else "cpu")
        _electra_model = ArticleMultiLabelElectra.load(MODEL_DIR).to(_electra_device).eval()
        _electra_tokenizer = ElectraTokenizerFast.from_pretrained(str(MODEL_DIR))
        _thresholds = np.load(MODEL_DIR / "thresholds.npy")
    return _electra_model, _electra_tokenizer, _electra_device


def electra_predict(text: str) -> list[str]:
    """조항이 위반할 가능성이 있는 **약관규제법 조 목록**을 반환한다. 빈 리스트면 미지목.

    ## 위험도 3단계를 더 이상 내지 않는다 (2026-08-31)

    조 multi-label 모델에는 risk 헤드가 없다 — gold가 정의되지 않아 일부러 뺐다.
    `models/v4`의 3단계 위험도는 외부 준거로 검증된 적이 없고, 그 위에 얹혀 있던
    `confidence_band`의 실측 정확도(57.5% / 46.8% / 38.2%)도 v4 전용이라 여기 옮길 수 없다.
    **없는 근거를 화면에 표시하지 않는다.**

    대신 제품이 주장하는 것은 **조항 지목**이다. 그 주장이 서는 지표는 측정돼 있다
    (배포 임계값, `article_gold_eval`):

        조항 단위 재현 78.0% · 오경보 2.6%     ← r(계약서 내 위반 비율)과 무관한 모델 속성
        조 단위 정밀도는 그보다 낮으므로(44%대) **조 이름은 참고로만 붙인다**

    화면 문구가 "이 조항을 확인해 보세요 — 제N조 관련으로 보입니다"인 이유가 이것이다.
    """
    model, tokenizer, device = _get_electra()
    enc = tokenizer(text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
            enc.get("token_type_ids", torch.zeros(1, 256, dtype=torch.long)).to(device),
        )
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    return [n for n, p, t in zip(model.article_names, probs, _thresholds) if p >= t]


def judgment_node(state: ClauseState) -> dict:
    # models/v4는 evidence_span 길이(평균 40자대) 위주로 학습됐으므로, 전체 조항
    # 대신 evidence_span을 그대로 넣어야 학습·추론 입력 분포가 맞는다
    # (evidence_span이 없으면 전체 조항으로 폴백).
    query = state.get("evidence_span") or state["clause"]
    articles = electra_predict(query)
    return {
        "model_articles": articles,
        # 조항을 지목했는가 = 사용자에게 보여줄 것인가. 이진이다.
        "needs_review": bool(articles),
        # GPT와 모델이 **같은 조**를 짚었는지. 신뢰도가 아니라 두 판단의 일치 여부다.
        # 옛 구현은 `electra_domain == state["domain"]`으로 2-도메인 값을 비교했는데,
        # taxonomy가 조 단위로 바뀐 지금은 조 교집합이 맞는 비교다.
        "verified": bool(set(articles) & set(state.get("articles") or [])),
    }
