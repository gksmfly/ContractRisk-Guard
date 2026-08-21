# backend/agents/judgment_agent.py
"""Judgment Agent — 파인튜닝된 KoELECTRA로 domain·risk_level 최종 판단을 내린다.

GPT-4o(Analysis Agent)는 조항 유형 1차 분석·근거 문구 추출·설명 생성에만 쓰고,
실제 risk_level 판단은 이 모듈(분류 모델)이 맡는다 — 어느 체크포인트를 쓰는지,
왜 그런지는 ../../models/README.md의 버전 히스토리 참고.

## risk_confidence — 구간으로만 쓸 것 (2026-08-16)

이전 버전은 softmax 확률을 계산한 뒤 argmax만 쓰고 확률을 버렸다(softmax는 순서를 바꾸지
않으므로 라벨만 쓸 거면 계산할 이유가 없는 죽은 연산이었다). 대신 `analyze.py`가
`confidence = 1.0 if verified else 0.7`이라는 하드코딩을 "신뢰도"로 내보냈는데,
`verified`는 GPT와의 domain 일치 여부이지 신뢰도가 아니다.

`backend/eval/confidence_calibration.py`로 실측한 결과(v4, 860건 중 evidence_span 보유 453건):

| 모델이 낸 확률 | 건수 | 실제 정확도 |
|---|---|---|
| 0.9~1.0 | 87 | **57.5%** |
| 0.8~0.9 | 111 | 46.8% |
| 0.7~0.8 | 88 | 45.5% |
| 0.5~0.7 | 130 | 38.2% |

두 가지가 동시에 참이다:
- **신호는 있다** — 최저 38.2% → 최고 57.5%, 23.6%p 차이. 버릴 정보가 아니다.
- **숫자는 거짓말이다** — 평균 93.7%라고 말한 구간의 실제 정확도가 57.5%다(ECE 0.289).

그래서 확률은 살리되 **원시 확률값을 그대로 노출하지 않는다.** `_confidence_band()`로
높음/중간/낮음 구간만 내보내고, 각 구간의 실측 정확도를 함께 전달해 화면에서
"이 구간의 과거 정확도"를 같이 보여줄 수 있게 한다.
"""

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import ElectraTokenizerFast

from backend.agents.state import ClauseState
from backend.model.electra import DualHeadElectra, INV_DOMAIN_MAP, INV_RISK_MAP
from backend.utils import PROJECT_ROOT

MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(PROJECT_ROOT / "models/v4")))
# 프로젝트 규칙상 GPU는 cuda:1 고정(`Claude.md`). 이전에는 `torch.device("cuda")`라
# 인덱스 없이 잡아 항상 cuda:0으로 갔다 — EXAONE(16GB)과 같은 GPU에 몰리는 원인이었다.
JUDGMENT_DEVICE = os.environ.get("JUDGMENT_DEVICE", "cuda:1")

# (확률 하한, 라벨) — 경계값은 confidence_calibration.py 실측 구간에서 가져왔다.
# accuracy는 그 구간의 **실제** 정확도(v4, n=453)로, 화면에 함께 표시해 "높음"이
# 절대적 확신처럼 읽히지 않게 한다.
_CONFIDENCE_BANDS = (
    (0.9, "높음", 0.575),
    (0.7, "중간", 0.461),
    (0.0, "낮음", 0.382),
)

_electra_model: DualHeadElectra | None = None
_electra_tokenizer: ElectraTokenizerFast | None = None
_electra_device: Any = None


def _confidence_band(prob: float) -> tuple[str, float]:
    """모델 확률 → (구간 라벨, 그 구간의 실측 정확도). 원시 확률은 노출하지 않는다."""
    for threshold, label, accuracy in _CONFIDENCE_BANDS:
        if prob >= threshold:
            return label, accuracy
    return _CONFIDENCE_BANDS[-1][1], _CONFIDENCE_BANDS[-1][2]


def _get_electra() -> tuple[DualHeadElectra, ElectraTokenizerFast, Any]:
    global _electra_model, _electra_tokenizer, _electra_device
    if _electra_model is None:
        _electra_device = torch.device(JUDGMENT_DEVICE if torch.cuda.is_available() else "cpu")
        _electra_model  = DualHeadElectra(str(MODEL_DIR))
        heads = torch.load(MODEL_DIR / "heads.pt", map_location=_electra_device, weights_only=True)
        _electra_model.domain_head.load_state_dict(heads["domain_head"])
        _electra_model.risk_head.load_state_dict(heads["risk_head"])
        _electra_model.to(_electra_device).eval()
        _electra_tokenizer = ElectraTokenizerFast.from_pretrained(str(MODEL_DIR))
    return _electra_model, _electra_tokenizer, _electra_device


def electra_predict(text: str) -> tuple[str, str, float]:
    """(domain, risk_level, risk 확률)을 반환한다.

    세 번째 값은 risk_head softmax의 최대 확률이다 — **보정되지 않은 값이므로 그대로
    사용자에게 %로 노출하면 안 된다**(모듈 docstring의 실측표 참고). `_confidence_band()`를
    거쳐 구간으로만 쓴다.
    """
    model, tokenizer, device = _get_electra()
    enc = tokenizer(text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        d_logits, r_logits = model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
            enc.get("token_type_ids", torch.zeros(1, 256, dtype=torch.long)).to(device),
        )
    d_probs = F.softmax(d_logits, dim=-1)[0]
    r_probs = F.softmax(r_logits, dim=-1)[0]
    return (
        INV_DOMAIN_MAP[int(d_probs.argmax())],
        INV_RISK_MAP[int(r_probs.argmax())],
        float(r_probs.max()),
    )


def judgment_node(state: ClauseState) -> dict:
    # models/v4는 evidence_span 길이(평균 40자대) 위주로 학습됐으므로, 전체 조항
    # 대신 evidence_span을 그대로 넣어야 학습·추론 입력 분포가 맞는다
    # (evidence_span이 없으면 전체 조항으로 폴백).
    query = state.get("evidence_span") or state["clause"]
    electra_domain, risk_level, risk_prob = electra_predict(query)
    band, band_accuracy = _confidence_band(risk_prob)
    verified = (electra_domain == state.get("domain"))
    return {
        "risk_level": risk_level,
        "verified": verified,
        "confidence_band": band,
        "confidence_band_accuracy": band_accuracy,
    }
