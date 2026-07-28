# backend/agents/judgment_agent.py
"""Judgment Agent — 파인튜닝된 KoELECTRA로 domain·risk_level 최종 판단을 내린다.

GPT-4o(Analysis Agent)는 조항 유형 1차 분석·근거 문구 추출·설명 생성에만 쓰고,
실제 risk_level 판단은 이 모듈(분류 모델)이 맡는다 — 어느 체크포인트를 쓰는지,
왜 그런지는 ../../models/README.md의 버전 히스토리 참고.
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

_electra_model: DualHeadElectra | None = None
_electra_tokenizer: ElectraTokenizerFast | None = None
_electra_device: Any = None


def _get_electra() -> tuple[DualHeadElectra, ElectraTokenizerFast, Any]:
    global _electra_model, _electra_tokenizer, _electra_device
    if _electra_model is None:
        _electra_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _electra_model  = DualHeadElectra(str(MODEL_DIR))
        heads = torch.load(MODEL_DIR / "heads.pt", map_location=_electra_device, weights_only=True)
        _electra_model.domain_head.load_state_dict(heads["domain_head"])
        _electra_model.risk_head.load_state_dict(heads["risk_head"])
        _electra_model.to(_electra_device).eval()
        _electra_tokenizer = ElectraTokenizerFast.from_pretrained(str(MODEL_DIR))
    return _electra_model, _electra_tokenizer, _electra_device


def electra_predict(text: str) -> tuple[str, str]:
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
    return INV_DOMAIN_MAP[int(d_probs.argmax())], INV_RISK_MAP[int(r_probs.argmax())]


def judgment_node(state: ClauseState) -> dict:
    # models/v4는 evidence_span 길이(평균 40자대) 위주로 학습됐으므로, 전체 조항
    # 대신 evidence_span을 그대로 넣어야 학습·추론 입력 분포가 맞는다
    # (evidence_span이 없으면 전체 조항으로 폴백).
    query = state.get("evidence_span") or state["clause"]
    electra_domain, risk_level = electra_predict(query)
    verified = (electra_domain == state.get("domain"))
    return {"risk_level": risk_level, "verified": verified}
