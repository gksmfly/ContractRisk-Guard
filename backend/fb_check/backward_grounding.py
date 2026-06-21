# backend/fb_check/backward_grounding.py
"""
Backward Grounding: evidence_span ⊂ C 검증 + KoELECTRA 예측

두 가지 검증을 수행한다.
  1. snippet_exists: GPT가 추출한 evidence_span이 원문에 실제로 존재하는지 확인
  2. predict: KoELECTRA로 조항 텍스트의 domain·risk_level을 독립적으로 예측
"""

import re

import torch
from transformers import ElectraTokenizerFast

from backend.model.electra import DualHeadElectra, INV_DOMAIN_MAP, INV_RISK_MAP

_PAGE_MARKER = re.compile(r'\s*-\s*\d+\s*-\s*')


def load_model(model_dir, device: torch.device):
    model = DualHeadElectra(str(model_dir))
    heads = torch.load(model_dir / "heads.pt", map_location=device, weights_only=True)
    model.domain_head.load_state_dict(heads["domain_head"])
    model.risk_head.load_state_dict(heads["risk_head"])
    model.to(device).eval()
    tokenizer = ElectraTokenizerFast.from_pretrained(str(model_dir))
    return model, tokenizer


def snippet_exists(clause_text: str, evidence_span: str) -> bool:
    if not evidence_span or len(evidence_span) < 10:
        return False
    norm_text = " ".join(_PAGE_MARKER.sub(" ", clause_text).split())
    norm_span = " ".join(evidence_span.split())
    return norm_span in norm_text


def predict(text: str, model, tokenizer, device: torch.device) -> tuple[str, str]:
    enc = tokenizer(text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        d_logits, r_logits = model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
            enc.get("token_type_ids", torch.zeros(1, 256, dtype=torch.long)).to(device),
        )
    return INV_DOMAIN_MAP[d_logits.argmax().item()], INV_RISK_MAP[r_logits.argmax().item()]
