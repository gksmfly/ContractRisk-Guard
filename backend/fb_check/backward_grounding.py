# backend/fb_check/backward_grounding.py
"""
Backward Grounding: evidence_span ⊂ C 검증 + KoELECTRA 예측

두 가지 검증을 수행한다.
  1. snippet_exists: GPT가 추출한 evidence_span이 원문에 실제로 존재하는지 확인
  2. predict: KoELECTRA로 조항 텍스트의 domain·risk_level을 독립적으로 예측
"""

import difflib
import re
from pathlib import Path

import torch
from transformers import ElectraTokenizerFast

from backend.model.electra import DualHeadElectra, INV_DOMAIN_MAP, INV_RISK_MAP

_PAGE_MARKER = re.compile(r'\s*-\s*\d+\s*-\s*')
_FUZZY_MATCH_THRESHOLD = 0.85  # 완전 일치 실패 시, 최장 공통 부분열이 근거 문구의 이 비율 이상이면 인정


def load_model(model_dir: Path, device: torch.device) -> tuple[DualHeadElectra, ElectraTokenizerFast]:
    model = DualHeadElectra(str(model_dir))
    heads = torch.load(model_dir / "heads.pt", map_location=device, weights_only=True)
    model.domain_head.load_state_dict(heads["domain_head"])
    model.risk_head.load_state_dict(heads["risk_head"])
    model.to(device).eval()
    tokenizer = ElectraTokenizerFast.from_pretrained(str(model_dir))
    return model, tokenizer


def snippet_exists(clause_text: str, evidence_span: str) -> bool:
    """evidence_span이 clause_text 안에 있는지 확인한다.

    완전 일치를 우선 시도하고, 실패하면 퍼지 매칭으로 한 번 더 확인한다 — PDF에서
    텍스트를 추출할 때 표(비교표 등)의 셀이 뒤섞여 문장이 중복·재배열되는 경우가
    있는데, 이때는 근거 문구 자체는 실존해도 완전 일치가 깨진다.
    """
    if not evidence_span or len(evidence_span) < 10:
        return False
    norm_text = " ".join(_PAGE_MARKER.sub(" ", clause_text).split())
    norm_span = " ".join(evidence_span.split())

    if norm_span in norm_text:
        return True

    matcher = difflib.SequenceMatcher(None, norm_span, norm_text, autojunk=False)
    match = matcher.find_longest_match(0, len(norm_span), 0, len(norm_text))
    return (match.size / len(norm_span)) >= _FUZZY_MATCH_THRESHOLD


def predict(
    text: str, model: DualHeadElectra, tokenizer: ElectraTokenizerFast, device: torch.device,
) -> tuple[str, str]:
    enc = tokenizer(text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        d_logits, r_logits = model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
            enc.get("token_type_ids", torch.zeros(1, 256, dtype=torch.long)).to(device),
        )
    return INV_DOMAIN_MAP[d_logits.argmax().item()], INV_RISK_MAP[r_logits.argmax().item()]
