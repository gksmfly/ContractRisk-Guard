import os
import re
from pathlib import Path

from fastapi import HTTPException

from backend.api.schemas import AnalyzeResponse, ClauseResult, EvidenceSpan, LegalBasis
from backend.api.services.retrieval import search_legal_basis

MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(Path(__file__).parent.parent.parent.parent / "models/v2")))

# search_legal_basis()가 DB 미가동/미적재 등으로 결과를 못 찾을 때만 쓰는 최종 fallback.
LEGAL_BASIS_FALLBACK: dict[str, list[dict]] = {
    "해지_조항": [
        {"law": "약관규제법", "article": "제9조",      "description": "고객에게 부당하게 불리한 해제·해지권 부여 조항 무효"},
        {"law": "민법",      "article": "제543~553조", "description": "계약 해제·해지의 일반 요건 및 효과"},
    ],
    "책임제한_조항": [
        {"law": "약관규제법", "article": "제7조",      "description": "사업자 책임 부당 면제·제한 조항 무효"},
        {"law": "민법",      "article": "제750~766조", "description": "불법행위 손해배상 책임"},
    ],
    "해당없음": [],
}

_electra_model     = None
_electra_tokenizer = None
_electra_device    = None


def _get_electra():
    global _electra_model, _electra_tokenizer, _electra_device
    if _electra_model is None:
        import torch
        from transformers import ElectraTokenizerFast
        from backend.model.electra import DualHeadElectra
        _electra_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _electra_model  = DualHeadElectra(str(MODEL_DIR))
        heads = torch.load(MODEL_DIR / "heads.pt", map_location=_electra_device, weights_only=True)
        _electra_model.domain_head.load_state_dict(heads["domain_head"])
        _electra_model.risk_head.load_state_dict(heads["risk_head"])
        _electra_model.to(_electra_device).eval()
        _electra_tokenizer = ElectraTokenizerFast.from_pretrained(str(MODEL_DIR))
    return _electra_model, _electra_tokenizer, _electra_device


def _electra_predict(text: str) -> tuple[str, str]:
    """파인튜닝된 KoELECTRA로 domain·risk_level을 판단한다 (Judgment Agent 역할).

    GPT-4o는 조항 유형 1차 분석·근거 문구 추출·설명 생성에만 쓰고,
    실제 risk_level 판단은 이 함수(분류 모델)가 맡는다.
    """
    import torch
    import torch.nn.functional as F
    from backend.model.electra import INV_DOMAIN_MAP, INV_RISK_MAP
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


def _get_openai():
    from openai import OpenAI
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def split_clauses(text: str) -> list[str]:
    parts = re.split(
        r"(?=제\s*\d+\s*조|^\s*\d+\.\s|^[①②③④⑤⑥⑦⑧⑨⑩]|\n{2,})",
        text,
        flags=re.MULTILINE,
    )
    return [s.strip() for s in parts if len(s.strip()) > 20][:20]


def _extract_spans(clause_text: str, evidence_span: str) -> list[EvidenceSpan]:
    if not evidence_span:
        return []
    idx = clause_text.find(evidence_span)
    if idx == -1:
        return []
    return [EvidenceSpan(text=evidence_span, start=idx, end=idx + len(evidence_span))]


async def run_analyze(text: str) -> AnalyzeResponse:
    from backend.fb_check.forward_labeling import run_forward

    clauses = split_clauses(text)
    if not clauses:
        raise HTTPException(status_code=400, detail="조항을 분리할 수 없습니다.")

    client  = _get_openai()
    results: list[ClauseResult] = []

    for i, clause in enumerate(clauses):
        fwd = run_forward(client, clause)
        if fwd is None:
            fwd = {"domain": "해당없음", "risk_level": "Low", "evidence_span": "", "reasoning": ""}

        domain        = fwd.get("domain", "해당없음")
        evidence_span = fwd.get("evidence_span", "")
        reasoning     = fwd.get("reasoning", "")

        if domain not in ("해지_조항", "책임제한_조항"):
            domain = "해당없음"

        if domain == "해당없음":
            continue

        # risk_level의 최종 판단은 KoELECTRA(Judgment Agent)가 맡는다.
        # models/v2는 evidence_span 길이(평균 49자)로 학습됐으므로, 전체 조항 대신
        # evidence_span을 그대로 넣어야 학습·추론 입력 분포가 맞는다 (evidence_span이
        # 없으면 전체 조항으로 폴백).
        # GPT-4o의 domain 추정은 KoELECTRA와의 일치 여부로 신뢰도(verified)에만 반영한다.
        electra_domain, risk_level = _electra_predict(evidence_span or clause)
        verified = (electra_domain == domain)

        legal_basis = search_legal_basis(evidence_span or clause, top_k=2)
        if not legal_basis:
            legal_basis = [LegalBasis(**b) for b in LEGAL_BASIS_FALLBACK.get(domain, [])]

        results.append(ClauseResult(
            id             = i + 1,
            original       = clause,
            domain         = domain,
            risk_level     = risk_level,
            confidence     = 1.0 if verified else 0.7,
            evidence_spans = _extract_spans(clause, evidence_span),
            legal_basis    = legal_basis,
            reasoning      = reasoning,
            verified       = verified,
        ))

    high   = sum(1 for r in results if r.risk_level == "High")
    medium = sum(1 for r in results if r.risk_level == "Medium")
    low    = sum(1 for r in results if r.risk_level == "Low")

    return AnalyzeResponse(
        total_clauses = len(results),
        high_count    = high,
        medium_count  = medium,
        low_count     = low,
        clauses       = results,
    )
