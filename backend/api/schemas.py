# backend/api/schemas.py
from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    text: str


class EvidenceSpan(BaseModel):
    text: str
    start: int
    end: int


class LegalBasis(BaseModel):
    law: str
    article: str
    description: str


class ClauseResult(BaseModel):
    id: int
    original: str
    domain: str
    risk_level: str
    # 신뢰도는 구간으로만 내보낸다 — KoELECTRA softmax는 보정이 안 돼 있어(ECE 0.289)
    # 원시 확률을 %로 노출하면 실제보다 30%p 이상 과신하게 된다.
    # confidence_band_accuracy는 그 구간의 실측 정확도(v4, n=453)로, 화면에서
    # "높음"이 절대적 확신처럼 읽히지 않도록 함께 표시한다.
    # 근거: backend/eval/confidence_calibration.py, judgment_agent.py docstring
    confidence_band: str
    confidence_band_accuracy: float
    evidence_spans: list[EvidenceSpan]
    legal_basis: list[LegalBasis]
    reasoning: str
    verified: bool
    redteam_note: str = ""
    evidence_verified: bool = True


class AnalyzeResponse(BaseModel):
    total_clauses: int
    high_count: int
    medium_count: int
    low_count: int
    clauses: list[ClauseResult]
