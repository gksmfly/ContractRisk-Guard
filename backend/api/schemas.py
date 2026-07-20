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
    confidence: float
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
