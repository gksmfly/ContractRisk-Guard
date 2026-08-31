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


class OutOfScopeClause(BaseModel):
    """분석 범위 밖이라 판단하지 않은 조항.

    **위험도를 붙이지 않는다.** 지금 도메인 체계(해지·책임제한 2종)에 안 걸린다는 뜻이지
    "안전하다"는 뜻이 아니다 — 개인정보·관할·정의 조항이 전부 여기 들어온다. `risk_level`을
    Low로 채우면 화면에서 "검토했고 문제없음"으로 읽혀, 사용자가 그 조항을 다시 안 본다.
    누락보다 거짓 안심이 나쁘다(판례 근거를 그대로 노출하지 않기로 한 것과 같은 계열).
    """
    id: int
    original: str
    reason: str


class AnalyzeResponse(BaseModel):
    total_clauses: int
    high_count: int
    medium_count: int
    low_count: int
    clauses: list[ClauseResult]

    # 입력에서 분리된 조항 수. `total_clauses`(=분석된 수)와 다를 수 있다 —
    # 예전에는 이 차이가 응답에 전혀 안 드러나 조항이 조용히 사라졌다.
    input_clauses: int = 0
    # 조항 수 상한을 넘겨 분석하지 않은 수(0이면 절삭 없음)
    truncated_clauses: int = 0
    # 도메인 범위 밖이라 판단하지 않은 조항들 — 위험도 없이 목록만
    out_of_scope: list[OutOfScopeClause] = []

    # 어느 체크포인트가 이 판단을 냈는지. 프론트가 `analyses.result`(JSONB)에 통째로
    # 저장하므로 이 필드만 있으면 별도 컬럼·마이그레이션 없이 추적된다.
    # 모델을 교체할 때 이전 결과를 골라내려면 이 값이 반드시 있어야 한다.
    model_version: str
