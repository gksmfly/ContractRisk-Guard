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
    """확인이 필요하다고 판단된 조항.

    **위험도 3단계와 신뢰도 구간을 내지 않는다 (2026-08-31).** 조 multi-label 모델에는
    risk 헤드가 없고(gold 미정의로 일부러 제외), `confidence_band`의 실측 정확도는
    `models/v4` 전용 값이라 옮길 수 없다. 검증되지 않은 등급을 화면에 띄우는 것은
    `OutOfScopeClause`가 막으려는 것과 같은 종류의 거짓 확신이다.

    대신 이진 판단(`needs_review`)과 참고용 조 목록(`articles`)을 낸다. 근거:

        조항 단위 재현 78.0%     ← "이 조항을 확인하라"는 주장이 서는 지표
        조 단위는 상수와 미판정    ← 그래서 조 이름은 **단정하지 않고 참고로만**

    오경보율은 아직 없다 — 예전 "2.6%"는 준거가 GPT 라벨이라 순환이어서 철회했다
    (2026-08-31). **이 연구 범위에서는 측정하지 않는다** — 독립 준거(사람 판단)를 만들
    방법이 없어 한계로 남긴다. 평가셋은 `data/eval/prevalence/evalset_v1.json`에 얼려 뒀다.
    """
    id: int
    original: str
    # 모델이 지목한 약관규제법 조. **참고값이다** — 조 단위 정밀도가 재현보다 훨씬 낮다.
    articles: list[str] = []
    needs_review: bool = True
    domain: str = ""          # 옛 2-도메인 파생값. 저장된 과거 결과와의 호환용으로만 남긴다
    evidence_spans: list[EvidenceSpan]
    # 예측한 조에서 매핑한 약관규제법 조문. 검색을 쓰지 않으므로 무관한 법이 안 섞인다.
    legal_basis: list[LegalBasis]
    # 검색으로 찾은 유사 판례. **근거가 아니라 참고다** — hit@5 14%(무작위 5.3%)이므로
    # 화면에서 "적용 법령"과 같은 위계로 두면 안 된다.
    precedent_refs: list[LegalBasis] = []
    reasoning: str
    verified: bool
    redteam_note: str = ""
    evidence_verified: bool = True


class OutOfScopeClause(BaseModel):
    """모델이 조를 지목하지 않은 조항.

    **어떤 등급도 붙이지 않는다.** "확인되지 않았다"이지 "안전하다"가 아니다.
    조 단위 재현이 78%이므로 **약 5건 중 1건은 여기 잘못 들어와 있다** — 등급을 붙이면
    화면에서 "검토했고 문제없음"으로 읽혀 사용자가 그 조항을 다시 안 본다.
    누락보다 거짓 안심이 나쁘다.
    """
    id: int
    original: str
    reason: str


class AnalyzeResponse(BaseModel):
    total_clauses: int
    # 확인이 필요하다고 판단된 조항 수. 위험도 3단계를 내지 않으므로 세 칸이 아니라 하나다.
    review_count: int = 0
    # 옛 3단계 카운트. 저장된 과거 결과(analyses.result JSONB)와의 호환용으로만 남기며
    # 새 응답에서는 항상 0이다 — 화면에서 읽지 말 것.
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
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
    # **키 이름을 바꾸지 말 것.** 대시보드가 `result->>'model_version'`으로 읽고,
    # 저장된 판정의 세대(v4 / article_v1 / article_v2)를 가르는 **유일한 수단**이다.
    # 예전에 일괄 rename이 함수명(`get_model_version`)을 필드까지 끌고 가서 JSON 키가
    # `get_model_version`이 됐고, 대시보드 쿼리가 조용히 NULL만 받고 있었다 —
    # 빈칸으로 렌더되므로 화면으로는 안 드러난다.
    model_version: str
