# backend/api/services/analyze.py
import asyncio
import os
import re
from typing import Any

from fastapi import HTTPException

from backend.agents.graph import get_graph
from backend.api.schemas import AnalyzeResponse, ClauseResult, EvidenceSpan


def _get_openai() -> Any:
    from openai import OpenAI
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


# 요청 하나가 조항을 순차 처리하는 동안 to_thread 워커 하나를 계속 점유한다
# (동시에 여러 개를 쓰지 않음). 동시 요청 수를 안 막으면 기본 스레드풀
# (min(32, cpu+4))이 고갈되거나 OpenAI rate limit에 한꺼번에 부딪힐 수 있어,
# 요청 단위로 동시 처리 개수를 제한한다. 큐가 너무 길어지면 무한정 기다리게
# 두지 않고 503으로 명확히 알린다.
_MAX_CONCURRENT_ANALYSES = int(os.environ.get("MAX_CONCURRENT_ANALYSES", "4"))
_QUEUE_TIMEOUT_SECONDS = 30
_analyze_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_ANALYSES)


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


def _process_clause(client: Any, clause: str, index: int) -> ClauseResult | None:
    """단일 조항을 LangGraph 파이프라인(Analysis→Retrieval Strategy→Judgment)으로 분석한다.

    domain이 "해당없음"이면 그래프가 검색·판단 단계를 건너뛰고 바로 끝나므로,
    반환된 상태에도 domain만 있고 나머지 필드는 비어 있다 — 이 경우 None을 반환한다.
    """
    graph = get_graph()
    result = graph.invoke({"clause": clause}, config={"configurable": {"client": client}})

    domain = result.get("domain", "해당없음")
    if domain == "해당없음":
        return None

    evidence_span = result.get("evidence_span", "")
    verified = result.get("verified", False)

    return ClauseResult(
        id                = index + 1,
        original          = clause,
        domain            = domain,
        risk_level        = result.get("risk_level", "Low"),
        confidence        = 1.0 if verified else 0.7,
        evidence_spans    = _extract_spans(clause, evidence_span),
        legal_basis       = result.get("legal_basis", []),
        reasoning         = result.get("reasoning", ""),
        verified          = verified,
        redteam_note      = result.get("redteam_note", ""),
        evidence_verified = result.get("evidence_verified", True),
    )


async def run_analyze(text: str) -> AnalyzeResponse:
    clauses = split_clauses(text)
    if not clauses:
        raise HTTPException(status_code=400, detail="조항을 분리할 수 없습니다.")

    try:
        await asyncio.wait_for(_analyze_semaphore.acquire(), timeout=_QUEUE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="현재 처리 중인 분석 요청이 많습니다. 잠시 후 다시 시도해주세요.",
        )

    try:
        client  = _get_openai()
        results: list[ClauseResult] = []

        for i, clause in enumerate(clauses):
            # graph.invoke는 동기 호출(OpenAI/DB 왕복 포함)이라 그대로 부르면
            # 이 async 핸들러가 이벤트 루프를 막아 다른 요청을 전부 지연시킨다.
            # 스레드로 넘겨 이벤트 루프는 그동안 다른 요청을 계속 처리하게 한다.
            result = await asyncio.to_thread(_process_clause, client, clause, i)
            if result is not None:
                results.append(result)
    finally:
        _analyze_semaphore.release()

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
