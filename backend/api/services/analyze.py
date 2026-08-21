# backend/api/services/analyze.py
import asyncio
import difflib
import os
import re
from typing import Any

from fastapi import HTTPException

from backend.agents.graph import get_graph
from backend.api.schemas import AnalyzeResponse, ClauseResult, EvidenceSpan
# 근거 문구 매칭 기준을 FB-Check와 공유한다 — 검증 파이프라인이 통과시킨 근거를
# 서빙이 버리면 화면에서 하이라이트가 조용히 사라진다(_extract_spans docstring 참고).
from backend.fb_check.backward_grounding import _FUZZY_MATCH_THRESHOLD, _PAGE_MARKER


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


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """공백을 하나로 접은 문자열과, 그 각 문자가 원문 몇 번째였는지의 대응표를 만든다.

    하이라이트는 **원문 기준 offset**이 필요하므로 정규화만 하면 안 되고 되돌릴 수
    있어야 한다. 페이지 마커(`- 12 -`)는 공백으로 바꿔 없앤다 — PDF 추출물에 섞여
    들어와 근거 문구를 끊어놓는 주범이다.
    """
    text = _PAGE_MARKER.sub(lambda m: " " * (m.end() - m.start()), text)  # 길이를 보존해야 대응표가 안 깨진다
    out: list[str] = []
    idx_map: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if out and not prev_space:
                out.append(" ")
                idx_map.append(i)
            prev_space = True
        else:
            out.append(ch)
            idx_map.append(i)
            prev_space = False
    return "".join(out), idx_map


def _span_from_norm(idx_map: list[int], start: int, length: int) -> tuple[int, int]:
    """정규화 문자열의 [start, start+length) 구간을 원문 offset으로 되돌린다."""
    return idx_map[start], idx_map[start + length - 1] + 1


def _extract_spans(clause_text: str, evidence_span: str) -> list[EvidenceSpan]:
    """근거 문구의 원문 위치를 찾는다. 완전 일치 → 공백 정규화 → 퍼지 순으로 시도한다.

    이전 구현은 `clause_text.find(evidence_span)` 완전 일치 하나뿐이라, 실패하면 빈
    배열을 조용히 돌려줬다 — 화면에서 근거 하이라이트가 **에러 없이 사라진다.**
    `clean.jsonl`(FB-Check가 CLEAN으로 통과시킨 694건)로 재보니 완전 일치 실패가
    **210건(30.3%)**이었다. 즉 세 조항 중 하나는 "왜 위험한지"가 표시되지 않았다.

    FB-Check의 `backward_grounding.snippet_exists()`는 같은 상황을 이미 퍼지 매칭으로
    받아들이고 있었다(PDF 표 추출 시 공백·순서가 깨져도 문구 자체는 실존). 검증
    파이프라인이 통과시킨 근거를 서빙이 버리는 불일치였으므로, 같은 기준
    (`_FUZZY_MATCH_THRESHOLD=0.85`)을 쓴다.

    같은 문구가 여러 번 나오면 **전부** 반환한다 — 반환 타입이 리스트인데 이전엔
    항상 최대 1개만 담겼다.
    """
    if not evidence_span or not evidence_span.strip():
        return []

    # 1) 완전 일치 — 모든 출현 위치
    spans = [
        EvidenceSpan(text=evidence_span, start=m.start(), end=m.end())
        for m in re.finditer(re.escape(evidence_span), clause_text)
    ]
    if spans:
        return spans

    norm_text, idx_map = _normalize_with_map(clause_text)
    norm_span = " ".join(evidence_span.split())
    if not norm_span or not norm_text:
        return []

    # 2) 공백만 다른 경우
    hits = [m.start() for m in re.finditer(re.escape(norm_span), norm_text)]
    if hits:
        out = []
        for h in hits:
            s, e = _span_from_norm(idx_map, h, len(norm_span))
            out.append(EvidenceSpan(text=clause_text[s:e], start=s, end=e))
        return out

    # 3) 퍼지 — 최장 공통 부분열이 근거 문구의 85% 이상이면 그 구간을 하이라이트한다
    matcher = difflib.SequenceMatcher(None, norm_span, norm_text, autojunk=False)
    match = matcher.find_longest_match(0, len(norm_span), 0, len(norm_text))
    if match.size and match.size / len(norm_span) >= _FUZZY_MATCH_THRESHOLD:
        s, e = _span_from_norm(idx_map, match.b, match.size)
        return [EvidenceSpan(text=clause_text[s:e], start=s, end=e)]
    return []


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
        # 이전엔 `1.0 if verified else 0.7` 하드코딩이었다 — verified는 GPT와의 domain
        # 일치 여부이지 신뢰도가 아니었고, 정작 모델이 낸 확률은 버려지고 있었다.
        # 지금은 judgment_node가 계산한 구간을 그대로 쓴다(judgment_agent.py 참고).
        confidence_band          = result.get("confidence_band", "낮음"),
        confidence_band_accuracy = result.get("confidence_band_accuracy", 0.382),
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


async def run_analyze_stream(text: str):
    """조항이 하나씩 끝날 때마다 진행 상황을 흘려보내는 스트리밍 버전.

    프론트가 "6단계 파이프라인을 도는 척" 타이머로 꾸며낸 진행률을 보여주고
    있었는데, 실제로는 조항을 순차 처리한다는 사실 자체는 진짜다. 그래서
    가짜 애니메이션 대신 "조항 N/M 처리 완료"를 실제로 스트리밍한다 — 그래프
    내부(Analysis→Judgment/Retrieval fan-out, 최대 3회 재검색 루프)는 조항
    하나 안에서 병렬·가변 경로라 노드 단위로는 깔끔하게 스트리밍할 수 없어서,
    조항 단위를 진행률의 최소 단위로 삼는다.

    검증(빈 조항, 세마포어 타임아웃)은 이 함수가 즉시 await되는 시점에 끝내고,
    실제 스트리밍은 내부 제너레이터가 맡는다 — StreamingResponse가 시작된
    뒤에는 HTTP 상태 코드를 바꿀 수 없으므로, 실패할 수 있는 경로는 첫 바이트가
    나가기 전에 전부 해치워야 한다.
    """
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

    async def _events():
        try:
            client = _get_openai()
            results: list[ClauseResult] = []
            total = len(clauses)

            for i, clause in enumerate(clauses):
                result = await asyncio.to_thread(_process_clause, client, clause, i)
                if result is not None:
                    results.append(result)
                    yield {
                        "type": "progress",
                        "index": i + 1,
                        "total": total,
                        "risk_level": result.risk_level,
                        "domain": result.domain,
                    }
                else:
                    yield {"type": "progress", "index": i + 1, "total": total, "skipped": True}

            high   = sum(1 for r in results if r.risk_level == "High")
            medium = sum(1 for r in results if r.risk_level == "Medium")
            low    = sum(1 for r in results if r.risk_level == "Low")

            final = AnalyzeResponse(
                total_clauses = len(results),
                high_count    = high,
                medium_count  = medium,
                low_count     = low,
                clauses       = results,
            )
            yield {"type": "done", "result": final.model_dump()}
        finally:
            _analyze_semaphore.release()

    return _events()
