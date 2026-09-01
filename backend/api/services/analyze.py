# backend/api/services/analyze.py
import asyncio
import difflib
import os
import re
from typing import Any

from fastapi import HTTPException

from backend.agents.graph import get_graph
from backend.agents.judgment_agent import model_version
from backend.api.schemas import AnalyzeResponse, ClauseResult, EvidenceSpan, OutOfScopeClause

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

# 조항 단위 동시 처리 한도. **요청별이 아니라 모듈 전역**이어야 한다 —
# 요청마다 세마포어를 만들면 `요청 4개 × 조항 30개 = 동시 호출 120`으로 곱해져
# OpenAI rate limit(429)에 그대로 부딪힌다. 전역이면 두 세마포어가 곱해지지 않고
# 조항 호출 총량이 이 값으로 상한된다.
#
# 조항 하나가 forward·verify·red-team 등 여러 번 호출하고 조항당 약 2,000 토큰을
# 쓰므로, TPM 한도에서 역산해 보수적으로 잡는다. 개별 호출의 429 백오프는
# `forward_labeling.run_forward`/`consistency_verification.run_verify`가 이미 한다.
_MAX_CONCURRENT_CLAUSES = int(os.environ.get("MAX_CONCURRENT_CLAUSES", "6"))
_clause_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_CLAUSES)


async def _process_clause_async(client: Any, clause: str, index: int):
    """조항 하나를 스레드로 넘겨 처리하되, 전역 세마포어로 동시 호출 수를 묶는다."""
    async with _clause_semaphore:
        return await asyncio.to_thread(_process_clause, client, clause, index)


def _tally(outcomes: list) -> tuple[list[ClauseResult], list[OutOfScopeClause]]:
    analyzed = [o for o in outcomes if isinstance(o, ClauseResult)]
    skipped  = [o for o in outcomes if isinstance(o, OutOfScopeClause)]
    return analyzed, skipped


# 예전에는 두 값이 모두 리터럴 `20`이라 뜻이 섞여 있었다 — 하나는 "조각의 최소 글자수",
# 하나는 "분석할 조항 개수 상한"이다.
_MIN_CLAUSE_CHARS = 20

# 조항 하나당 OpenAI 호출이 최소 1회라 상한 자체는 있어야 한다(긴 문서 붙여넣기 =
# 비용·DoS 노출). 다만 예전 상한 20은 실제 계약서(30조항대)를 조용히 잘라냈다 —
# 벤치마크에서 입력 20과 30의 소요 시간이 같았던 이유가 이것이다. 상한은 올리되
# **넘긴 사실을 응답에 반드시 남긴다**(`truncated_clauses`).
_MAX_CLAUSES = int(os.environ.get("MAX_CLAUSES", "60"))


def split_clauses(text: str) -> tuple[list[str], int]:
    """(분석할 조항, 상한 초과로 잘라낸 개수)를 반환한다."""
    parts = re.split(
        r"(?=제\s*\d+\s*조|^\s*\d+\.\s|^[①②③④⑤⑥⑦⑧⑨⑩]|\n{2,})",
        text,
        flags=re.MULTILINE,
    )
    found = [s.strip() for s in parts if len(s.strip()) > _MIN_CLAUSE_CHARS]
    return found[:_MAX_CLAUSES], max(0, len(found) - _MAX_CLAUSES)


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


# 전달할 한계가 **둘**이다 — 범위(제6~14조만 본다)와 재현율(그 안에서도 놓친다).
# 첫 초안은 후자만 담았고 전자는 조 이름으로 암시만 됐다. 조 taxonomy로 바뀌면서
# 경고가 약해지는 게 아니라 **강해져야** 한다 — 예전에는 "범위 밖이라 모른다"였는데
# 이제는 "범위 안인데 못 찾았을 수 있다"이기 때문이다(조항 단위 재현 78.0%).
_OUT_OF_SCOPE_REASON = (
    "약관규제법 제6~14조 기준으로는 위반이 확인되지 않았습니다. "
    "다른 법령은 검토하지 않으며, 이 범위 안에서도 일부만 찾아냅니다."
)


def _process_clause(client: Any, clause: str, index: int) -> ClauseResult | OutOfScopeClause:
    """단일 조항을 LangGraph 파이프라인(Analysis→Judgment→근거/반박 브랜치)으로 분석한다.

    **게이트는 모델이다** — `model_articles`가 비면(조를 하나도 지목하지 않으면) 그래프가
    근거·반박 브랜치를 건너뛰고, 여기서 `OutOfScopeClause`로 돌린다. 예전에는 GPT의
    2-도메인 값(`domain == "해당없음"`)으로 끊었다.

    빠진 조항에 **어떤 등급도 붙이지 않는다.** 예전에는 None을 반환하고 호출부가 그대로
    버려서 **조항이 응답에서 통째로 사라졌다**(벤치마크에서 입력 20 → 결과 10건) — 사용자는
    나머지가 안전하다고 오해한다. 지금은 목록에 남기되 "확인되지 않았다"로만 표시한다.
    조 단위 재현이 78%이므로 **약 5건 중 1건은 여기 잘못 들어와 있다**(`schemas.py` 참고).
    """
    graph = get_graph()
    result = graph.invoke({"clause": clause}, config={"configurable": {"client": client}})

    # **게이트가 바뀌었다 (2026-08-31).** 예전에는 GPT가 낸 2-도메인 값이 "해당없음"이면
    # 버렸다. 지금은 **분류 모델이 조를 하나라도 지목했는가**로 가른다 — 판단 주체가
    # GPT에서 모델로 옮겨간 것이고, taxonomy도 9개 조로 늘었다.
    model_articles = result.get("model_articles") or []
    if not model_articles:
        return OutOfScopeClause(id=index + 1, original=clause, reason=_OUT_OF_SCOPE_REASON)

    evidence_span = result.get("evidence_span", "")

    return ClauseResult(
        id                = index + 1,
        original          = clause,
        articles          = model_articles,
        needs_review      = True,
        domain            = result.get("domain", ""),      # 과거 결과 호환용 파생값
        evidence_spans    = _extract_spans(clause, evidence_span),
        legal_basis       = result.get("legal_basis", []),
        precedent_refs    = result.get("precedent_refs", []),
        reasoning         = result.get("reasoning", ""),
        verified          = bool(result.get("verified", False)),
        redteam_note      = result.get("redteam_note", ""),
        evidence_verified = result.get("evidence_verified", True),
    )


async def run_analyze(text: str) -> AnalyzeResponse:
    clauses, truncated = split_clauses(text)
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
        client = _get_openai()
        # 예전에는 `for` 루프에서 매번 await 해 조항을 **직렬** 처리했다 — 비동기 구조를
        # 갖춰놓고 쓰지 않은 셈이라, 조항당 약 10초가 그대로 누적됐다(30조항 = 5분).
        # 시간의 대부분이 OpenAI 왕복 대기라 병렬화 이득이 크다. 동시 호출 수는
        # `_clause_semaphore`(전역)가 묶으므로 여기서 한 번에 던져도 안전하다.
        outcomes = await asyncio.gather(
            *(_process_clause_async(client, c, i) for i, c in enumerate(clauses))
        )
    finally:
        _analyze_semaphore.release()

    results, skipped = _tally(list(outcomes))
    results.sort(key=lambda r: r.id)      # gather는 순서를 보존하지만 명시해 둔다
    skipped.sort(key=lambda r: r.id)
    return AnalyzeResponse(
        total_clauses     = len(results),
        review_count      = len(results),
        clauses           = results,
        input_clauses     = len(clauses) + truncated,
        truncated_clauses = truncated,
        out_of_scope      = skipped,
        model_version     = model_version(),
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
    clauses, truncated = split_clauses(text)
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
            total = len(clauses)
            outcomes: list = []

            # 조항을 병렬로 던지고 **끝나는 순서대로** 진행률을 흘린다.
            # 순차 처리 시절에는 진행률 index가 곧 조항 번호였지만, 병렬에서는
            # 완료 순서가 뒤섞이므로 진행률은 "몇 개 끝났는지"(done/total)로 센다.
            async def _one(i: int, c: str):
                return i, await _process_clause_async(client, c, i)

            tasks = [asyncio.create_task(_one(i, c)) for i, c in enumerate(clauses)]
            done_n = 0
            for fut in asyncio.as_completed(tasks):
                i, outcome = await fut
                outcomes.append(outcome)
                done_n += 1
                if isinstance(outcome, ClauseResult):
                    yield {
                        "type": "progress", "index": done_n, "total": total,
                        "clause_no": i + 1,
                        "articles": outcome.articles, "needs_review": True,
                    }
                else:
                    yield {"type": "progress", "index": done_n, "total": total,
                           "clause_no": i + 1, "skipped": True}

            results, skipped = _tally(outcomes)
            results.sort(key=lambda r: r.id)      # 완료 순서가 아니라 조항 순서로 돌려준다
            skipped.sort(key=lambda r: r.id)
            final = AnalyzeResponse(
                total_clauses     = len(results),
                review_count      = len(results),
                clauses           = results,
                input_clauses     = total + truncated,
                truncated_clauses = truncated,
                out_of_scope      = skipped,
                model_version     = model_version(),
            )
            yield {"type": "done", "result": final.model_dump()}
        finally:
            _analyze_semaphore.release()

    return _events()
