# backend/eval/lightrag_pilot.py
"""
LightRAG vs 지금 Hybrid RRF(backend.api.services.retrieval) — 법조문 검색 적중률 비교
파일럿. 원래 20.1% 실험은 스크립트가 안 남아있어 재현 불가능했으므로, FTC 케이스의
근거_법령 필드(공정위가 실제로 인용한 법조문)를 새 ground truth로 쓴다 — 981개
케이스, 2,444개 인용이 법령 코퍼스(data/processed/laws.jsonl)의 (law_name, article_no)
메타데이터와 100% 파싱·매칭 확인됨.

파일럿 범위: 전체 코퍼스(3,323청크)를 인덱싱하기 전에, 가장 많이 인용되는 법령
(약관의 규제에 관한 법률, 43청크)만 먼저 인덱싱해서 파이프라인(삽입→쿼리→
only_need_context로 원문 추출→적중 판정)이 실제로 도는지 확인한다. 비용·시간이
가장 큰 작업이라 전체(3,323청크)로 바로 가지 않고 여기서 먼저 검증한다.

실행: python -m backend.eval.lightrag_pilot
"""

import asyncio
import json
import os
import re
import shutil

from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc

from backend.utils import PROJECT_ROOT, load_jsonl, load_logger

logger = load_logger("lightrag_pilot.log")

WORKING_DIR = str(PROJECT_ROOT / "data/eval/lightrag_pilot_storage")
LAWS_PATH = PROJECT_ROOT / "data/processed/laws.jsonl"
FTC_PATH = PROJECT_ROOT / "data/raw/ftc_cases/ftc_cases_parsed.json"

_PAT = re.compile(r"^(.*?)\s*제\s*(\d+)\s*조")
_PILOT_LAW_NAME = "약관의 규제에 관한 법률"


def build_ground_truth(n_cases: int = 15) -> list[dict]:
    """FTC 근거_법령을 파싱해 (조항 텍스트 -> 정답 조문 집합) 평가 쿼리를 만든다."""
    with open(FTC_PATH, encoding="utf-8") as f:
        cases = json.load(f).get("사례", [])

    queries = []
    for case in cases:
        grounds = case.get("근거_법령", [])
        clauses = case.get("조항_원문", [])
        if not grounds or not clauses:
            continue
        pairs = set()
        for g in grounds:
            m = _PAT.match(g.replace("\n", " ").strip())
            if m and m.group(1).strip() == _PILOT_LAW_NAME:
                pairs.add(m.group(2))  # article_no만 저장(법령명은 파일럿 내내 고정)
        if not pairs:
            continue
        queries.append({"case_name": case.get("사건명", ""), "clause": str(clauses[0])[:300], "correct_articles": sorted(pairs)})
        if len(queries) >= n_cases:
            break
    return queries


async def build_index() -> LightRAG:
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)  # 파일럿은 매번 깨끗한 상태로 재현
    os.makedirs(WORKING_DIR)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=EmbeddingFunc(embedding_dim=1536, max_token_size=8192, func=openai_embed),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    law_recs = [r for r in load_jsonl(LAWS_PATH) if r["metadata"]["law_name"] == _PILOT_LAW_NAME]
    logger.info(f"  인덱싱 대상: {_PILOT_LAW_NAME} {len(law_recs)}청크")

    ids = [r["chunk_id"] for r in law_recs]
    texts = [r["text"] for r in law_recs]
    await rag.ainsert(texts, ids=ids)
    logger.info("  인덱싱 완료")
    return rag


def hybrid_rrf_hit(query_text: str, correct_articles: list[str]) -> bool:
    """지금 프로덕션이 쓰는 Hybrid RRF 방식으로 같은 질의를 검색해 적중 여부 확인."""
    from backend.api.services.retrieval import fetch_candidates

    candidates = fetch_candidates(query_text, top_k_per_source=5, sparse_similarity_threshold=0.10, unified=False)
    law_candidates = candidates.get("law", [])
    found_articles = {c["metadata"].get("article_no") for c in law_candidates if c["metadata"].get("law_name") == _PILOT_LAW_NAME}
    return bool(found_articles & set(correct_articles))


async def lightrag_hit(rag: LightRAG, query_text: str, correct_articles: list[str]) -> bool:
    from lightrag import QueryParam

    result = await rag.aquery(query_text, param=QueryParam(mode="hybrid", only_need_context=True, top_k=5))
    context = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    return any(f"제{a}조" in context for a in correct_articles)


async def main() -> None:
    queries = build_ground_truth(n_cases=15)
    logger.info(f"  평가 쿼리 {len(queries)}건 생성(대상 법령: {_PILOT_LAW_NAME})")
    for q in queries:
        logger.info(f"    - {q['case_name'][:30]} | 정답 조문: {q['correct_articles']}")

    rag = await build_index()

    rrf_hits = 0
    lightrag_hits = 0
    for q in queries:
        r_hit = hybrid_rrf_hit(q["clause"], q["correct_articles"])
        l_hit = await lightrag_hit(rag, q["clause"], q["correct_articles"])
        rrf_hits += r_hit
        lightrag_hits += l_hit
        logger.info(f"  [{q['case_name'][:25]}] RRF={'O' if r_hit else 'X'} | LightRAG={'O' if l_hit else 'X'} | 정답={q['correct_articles']}")

    n = len(queries)
    logger.info(f"========== 파일럿 결과 (n={n}) ==========")
    logger.info(f"  Hybrid RRF 적중률: {rrf_hits}/{n} ({rrf_hits/n*100:.1f}%)")
    logger.info(f"  LightRAG 적중률:   {lightrag_hits}/{n} ({lightrag_hits/n*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
