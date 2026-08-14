# backend/eval/lightrag_compare.py
"""
LightRAG vs 지금 Hybrid RRF(backend.api.services.retrieval) — 법조문 검색 적중률
본비교. lightrag_pilot.py로 파이프라인 동작은 검증됨(RRF 20.0% ≈ 원래 문서 20.1%,
방법론 신뢰성 확인) — 단 파일럿은 LightRAG에 약관규제법 43청크만 넣고 RRF는 법령
전체 3,323청크에서 찾게 해서 불공정 비교였다(LightRAG 93.3% vs RRF 20.0%, 코퍼스
크기 차이 때문). 이번엔 **둘 다 법령 전체 3,323청크**를 검색 대상으로 맞춘다.

정답 판정 방식: LightRAG 쿼리 응답(only_need_context=True)엔 reference_id가
비어있어(file_paths 미지정) 청크가 어느 법령 소속인지 텍스트만으론 구분 안 됨
(예: 민법 제7조 vs 약관규제법 제7조 조문번호가 겹칠 수 있음). 그래서 조문번호
대신 **정답 조문의 실제 본문 텍스트**(법령마다 문구가 고유함)가 컨텍스트에 부분
문자열로 등장하는지로 판정한다 — 법령명 구분 문제가 자동으로 해결됨.

실행: python -m backend.eval.lightrag_compare
"""

import asyncio
import json
import os
import random
import re
import shutil

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc

from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("lightrag_compare.log")

WORKING_DIR = str(PROJECT_ROOT / "data/eval/lightrag_full_storage")
LAWS_PATH = PROJECT_ROOT / "data/processed/laws.jsonl"
FTC_PATH = PROJECT_ROOT / "data/raw/ftc_cases/ftc_cases_parsed.json"
OUT_PATH = PROJECT_ROOT / "data/eval/lightrag_vs_rrf_report.json"

_PAT = re.compile(r"^(.*?)\s*제\s*(\d+)\s*조")
_N_QUERIES = 100
_INDEX_LOG_EVERY = 250
_QUERY_LOG_EVERY = 50


def build_ground_truth(law_recs: list[dict], n_cases: int, seed: int = 42) -> list[dict]:
    """FTC 근거_법령 전체(3개 법령)를 파싱해 평가 쿼리를 만든다. 정답 조문의 실제 본문도 같이 붙인다."""
    article_text = {(r["metadata"]["law_name"], r["metadata"]["article_no"]): r["text"] for r in law_recs}

    with open(FTC_PATH, encoding="utf-8") as f:
        cases = json.load(f).get("사례", [])

    candidates = []
    for case in cases:
        grounds = case.get("근거_법령", [])
        clauses = case.get("조항_원문", [])
        if not grounds or not clauses:
            continue
        pairs = set()
        for g in grounds:
            m = _PAT.match(g.replace("\n", " ").strip())
            if m:
                pairs.add((m.group(1).strip(), m.group(2)))
        pairs = {p for p in pairs if p in article_text}  # 코퍼스에 실제 있는 것만
        if not pairs:
            continue
        candidates.append({
            "case_name": case.get("사건명", ""),
            "clause": str(clauses[0])[:300],
            "correct_pairs": sorted(pairs),
            "correct_texts": [article_text[p][:60] for p in pairs],  # 판정용 — 조문 본문 앞부분
        })

    random.seed(seed)
    return random.sample(candidates, min(n_cases, len(candidates)))


async def build_index(law_recs: list[dict]) -> LightRAG:
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=EmbeddingFunc(embedding_dim=1536, max_token_size=8192, func=openai_embed),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    logger.info(f"  인덱싱 대상: 법령 전체 {len(law_recs)}청크")
    ids = [r["chunk_id"] for r in law_recs]
    texts = [r["text"] for r in law_recs]

    # ainsert는 한 번에 통째로 넣으면 내부적으로 순차 처리되며 로그를 자체적으로 남긴다.
    # 진행률만 별도로 남기기 위해 배치로 쪼개 호출한다.
    batch = _INDEX_LOG_EVERY
    for i in range(0, len(texts), batch):
        await rag.ainsert(texts[i:i + batch], ids=ids[i:i + batch])
        logger.info(f"    인덱싱 진행: {min(i + batch, len(texts))}/{len(texts)}")

    logger.info("  인덱싱 완료")
    return rag


def hybrid_rrf_hit(query_text: str, correct_pairs: list[tuple]) -> bool:
    from backend.api.services.retrieval import fetch_candidates

    candidates = fetch_candidates(query_text, top_k_per_source=5, sparse_similarity_threshold=0.10, unified=False)
    found = {(c["metadata"].get("law_name"), c["metadata"].get("article_no")) for c in candidates.get("law", [])}
    return bool(found & set(map(tuple, correct_pairs)))


async def lightrag_hit(rag: LightRAG, query_text: str, correct_texts: list[str]) -> bool:
    result = await rag.aquery(query_text, param=QueryParam(mode="hybrid", only_need_context=True, top_k=5))
    context = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    return any(t in context for t in correct_texts)


async def main() -> None:
    law_recs = load_jsonl(LAWS_PATH)
    queries = build_ground_truth(law_recs, n_cases=_N_QUERIES)
    logger.info(f"  평가 쿼리 {len(queries)}건 생성(법령 전체 대상, 샘플링)")

    rag = await build_index(law_recs)

    rrf_hits = lightrag_hits = 0
    per_query = []
    for i, q in enumerate(queries):
        r_hit = hybrid_rrf_hit(q["clause"], q["correct_pairs"])
        l_hit = await lightrag_hit(rag, q["clause"], q["correct_texts"])
        rrf_hits += r_hit
        lightrag_hits += l_hit
        per_query.append({"case_name": q["case_name"], "correct_pairs": q["correct_pairs"], "rrf_hit": r_hit, "lightrag_hit": l_hit})
        if (i + 1) % _QUERY_LOG_EVERY == 0:
            logger.info(f"  평가 진행: {i + 1}/{len(queries)} | 누적 RRF={rrf_hits} LightRAG={lightrag_hits}")

    n = len(queries)
    result = {
        "n_queries": n,
        "n_law_chunks_indexed": len(law_recs),
        "rrf_hit_rate": rrf_hits / n,
        "lightrag_hit_rate": lightrag_hits / n,
        "rrf_hits": rrf_hits,
        "lightrag_hits": lightrag_hits,
        "per_query": per_query,
        "note": (
            "정답 판정은 조문번호가 아니라 정답 조문의 실제 본문 텍스트(앞 60자)가 "
            "결과 컨텍스트에 부분 문자열로 등장하는지로 확인 — 법령명 다른데 조문번호만 "
            "같은 경우(예: 민법 제7조 vs 약관규제법 제7조) 오탐 방지."
        ),
    }
    save_json(result, OUT_PATH)
    logger.info(f"========== 최종 결과 (n={n}) ==========")
    logger.info(f"  Hybrid RRF 적중률: {rrf_hits}/{n} ({rrf_hits/n*100:.1f}%)")
    logger.info(f"  LightRAG 적중률:   {lightrag_hits}/{n} ({lightrag_hits/n*100:.1f}%)")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
