# backend/eval/lightrag_compare_final.py
"""
법령 코퍼스(3,323청크) 전체 인덱싱이 완료된 뒤 돌리는 최종 비교 — 민법 포함
전체 법령 대상, 재인덱싱 없이 기존 완성된 인덱스(data/eval/lightrag_full_storage)를
그대로 연다. lightrag_compare_partial.py(약관규제법 등 94.7%만 커버)의 완전판.

실행: python -m backend.eval.lightrag_compare_final
"""

import asyncio

from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc

from backend.eval.lightrag_compare import (
    _N_QUERIES,
    _QUERY_LOG_EVERY,
    LAWS_PATH,
    WORKING_DIR,
    build_ground_truth,
    hybrid_rrf_hit,
    lightrag_hit,
)
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("lightrag_compare_final.log")
OUT_PATH = PROJECT_ROOT / "data/eval/lightrag_vs_rrf_report_final.json"


async def main() -> None:
    all_recs = load_jsonl(LAWS_PATH)
    logger.info(f"  법령 전체 인덱싱 완료 상태 재사용({len(all_recs)}청크, 민법 포함)")

    queries = build_ground_truth(all_recs, n_cases=_N_QUERIES)
    logger.info(f"  평가 쿼리 {len(queries)}건 생성(전체 법령 대상, 민법 포함)")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=EmbeddingFunc(embedding_dim=1536, max_token_size=8192, func=openai_embed),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

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
        "n_law_chunks_indexed": len(all_recs),
        "scope_note": "법령 전체(민법 포함) 완전 인덱싱 후 평가 — 부분 인덱스 한계 해소됨",
        "rrf_hit_rate": rrf_hits / n,
        "lightrag_hit_rate": lightrag_hits / n,
        "rrf_hits": rrf_hits,
        "lightrag_hits": lightrag_hits,
        "per_query": per_query,
    }
    save_json(result, OUT_PATH)
    logger.info(f"========== 최종 결과 (n={n}, 법령 전체) ==========")
    logger.info(f"  Hybrid RRF 적중률: {rrf_hits}/{n} ({rrf_hits/n*100:.1f}%)")
    logger.info(f"  LightRAG 적중률:   {lightrag_hits}/{n} ({lightrag_hits/n*100:.1f}%)")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
