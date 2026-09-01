# backend/eval/lightrag_compare_partial.py
"""
lightrag_compare.py의 본작업이 잔액 부족(인덱싱 도중 $0.82로 하락) 위험으로 중단됨 —
법령 3,323청크 중 1,500청크(45%)까지 인덱싱된 상태에서 멈췄다. 다행히 그 1,500청크
안에 **약관규제법 43청크 전부**가 포함돼 있고, 이게 FTC 근거_법령 인용의 94.7%
(2,314/2,444건)를 차지한다 — 민법(5.3%)만 빠진 상태.

재인덱싱 없이(추가 비용 0원) 기존 인덱스(`data/eval/lightrag_full_storage`)를 그대로
열어서, 정답이 이미 인덱싱된 법령(민법 제외) 안에 있는 케이스만으로 평가한다.

실행: python -m backend.eval.lightrag_compare_partial
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

logger = load_logger("lightrag_compare_partial.log")
OUT_PATH = PROJECT_ROOT / "data/eval/lightrag_vs_rrf_report_agb_only.json"


async def main() -> None:
    all_recs = load_jsonl(LAWS_PATH)
    indexed_recs = all_recs[:1500]  # 실제로 인덱싱된 부분(로그로 확인됨)
    logger.info(f"  기존 인덱스 재사용(재인덱싱 없음) — 인덱싱된 청크: {len(indexed_recs)}/{len(all_recs)}")

    # 정답이 인덱싱된 법령 안에 있는 케이스만 남긴다(build_ground_truth가 article_text에
    # 없는 pair는 자동 제외 — indexed_recs만 넘기면 민법 인용 케이스가 자연히 걸러짐)
    queries = build_ground_truth(indexed_recs, n_cases=_N_QUERIES)
    logger.info(f"  평가 쿼리 {len(queries)}건 생성(약관규제법 등 이미 인덱싱된 법령만 대상)")

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
        "n_law_chunks_indexed": len(indexed_recs),
        "n_law_chunks_total": len(all_recs),
        "scope_note": "예산 제약으로 법령 전체(3,323청크) 대신 1,500청크(약관규제법 43개 전부 포함, 정답의 94.7% 커버)만 인덱싱 — 민법 인용 케이스는 평가에서 제외됨",
        "rrf_hit_rate": rrf_hits / n,
        "lightrag_hit_rate": lightrag_hits / n,
        "rrf_hits": rrf_hits,
        "lightrag_hits": lightrag_hits,
        "per_query": per_query,
    }
    save_json(result, OUT_PATH)
    logger.info(f"========== 결과 (n={n}, 약관규제법 등 부분 인덱스 기준) ==========")
    logger.info(f"  Hybrid RRF 적중률: {rrf_hits}/{n} ({rrf_hits/n*100:.1f}%)")
    logger.info(f"  LightRAG 적중률:   {lightrag_hits}/{n} ({lightrag_hits/n*100:.1f}%)")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
