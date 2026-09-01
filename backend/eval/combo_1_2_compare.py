# backend/eval/combo_1_2_compare.py
"""
옵션1(RRF + Cross-Encoder 재랭킹)과 옵션2(도메인 필터링 Hybrid)를 합친 조합 —
둘 다 그래프 없이 동작하고 서로 다른 단계(후보 생성 vs 정밀 재정렬)에 작용해서
시너지가 날 수 있다는 가설.

파이프라인: law_name 파티션별로 dense+sparse 후보를 확보(옵션2 방식 — 소수 파티션도
후보 풀에서 밀리지 않게 보장) → 파티션 경계 없이 합쳐 넉넉한 후보 풀(top-20)을 만듦
→ Cross-Encoder(BAAI/bge-reranker-v2-m3, 옵션1과 동일 모델)로 재정렬해 최종 top-5.

동일 ground truth(FTC 근거_법령, 법령 전체 3,323청크, seed=42, 100건) — rerank_compare.py,
domain_filter_compare.py와 같은 쿼리 세트라 4개 리포트를 직접 비교할 수 있다.

실행: .venv/bin/python -m backend.eval.combo_1_2_compare
"""

from typing import Any

from sentence_transformers import CrossEncoder

from backend.api.services.retrieval import _get_cached_embedder
from backend.db.connection import get_conn
from backend.db.loader import embed_texts
from backend.eval.domain_filter_compare import _dense_with_score, _law_names, _sparse_with_score, rrf_baseline_hit
from backend.eval.lightrag_compare import _N_QUERIES, LAWS_PATH, build_ground_truth
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("combo_1_2_compare.log")
OUT_PATH = PROJECT_ROOT / "data/eval/combo_1_2_vs_rrf_report.json"

_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_CANDIDATE_K_PER_PARTITION = 5  # 파티션이 16개나 되므로 옵션2보다 파티션당은 적게, 대신 전부 모음
_POOL_CAP = 40  # 재랭킹 전 후보 풀 상한(파티션이 16개라 무제한 합치면 너무 커짐)
_TOP_K = 5


def combo_hit(cur: Any, embedder: Any, reranker: CrossEncoder, query_text: str, law_names: list[str], correct_pairs: list[tuple]) -> bool:
    query_vec = embed_texts(embedder, [query_text], prefix="query: ")[0]
    vec_literal = "[" + ",".join(repr(x) for x in query_vec) + "]"

    pool: dict[str, dict] = {}
    for law_name in law_names:
        for row in _dense_with_score(cur, law_name, vec_literal, _CANDIDATE_K_PER_PARTITION):
            chunk_id, source, metadata, text, score = row
            pool[chunk_id] = {"chunk_id": chunk_id, "metadata": metadata, "text": text, "score": score}
        for row in _sparse_with_score(cur, law_name, query_text, _CANDIDATE_K_PER_PARTITION):
            chunk_id, source, metadata, text, score = row
            pool.setdefault(chunk_id, {"chunk_id": chunk_id, "metadata": metadata, "text": text, "score": score})

    candidates = sorted(pool.values(), key=lambda c: c["score"], reverse=True)[:_POOL_CAP]
    if not candidates:
        return False

    scores = reranker.predict([(query_text, c["text"]) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:_TOP_K]
    found = {(c["metadata"].get("law_name"), c["metadata"].get("article_no")) for c, _ in ranked}
    return bool(found & set(map(tuple, correct_pairs)))


def main() -> None:
    law_recs = load_jsonl(LAWS_PATH)
    queries = build_ground_truth(law_recs, n_cases=_N_QUERIES)
    logger.info(f"  평가 쿼리 {len(queries)}건 생성(법령 전체 대상, 동일 seed/로직)")

    logger.info(f"  Cross-Encoder 로드: {_RERANKER_MODEL}")
    reranker = CrossEncoder(_RERANKER_MODEL, max_length=512)

    embedder = _get_cached_embedder()
    with get_conn() as conn, conn.cursor() as cur:
        law_names = _law_names(cur)

        rrf_hits = combo_hits = 0
        per_query = []
        for i, q in enumerate(queries):
            r_hit = rrf_baseline_hit(cur, embedder, q["clause"], q["correct_pairs"])
            c_hit = combo_hit(cur, embedder, reranker, q["clause"], law_names, q["correct_pairs"])
            rrf_hits += r_hit
            combo_hits += c_hit
            per_query.append({"case_name": q["case_name"], "correct_pairs": q["correct_pairs"], "rrf_hit": r_hit, "combo_hit": c_hit})
            if (i + 1) % 25 == 0:
                logger.info(f"  평가 진행: {i + 1}/{len(queries)} | 누적 RRF={rrf_hits} Combo(1+2)={combo_hits}")

    n = len(queries)
    result = {
        "n_queries": n,
        "reranker_model": _RERANKER_MODEL,
        "rrf_hit_rate": rrf_hits / n,
        "combo_hit_rate": combo_hits / n,
        "rrf_hits": rrf_hits,
        "combo_hits": combo_hits,
        "per_query": per_query,
        "note": "도메인 파티션별 후보 확보(옵션2) + Cross-Encoder 재랭킹(옵션1) 조합. 그래프 없음.",
    }
    save_json(result, OUT_PATH)
    logger.info(f"========== 최종 결과 (n={n}) ==========")
    logger.info(f"  RRF(파티션 없음):   {rrf_hits}/{n} ({rrf_hits/n*100:.1f}%)")
    logger.info(f"  Combo(1+2):         {combo_hits}/{n} ({combo_hits/n*100:.1f}%)")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
