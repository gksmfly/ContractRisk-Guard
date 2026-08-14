# backend/eval/seal_rag_lite_compare.py
"""
옵션5(SEAL-RAG, "Replace, don't Expand")를 "1번이랑 개념 겹침"이라고 판단만 하고 넘어가지
않고 실제로 구현해서 옵션1(Cross-Encoder 전체 재정렬)과 수치로 비교한다.

옵션1은 넓은 후보 풀(top-20)을 Cross-Encoder로 통째로 재정렬한다 — RRF가 원래 매긴
순서를 완전히 버린다. SEAL-RAG 설명("고정 k 안에서 방해 요소만 교체")을 더 literal하게
구현하면: **RRF top-5는 그대로 유지하고, top-5 중 Cross-Encoder 점수가 가장 낮은 슬롯 1개만
후보 풀(top-5 밖, top-20 이내)에서 가장 점수가 높은 걸로 교체**하는 "1회 스왑"이 원 설명에
더 가깝다 — RRF의 원래 순서를 최대한 보존하면서 확실히 나쁜 것 하나만 걷어낸다는 뜻.

원 논문(정확한 출처 미확인)의 세부 알고리즘까지 재현한 것은 아니고, "고정 예산 안에서
최소 교체"라는 핵심 아이디어만 근사 구현한 것이다.

동일 ground truth, 동일 Cross-Encoder(BAAI/bge-reranker-v2-m3), 동일 후보 풀(top-20) —
rerank_compare.py(옵션1)와 직접 비교 가능.

실행: .venv/bin/python -m backend.eval.seal_rag_lite_compare
"""

from sentence_transformers import CrossEncoder

from backend.api.services.retrieval import fetch_candidates
from backend.eval.lightrag_compare import LAWS_PATH, _N_QUERIES, build_ground_truth
from backend.eval.rerank_compare import _CANDIDATE_POOL, _RERANKER_MODEL, _TOP_K, rrf_only_hit
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("seal_rag_lite_compare.log")
OUT_PATH = PROJECT_ROOT / "data/eval/seal_rag_lite_vs_rerank_report.json"


def seal_rag_lite_hit(reranker: CrossEncoder, query_text: str, correct_pairs: list[tuple]) -> bool:
    candidates = fetch_candidates(query_text, top_k_per_source=_CANDIDATE_POOL, sparse_similarity_threshold=0.10, unified=False)
    law_candidates = candidates.get("law", [])
    if not law_candidates:
        return False

    # RRF 원래 순서 그대로 top-5 유지, 나머지는 overflow 후보 풀
    top5 = law_candidates[:_TOP_K]
    overflow = law_candidates[_TOP_K:]
    if not overflow:
        found = {(c["metadata"].get("law_name"), c["metadata"].get("article_no")) for c in top5}
        return bool(found & set(map(tuple, correct_pairs)))

    scores = reranker.predict([(query_text, c["text"]) for c in top5 + overflow])
    top5_scores = scores[:len(top5)]
    overflow_scores = scores[len(top5):]

    worst_idx = min(range(len(top5)), key=lambda i: top5_scores[i])
    best_overflow_idx = max(range(len(overflow)), key=lambda i: overflow_scores[i])

    # 딱 1회만 스왑 — overflow의 최고점이 top5 최저점보다 나을 때만
    if overflow_scores[best_overflow_idx] > top5_scores[worst_idx]:
        top5[worst_idx] = overflow[best_overflow_idx]

    found = {(c["metadata"].get("law_name"), c["metadata"].get("article_no")) for c in top5}
    return bool(found & set(map(tuple, correct_pairs)))


def main() -> None:
    law_recs = load_jsonl(LAWS_PATH)
    queries = build_ground_truth(law_recs, n_cases=_N_QUERIES)
    logger.info(f"  평가 쿼리 {len(queries)}건 생성(rerank_compare.py와 동일 seed/로직)")

    logger.info(f"  Cross-Encoder 로드: {_RERANKER_MODEL}")
    reranker = CrossEncoder(_RERANKER_MODEL, max_length=512)

    rrf_hits = seal_hits = 0
    per_query = []
    for i, q in enumerate(queries):
        r_hit = rrf_only_hit(q["clause"], q["correct_pairs"])
        s_hit = seal_rag_lite_hit(reranker, q["clause"], q["correct_pairs"])
        rrf_hits += r_hit
        seal_hits += s_hit
        per_query.append({"case_name": q["case_name"], "correct_pairs": q["correct_pairs"], "rrf_hit": r_hit, "seal_rag_lite_hit": s_hit})
        if (i + 1) % 25 == 0:
            logger.info(f"  평가 진행: {i + 1}/{len(queries)} | 누적 RRF={rrf_hits} SEAL-lite={seal_hits}")

    n = len(queries)
    result = {
        "n_queries": n,
        "rrf_hit_rate": rrf_hits / n,
        "seal_rag_lite_hit_rate": seal_hits / n,
        "rrf_hits": rrf_hits,
        "seal_rag_lite_hits": seal_hits,
        "per_query": per_query,
        "note": "RRF top-5 순서 보존 + 최저점 슬롯 1개만 교체(1회 스왑) — 옵션1(전체 재정렬)과 동일 pool/모델, 교체 폭만 다름.",
    }
    save_json(result, OUT_PATH)
    logger.info(f"========== 최종 결과 (n={n}) ==========")
    logger.info(f"  RRF-only(top20 후보 중 top5): {rrf_hits}/{n} ({rrf_hits/n*100:.1f}%)")
    logger.info(f"  SEAL-RAG-lite(1회 스왑):       {seal_hits}/{n} ({seal_hits/n*100:.1f}%)")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
