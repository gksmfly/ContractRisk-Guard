# backend/eval/rerank_compare.py
"""
LightRAG의 실패 원인(그래프 희석)이 "그래프 구조 자체"에 있다는 분석(memory
project_lightrag_vs_rrf.md 참고) 이후, 그래프 없이 정밀도만 올리는 최소 변경안을
같은 ground truth(FTC 근거_법령, 법령 전체 3,323청크, seed=42, 동일 100건)로 검증한다.

RRF 후보군(top_k_per_source를 넉넉히 키움)에 Cross-Encoder(BAAI/bge-reranker-v2-m3,
로컬 캐시 완료 — 추가 다운로드·API 비용 없음)로 재랭킹만 추가. retrieval.py 독스트링에
있던 "10.7% vs 20.1%" 옛 실험은 스크립트가 레포에 없어 재현 불가·ground truth 불명이라
신뢰할 수 없다고 이미 문서화됨(project_lightrag_vs_rrf.md) — 이번이 이 비교의 첫
재현 가능한 버전이다.

실행: .venv/bin/python -m backend.eval.rerank_compare
"""

from sentence_transformers import CrossEncoder

from backend.api.services.retrieval import fetch_candidates
from backend.eval.lightrag_compare import _N_QUERIES, LAWS_PATH, build_ground_truth
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("rerank_compare.log")
OUT_PATH = PROJECT_ROOT / "data/eval/rerank_vs_rrf_report.json"

_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_CANDIDATE_POOL = 20  # 재랭킹이 뽑아낼 여지를 주기 위해 RRF top-5보다 넉넉히 확보
_TOP_K = 5


def rerank_hit(reranker: CrossEncoder, query_text: str, correct_pairs: list[tuple]) -> bool:
    candidates = fetch_candidates(query_text, top_k_per_source=_CANDIDATE_POOL, sparse_similarity_threshold=0.10, unified=False)
    law_candidates = candidates.get("law", [])
    if not law_candidates:
        return False

    pairs_scores = reranker.predict([(query_text, c["text"]) for c in law_candidates])
    ranked = sorted(zip(law_candidates, pairs_scores), key=lambda x: x[1], reverse=True)[:_TOP_K]

    found = {(c["metadata"].get("law_name"), c["metadata"].get("article_no")) for c, _ in ranked}
    return bool(found & set(map(tuple, correct_pairs)))


def rrf_only_hit(query_text: str, correct_pairs: list[tuple]) -> bool:
    """비교 기준 — 같은 후보 풀(top_k=20)에서 재랭킹 없이 RRF 순위 그대로 top-5만 본다."""
    candidates = fetch_candidates(query_text, top_k_per_source=_CANDIDATE_POOL, sparse_similarity_threshold=0.10, unified=False)
    law_candidates = candidates.get("law", [])[:_TOP_K]
    found = {(c["metadata"].get("law_name"), c["metadata"].get("article_no")) for c in law_candidates}
    return bool(found & set(map(tuple, correct_pairs)))


def main() -> None:
    law_recs = load_jsonl(LAWS_PATH)
    queries = build_ground_truth(law_recs, n_cases=_N_QUERIES)
    logger.info(f"  평가 쿼리 {len(queries)}건 생성(법령 전체 대상, lightrag_compare_final.py와 동일 seed/로직)")

    logger.info(f"  Cross-Encoder 로드: {_RERANKER_MODEL}")
    reranker = CrossEncoder(_RERANKER_MODEL, max_length=512)

    rrf_hits = rerank_hits = 0
    per_query = []
    for i, q in enumerate(queries):
        r_hit = rrf_only_hit(q["clause"], q["correct_pairs"])
        rr_hit = rerank_hit(reranker, q["clause"], q["correct_pairs"])
        rrf_hits += r_hit
        rerank_hits += rr_hit
        per_query.append({"case_name": q["case_name"], "correct_pairs": q["correct_pairs"], "rrf_hit": r_hit, "rerank_hit": rr_hit})
        if (i + 1) % 25 == 0:
            logger.info(f"  평가 진행: {i + 1}/{len(queries)} | 누적 RRF={rrf_hits} Rerank={rerank_hits}")

    n = len(queries)
    result = {
        "n_queries": n,
        "reranker_model": _RERANKER_MODEL,
        "candidate_pool_size": _CANDIDATE_POOL,
        "rrf_hit_rate": rrf_hits / n,
        "rerank_hit_rate": rerank_hits / n,
        "rrf_hits": rrf_hits,
        "rerank_hits": rerank_hits,
        "per_query": per_query,
        "note": (
            "동일 후보 풀(top_k_per_source=20) 안에서 RRF 순위 그대로 top-5 vs "
            "Cross-Encoder 재랭킹 top-5 비교 — 후보 풀 크기 차이로 인한 불공정 비교 방지."
        ),
    }
    save_json(result, OUT_PATH)
    logger.info(f"========== 최종 결과 (n={n}) ==========")
    logger.info(f"  RRF-only(top20 후보 중 top5): {rrf_hits}/{n} ({rrf_hits/n*100:.1f}%)")
    logger.info(f"  RRF+Cross-Encoder 재랭킹:      {rerank_hits}/{n} ({rerank_hits/n*100:.1f}%)")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
