# backend/eval/domain_filter_compare.py
"""
법령 코퍼스는 극단적으로 불균형하다 — 3,323청크 중 약관의 규제에 관한 법률은 43개뿐인데
FTC 근거_법령 인용의 94.7%(2,314/2,444)를 차지한다. 지금 RRF는 법령 전체(16개 법령/
시행령/시행규칙, 민법 1,305·상법 1,271 등 소수 대형 법령이 사실상 다수)를 한 풀에서
경쟁시키므로, 정답인 약관규제법 조문이 훨씬 큰 법령들에 밀려 top_k 밖으로 안 나올
가능성이 있다 — LightRAG의 실패 원인(그래프 희석)과 같은 문제의식을 RRF 레벨에서
재현한 것.

law_name 메타데이터로 파티션(16개)을 나눠 각 파티션에서 독립적으로 dense/sparse
top-K를 뽑되(소수 파티션의 후보가 애초에 후보 풀에도 못 들어가는 걸 방지), 파티션을
넘어 병합할 때는 **실제 유사도 점수**(dense=코사인 유사도, sparse=trigram 유사도 —
둘 다 파티션과 무관하게 같은 척도라 비교 가능)로 정렬한다.

(첫 구현은 파티션별 RRF 순위(rank)로 병합했는데, 이러면 모든 파티션의 1위 후보가
"파티션 내부 순위 1위"라는 이유만으로 동일 점수를 받아 실제 관련성과 무관하게
파티션 나열 순서로 승부가 갈리는 버그가 있었다 — 100건 중 25건 시점에 0건 적중이라는
비정상 수치로 발견, 재현 스크립트로 원인 확인 후 실제 유사도 값 기반 병합으로 교체.)

동일 ground truth(FTC 근거_법령, 법령 전체 3,323청크, seed=42, 100건) — rerank_compare.py,
lightrag_compare_final.py와 같은 쿼리 세트로 직접 비교 가능.

실행: .venv/bin/python -m backend.eval.domain_filter_compare
"""

from typing import Any

from backend.api.services.retrieval import _fuse_reciprocal_rank, _get_cached_embedder, _search_dense, _search_sparse
from backend.db.connection import get_conn
from backend.db.loader import embed_texts
from backend.eval.lightrag_compare import _N_QUERIES, LAWS_PATH, build_ground_truth
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("domain_filter_compare.log")
OUT_PATH = PROJECT_ROOT / "data/eval/domain_filter_vs_rrf_report.json"

_CANDIDATE_K = 8  # 파티션당 확보할 후보 수(production _CANDIDATE_K와 동일)
_TOP_K = 5


def _law_names(cur: Any) -> list[str]:
    cur.execute("SELECT DISTINCT metadata->>'law_name' FROM chunks WHERE source = 'law'")
    return [row[0] for row in cur.fetchall() if row[0]]


def _dense_with_score(cur: Any, law_name: str, vec_literal: str, top_k: int) -> list[tuple]:
    """파티션 안에서 dense 검색 — 실제 코사인 유사도 점수를 같이 반환(파티션 간 비교용)."""
    cur.execute(
        """
        SELECT chunk_id, source, metadata, text, 1 - (embedding <=> %s::vector) AS score
        FROM chunks
        WHERE source = 'law' AND metadata->>'law_name' = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (vec_literal, law_name, vec_literal, top_k),
    )
    return cur.fetchall()


def _sparse_with_score(cur: Any, law_name: str, query_text: str, top_k: int, threshold: float = 0.10) -> list[tuple]:
    """파티션 안에서 sparse(pg_trgm) 검색 — 실제 유사도 점수를 같이 반환(파티션 간 비교용)."""
    cur.execute("SET pg_trgm.similarity_threshold = %s", (threshold,))
    cur.execute(
        """
        SELECT chunk_id, source, metadata, text, similarity(text, %s) AS score
        FROM chunks
        WHERE source = 'law' AND metadata->>'law_name' = %s AND text %% %s
        ORDER BY score DESC
        LIMIT %s
        """,
        (query_text, law_name, query_text, top_k),
    )
    return cur.fetchall()


def domain_filtered_hit(cur: Any, embedder: Any, query_text: str, law_names: list[str], correct_pairs: list[tuple]) -> bool:
    query_vec = embed_texts(embedder, [query_text], prefix="query: ")[0]
    vec_literal = "[" + ",".join(repr(x) for x in query_vec) + "]"

    all_dense, all_sparse = [], []
    for law_name in law_names:
        all_dense.extend(_dense_with_score(cur, law_name, vec_literal, _CANDIDATE_K))
        all_sparse.extend(_sparse_with_score(cur, law_name, query_text, _CANDIDATE_K))

    # 실제 유사도 점수로 파티션 경계 없이 재정렬 — 각자 자기 파티션에서 이미 top-K로
    # 확보됐으니, 이제부터는 소수 파티션 후보도 다수 파티션 후보와 동등하게 경쟁한다.
    dense_sorted  = [row[:4] for row in sorted(all_dense, key=lambda r: r[4], reverse=True)]
    sparse_sorted = [row[:4] for row in sorted(all_sparse, key=lambda r: r[4], reverse=True)]

    ranked = _fuse_reciprocal_rank(dense_sorted, sparse_sorted)[:_TOP_K]
    found = {(c["metadata"].get("law_name"), c["metadata"].get("article_no")) for c in ranked}
    return bool(found & set(map(tuple, correct_pairs)))


def rrf_baseline_hit(cur: Any, embedder: Any, query_text: str, correct_pairs: list[tuple]) -> bool:
    """비교 기준 — 파티션 없이 법령 전체를 한 풀에서 RRF(현재 production 방식과 동일)."""
    query_vec = embed_texts(embedder, [query_text], prefix="query: ")[0]
    vec_literal = "[" + ",".join(repr(x) for x in query_vec) + "]"
    dense = _search_dense(cur, ["law"], vec_literal, _CANDIDATE_K)
    sparse = _search_sparse(cur, ["law"], query_text, _CANDIDATE_K, 0.10)
    ranked = _fuse_reciprocal_rank(dense, sparse)[:_TOP_K]
    found = {(c["metadata"].get("law_name"), c["metadata"].get("article_no")) for c in ranked}
    return bool(found & set(map(tuple, correct_pairs)))


def main() -> None:
    law_recs = load_jsonl(LAWS_PATH)
    queries = build_ground_truth(law_recs, n_cases=_N_QUERIES)
    logger.info(f"  평가 쿼리 {len(queries)}건 생성(법령 전체 대상, 동일 seed/로직)")

    embedder = _get_cached_embedder()
    with get_conn() as conn, conn.cursor() as cur:
        law_names = _law_names(cur)
        logger.info(f"  파티션(law_name) {len(law_names)}개: {law_names}")

        rrf_hits = domain_hits = 0
        per_query = []
        for i, q in enumerate(queries):
            r_hit = rrf_baseline_hit(cur, embedder, q["clause"], q["correct_pairs"])
            d_hit = domain_filtered_hit(cur, embedder, q["clause"], law_names, q["correct_pairs"])
            rrf_hits += r_hit
            domain_hits += d_hit
            per_query.append({"case_name": q["case_name"], "correct_pairs": q["correct_pairs"], "rrf_hit": r_hit, "domain_filter_hit": d_hit})
            if (i + 1) % 25 == 0:
                logger.info(f"  평가 진행: {i + 1}/{len(queries)} | 누적 RRF={rrf_hits} DomainFilter={domain_hits}")

    n = len(queries)
    result = {
        "n_queries": n,
        "law_names": law_names,
        "rrf_hit_rate": rrf_hits / n,
        "domain_filter_hit_rate": domain_hits / n,
        "rrf_hits": rrf_hits,
        "domain_filter_hits": domain_hits,
        "per_query": per_query,
        "note": "law_name 파티션별 top-K를 실제 유사도 점수로 재정렬 후 dense/sparse RRF — 그래프 없음, production fetch_candidates()와 별개 구현.",
    }
    save_json(result, OUT_PATH)
    logger.info(f"========== 최종 결과 (n={n}) ==========")
    logger.info(f"  RRF(파티션 없음):        {rrf_hits}/{n} ({rrf_hits/n*100:.1f}%)")
    logger.info(f"  도메인 필터링 Hybrid:    {domain_hits}/{n} ({domain_hits/n*100:.1f}%)")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
