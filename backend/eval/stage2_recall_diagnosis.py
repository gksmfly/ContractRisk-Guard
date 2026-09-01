# backend/eval/stage2_recall_diagnosis.py
"""
2단계(라우팅 이후 조문 검색) 실패 원인 진단 — "순위 문제"인가 "검색 문제"인가.

배경: RAPTOR-lite 실측에서 1단계 라우팅 정답률은 78/100인데 최종 조문 적중은 33/100이다.
즉 **법을 맞춰놓고 조문을 놓친 45건**이 전체 손실의 대부분이다. 이 45건이 어느 쪽인지에
따라 해법이 완전히 갈린다:

  (A) 정답 조문이 후보 풀에 **있는데 top-5 밖으로 밀린 것**  → 재랭킹·융합 개선이 답
  (B) 정답 조문이 후보 풀에 **아예 안 잡히는 것**            → 쿼리·임베딩이 답
                                                              (재랭킹은 아무 소용 없음)

이 스크립트는 `raptor_lite_vs_rrf_report.json`에 이미 기록된 EXAONE 예측 법령을 그대로
재사용한다 — LLM을 다시 호출하지 않으므로(GPU LLM 로드 없음, API 비용 0) 검색 단계만
독립적으로 측정한다. 후보 풀을 크게 잡고(법령당 top-50) 정답 조문이 **몇 위에** 있는지
분포를 낸다.

실행: .venv/bin/python -m backend.eval.stage2_recall_diagnosis
"""

import json
from collections import Counter

from backend.api.services.retrieval import _get_cached_embedder, _reciprocal_rank_fusion
from backend.db.connection import get_conn
from backend.db.loader import embed_texts
from backend.eval.domain_filter_compare import _dense_with_score, _sparse_with_score
from backend.eval.lightrag_compare import _N_QUERIES, LAWS_PATH, build_ground_truth
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("stage2_recall_diagnosis.log")
EXAONE_REPORT = PROJECT_ROOT / "data/eval/raptor_lite_vs_rrf_report.json"
OUT_PATH = PROJECT_ROOT / "data/eval/stage2_recall_diagnosis.json"

_POOL_K = 50   # 법령당 확보할 후보 수(현재 운영값 8보다 크게 잡아 "풀에 있긴 한가"를 본다)
_CUTOFFS = (1, 2, 3, 5, 10, 20, 50, 100)


def _first_correct_rank(ranked: list[dict], correct: set[tuple]) -> int | None:
    """RRF로 정렬된 후보에서 정답 (law_name, article_no)이 처음 등장하는 순위(1-based)."""
    for i, c in enumerate(ranked):
        md = c["metadata"] or {}
        if (md.get("law_name"), md.get("article_no")) in correct:
            return i + 1
    return None


def main() -> None:
    report = json.loads(EXAONE_REPORT.read_text())
    recorded = {q["case_name"]: q for q in report["per_query"]}

    law_recs = load_jsonl(LAWS_PATH)
    queries = build_ground_truth(law_recs, n_cases=_N_QUERIES)
    embedder = _get_cached_embedder()

    rows = []
    with get_conn() as conn, conn.cursor() as cur:
        for i, q in enumerate(queries):
            rec = recorded.get(q["case_name"])
            if rec is None or not rec["routing_correct"]:
                continue  # 라우팅이 틀린 건은 2단계 진단 대상이 아니다

            predicted = rec["predicted_laws"]
            correct = set(map(tuple, q["correct_pairs"]))

            query_vec = embed_texts(embedder, [q["clause"]], prefix="query: ")[0]
            vec_literal = "[" + ",".join(repr(x) for x in query_vec) + "]"

            all_dense, all_sparse = [], []
            for law_name in predicted:
                all_dense.extend(_dense_with_score(cur, law_name, vec_literal, _POOL_K))
                all_sparse.extend(_sparse_with_score(cur, law_name, q["clause"], _POOL_K))

            dense_sorted  = [r[:4] for r in sorted(all_dense,  key=lambda r: r[4], reverse=True)]
            sparse_sorted = [r[:4] for r in sorted(all_sparse, key=lambda r: r[4], reverse=True)]
            ranked = _reciprocal_rank_fusion(dense_sorted, sparse_sorted)

            rank = _first_correct_rank(ranked, correct)
            # 정답 조문이 그 법령 파티션 안에 애초에 존재하는지(코퍼스 커버리지 확인)
            in_corpus = []
            for law_name, article_no in correct:
                cur.execute(
                    "SELECT 1 FROM chunks WHERE source='law' AND metadata->>'law_name'=%s "
                    "AND metadata->>'article_no'=%s LIMIT 1",
                    (law_name, str(article_no)),
                )
                in_corpus.append(cur.fetchone() is not None)

            rows.append({
                "case_name": q["case_name"],
                "predicted_laws": predicted,
                "n_correct_articles": len(correct),
                "pool_size": len(ranked),
                "first_correct_rank": rank,
                "any_correct_in_corpus": any(in_corpus),
                "recorded_hit_at5": bool(rec["routed_hit"]),
            })
            if (i + 1) % 25 == 0:
                logger.info(f"  진행 {i + 1}/{len(queries)} (진단 대상 누적 {len(rows)}건)")

    n = len(rows)
    found = [r for r in rows if r["first_correct_rank"] is not None]
    missing = [r for r in rows if r["first_correct_rank"] is None]
    not_in_corpus = [r for r in missing if not r["any_correct_in_corpus"]]

    recall = {k: sum(1 for r in found if r["first_correct_rank"] <= k) for k in _CUTOFFS}

    result = {
        "n_routing_correct": n,
        "pool_k_per_law": _POOL_K,
        "found_anywhere_in_pool": len(found),
        "not_found_in_pool": len(missing),
        "of_which_article_absent_from_corpus": len(not_in_corpus),
        "recall_at": {str(k): v for k, v in recall.items()},
        "recall_at_rate": {str(k): round(v / n, 3) for k, v in recall.items()},
        "rank_histogram": dict(Counter(
            "1" if r["first_correct_rank"] == 1 else
            "2-5" if r["first_correct_rank"] <= 5 else
            "6-20" if r["first_correct_rank"] <= 20 else
            "21-50" if r["first_correct_rank"] <= 50 else "51+"
            for r in found
        )),
        "rows": rows,
    }
    save_json(result, OUT_PATH)

    logger.info(f"========== 2단계 진단 (라우팅 정답 {n}건 대상, 법령당 풀 {_POOL_K}) ==========")
    logger.info(f"  후보 풀에 정답이 존재:      {len(found)}/{n} ({len(found)/n*100:.1f}%)")
    logger.info(f"  풀에 아예 없음:             {len(missing)}/{n}")
    logger.info(f"    └ 그중 코퍼스에 조문 자체가 없음: {len(not_in_corpus)}건")
    for k in _CUTOFFS:
        logger.info(f"  recall@{k:<3}: {recall[k]}/{n} ({recall[k]/n*100:.1f}%)")
    logger.info(f"  정답 순위 분포: {result['rank_histogram']}")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
