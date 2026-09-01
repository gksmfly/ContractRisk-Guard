# backend/eval/raptor_lite_compare.py
"""
RAPTOR(계층적 인덱싱)의 핵심 아이디어 — "브루트포스로 전부 검색하지 않고 계층을 타고
내려가며 검색 범위를 좁힌다" — 를 경량 근사한다. 이 법령 코퍼스는 이미 law_name이라는
자연스러운 최상위 계층을 갖고 있어서(제/항/호 단위의 깊은 트리 구조까지는 안 만듦 —
RAPTOR 원 논문의 재귀적 클러스터링·다단계 요약 트리 전체를 재현한 게 아니라 "최상위
계층 라우팅"만 구현한 경량 근사임을 분명히 한다).

domain_filter_compare.py(옵션2)는 16개 파티션을 전부 검색해서 점수로 병합했는데,
여기서는 그 대신 로컬 LLM(EXAONE-3.5-7.8B-Instruct, API 비용 0)이 쿼리를 보고 가장
관련 있을 법령 top-2를 먼저 고르고(라우팅), 그 파티션만 검색한다 — "브루트포스 검색
+ 점수 병합"(옵션2) vs "스마트 라우팅 + 좁은 검색"(이 스크립트)의 비교.

동일 ground truth(FTC 근거_법령, 법령 전체 3,323청크, seed=42, 100건).

실행: .venv/bin/python -m backend.eval.raptor_lite_compare
"""

from typing import Any

from backend.api.services.retrieval import _fuse_reciprocal_rank, _get_cached_embedder
from backend.db.connection import get_conn
from backend.db.loader import embed_texts
from backend.eval.domain_filter_compare import _dense_with_score, _law_names, _sparse_with_score, rrf_baseline_hit
from backend.eval.lightrag_compare import _N_QUERIES, LAWS_PATH, build_ground_truth
from backend.eval.local_llm import generate_json
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("raptor_lite_compare.log")
OUT_PATH = PROJECT_ROOT / "data/eval/raptor_lite_vs_rrf_report.json"

_CANDIDATE_K = 8
_TOP_K = 5

_SYSTEM_TMPL = (
    "너는 한국 법률 전문가다. 계약 조항을 읽고, 이 조항의 불공정성을 다툴 때 가장 근거로 "
    "삼을 가능성이 높은 법령을 아래 목록에서 정확히 2개 골라라.\n"
    "목록: {law_list}\n"
    '반드시 JSON만 출력: {{"laws": ["법령명1", "법령명2"]}}'
)
_FEWSHOT = [
    {"role": "user", "content": "계약 조항:\n제5조 을은 계약기간 중 언제든지 갑에게 서면 통지만으로 계약을 해지할 수 있으며, 갑은 이에 대해 이의를 제기할 수 없다."},
    {"role": "assistant", "content": '{"laws": ["약관의 규제에 관한 법률", "민법"]}'},
    {"role": "user", "content": "계약 조항:\n제9조 회원이 이용약관을 위반한 경우 회사는 손해배상과 별도로 위약금으로 계약금의 3배를 청구할 수 있다."},
    {"role": "assistant", "content": '{"laws": ["약관의 규제에 관한 법률", "민법"]}'},
]


def routed_hit(cur: Any, embedder: Any, query_text: str, predicted_laws: list[str], correct_pairs: list[tuple]) -> bool:
    if not predicted_laws:
        return False
    query_vec = embed_texts(embedder, [query_text], prefix="query: ")[0]
    vec_literal = "[" + ",".join(repr(x) for x in query_vec) + "]"

    all_dense, all_sparse = [], []
    for law_name in predicted_laws:
        all_dense.extend(_dense_with_score(cur, law_name, vec_literal, _CANDIDATE_K))
        all_sparse.extend(_sparse_with_score(cur, law_name, query_text, _CANDIDATE_K))

    dense_sorted  = [row[:4] for row in sorted(all_dense, key=lambda r: r[4], reverse=True)]
    sparse_sorted = [row[:4] for row in sorted(all_sparse, key=lambda r: r[4], reverse=True)]
    ranked = _fuse_reciprocal_rank(dense_sorted, sparse_sorted)[:_TOP_K]
    found = {(c["metadata"].get("law_name"), c["metadata"].get("article_no")) for c in ranked}
    return bool(found & set(map(tuple, correct_pairs)))


def main() -> None:
    law_recs = load_jsonl(LAWS_PATH)
    queries = build_ground_truth(law_recs, n_cases=_N_QUERIES)
    logger.info(f"  평가 쿼리 {len(queries)}건 생성(법령 전체 대상, 동일 seed/로직)")

    embedder = _get_cached_embedder()
    with get_conn() as conn, conn.cursor() as cur:
        law_names = _law_names(cur)
        system = _SYSTEM_TMPL.format(law_list=", ".join(law_names))
        logger.info(f"  파티션 {len(law_names)}개, 라우팅 대상")

        rrf_hits = routed_hits = routing_correct = 0
        per_query = []
        for i, q in enumerate(queries):
            r_hit = rrf_baseline_hit(cur, embedder, q["clause"], q["correct_pairs"])

            content = f"계약 조항:\n{q['clause'][:800]}"
            result = generate_json(system, content, max_new_tokens=100, fewshot=_FEWSHOT)
            predicted = [law for law in (result or {}).get("laws", []) if law in law_names]
            correct_laws = {p[0] for p in q["correct_pairs"]}
            is_routing_correct = bool(set(predicted) & correct_laws)

            rt_hit = routed_hit(cur, embedder, q["clause"], predicted, q["correct_pairs"])

            rrf_hits += r_hit
            routed_hits += rt_hit
            routing_correct += is_routing_correct
            per_query.append({
                "case_name": q["case_name"], "correct_pairs": q["correct_pairs"],
                "predicted_laws": predicted, "routing_correct": is_routing_correct,
                "rrf_hit": r_hit, "routed_hit": rt_hit,
            })
            if (i + 1) % 10 == 0:
                logger.info(f"  평가 진행: {i + 1}/{len(queries)} | 누적 RRF={rrf_hits} Routed={routed_hits} 라우팅정답률={routing_correct}/{i+1}")

    n = len(queries)
    result = {
        "n_queries": n,
        "rrf_hit_rate": rrf_hits / n,
        "routed_hit_rate": routed_hits / n,
        "routing_accuracy": routing_correct / n,
        "rrf_hits": rrf_hits,
        "routed_hits": routed_hits,
        "per_query": per_query,
        "note": (
            "RAPTOR 원 논문의 재귀적 클러스터링·다단계 요약 트리는 미구현 — "
            "law_name 최상위 계층 라우팅만 경량 근사(로컬 EXAONE-3.5-7.8B top-2 예측, API 비용 0)."
        ),
    }
    save_json(result, OUT_PATH)
    logger.info(f"========== 최종 결과 (n={n}) ==========")
    logger.info(f"  RRF(파티션 없음):  {rrf_hits}/{n} ({rrf_hits/n*100:.1f}%)")
    logger.info(f"  RAPTOR-lite 라우팅: {routed_hits}/{n} ({routed_hits/n*100:.1f}%)")
    logger.info(f"  라우팅 자체 정답률: {routing_correct}/{n} ({routing_correct/n*100:.1f}%)")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
