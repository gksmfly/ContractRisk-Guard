# backend/eval/qwen_variant_compare.py
"""
가장 성능이 좋았던 두 방법(LegalMALR-lite 쿼리 재구성, RAPTOR-lite 법령 라우팅)을 로컬
모델만 EXAONE-3.5-7.8B-Instruct(한국어 특화, 7.8B) 대신 Qwen2.5-14B-Instruct(다국어 범용,
14B, 이미 캐시됨)로 바꿔서 재실행 — 모델 선택 자체가 33%라는 결과에 얼마나 기여했는지
분리해서 본다. 프롬프트·few-shot·ground truth는 legalmalr_lite_compare.py,
raptor_lite_compare.py와 완전히 동일하게 재사용한다.

실행: .venv/bin/python -m backend.eval.qwen_variant_compare
"""

from backend.api.services.retrieval import _get_cached_embedder
from backend.db.connection import get_conn
from backend.eval.domain_filter_compare import _law_names, rrf_baseline_hit
from backend.eval.legalmalr_lite_compare import _FEWSHOT as _REFORM_FEWSHOT, _SYSTEM as _REFORM_SYSTEM
from backend.eval.lightrag_compare import LAWS_PATH, _N_QUERIES, build_ground_truth
from backend.eval.local_llm import generate_json
from backend.eval.raptor_lite_compare import _FEWSHOT as _ROUTE_FEWSHOT, _SYSTEM_TMPL as _ROUTE_SYSTEM_TMPL, routed_hit
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("qwen_variant_compare.log")
OUT_PATH = PROJECT_ROOT / "data/eval/qwen_variant_report.json"

_MODEL_KEY = "qwen2.5-14b"


def main() -> None:
    law_recs = load_jsonl(LAWS_PATH)
    queries = build_ground_truth(law_recs, n_cases=_N_QUERIES)
    logger.info(f"  평가 쿼리 {len(queries)}건 생성(legalmalr_lite/raptor_lite와 동일 seed/로직)")
    logger.info(f"  모델: {_MODEL_KEY}(로딩에 시간 소요될 수 있음)")

    embedder = _get_cached_embedder()
    with get_conn() as conn, conn.cursor() as cur:
        law_names = _law_names(cur)
        route_system = _ROUTE_SYSTEM_TMPL.format(law_list=", ".join(law_names))

        rrf_hits = reformed_hits = routed_hits = 0
        per_query = []
        for i, q in enumerate(queries):
            r_hit = rrf_baseline_hit(cur, embedder, q["clause"], q["correct_pairs"])
            content = f"계약 조항:\n{q['clause'][:800]}"

            reform_result = generate_json(_REFORM_SYSTEM, content, max_new_tokens=150, fewshot=_REFORM_FEWSHOT, model_key=_MODEL_KEY)
            reformed = (reform_result or {}).get("reformed_query", "").strip()
            rf_hit = rrf_baseline_hit(cur, embedder, reformed, q["correct_pairs"]) if reformed else False

            route_result = generate_json(route_system, content, max_new_tokens=100, fewshot=_ROUTE_FEWSHOT, model_key=_MODEL_KEY)
            predicted = [law for law in (route_result or {}).get("laws", []) if law in law_names]
            rt_hit = routed_hit(cur, embedder, q["clause"], predicted, q["correct_pairs"])

            rrf_hits += r_hit
            reformed_hits += rf_hit
            routed_hits += rt_hit
            per_query.append({
                "case_name": q["case_name"], "correct_pairs": q["correct_pairs"],
                "reformed_query": reformed, "predicted_laws": predicted,
                "rrf_hit": r_hit, "reformed_query_hit": rf_hit, "routed_hit": rt_hit,
            })
            if (i + 1) % 10 == 0:
                logger.info(f"  평가 진행: {i + 1}/{len(queries)} | RRF={rrf_hits} 재구성(Qwen)={reformed_hits} 라우팅(Qwen)={routed_hits}")

    n = len(queries)
    result = {
        "n_queries": n,
        "model": _MODEL_KEY,
        "rrf_hit_rate": rrf_hits / n,
        "reformed_query_hit_rate_qwen": reformed_hits / n,
        "routed_hit_rate_qwen": routed_hits / n,
        "rrf_hits": rrf_hits,
        "reformed_query_hits_qwen": reformed_hits,
        "routed_hits_qwen": routed_hits,
        "per_query": per_query,
        "note": "legalmalr_lite_compare.py/raptor_lite_compare.py와 동일 프롬프트·few-shot·ground truth, 모델만 EXAONE→Qwen2.5-14B 교체.",
    }
    save_json(result, OUT_PATH)
    logger.info(f"========== 최종 결과 (n={n}, 모델={_MODEL_KEY}) ==========")
    logger.info(f"  RRF(원문 그대로):        {rrf_hits}/{n} ({rrf_hits/n*100:.1f}%)")
    logger.info(f"  쿼리 재구성(Qwen):        {reformed_hits}/{n} ({reformed_hits/n*100:.1f}%)  [EXAONE 버전: 33%]")
    logger.info(f"  법령 라우팅(Qwen):        {routed_hits}/{n} ({routed_hits/n*100:.1f}%)  [EXAONE 버전: 33%]")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
