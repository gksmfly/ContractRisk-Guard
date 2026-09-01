# backend/eval/stage2_improvement_compare.py
"""
2단계(라우팅 이후 조문 랭킹) 개선안 3종을 같은 100건 ground truth로 한꺼번에 비교한다.

`stage2_recall_diagnosis.py` 결과로 병목이 확정됐다: 라우팅이 맞은 78건 중 **77건(98.7%)은
정답 조문이 후보 풀 안에 이미 존재**하는데, top-5만 노출해서 39.7%만 건진다. 즉 검색
문제가 아니라 **순위·노출 개수 문제**다. 그래서 여기서는 라우팅을 건드리지 않는다 —
`raptor_lite_vs_rrf_report.json`에 기록된 EXAONE 라우팅 결과를 **그대로 재사용**해서
1단계를 상수로 고정하고, 2단계만 바꿔가며 측정한다(라우팅 재호출 없음 = 비교 통제 +
LLM 호출 절약).

측정 대상:
  A. 후보 풀·노출 개수 확대 (LLM 없음)      — pool 8/30/50 × top 2/5/10/20
  B. EXAONE listwise 재랭킹 (LLM 1회 추가)  — pool 30을 조문 본문까지 읽혀 재정렬
  C. 쿼리 재구성 별도 호출 (LLM 1회 추가)   — LegalMALR-lite 프롬프트를 라우팅과
     **분리된 호출**로 실행. 기존 "종합 콤보"(28%)는 재구성+라우팅을 한 호출에 합쳐서
     실패했는데, 분리 실행은 한 번도 측정된 적이 없다.

베이스라인 재현 검증: pool=8 / top=5 설정이 기존 실측치 33/100을 그대로 재현해야
이 하네스를 신뢰할 수 있다 — 결과표 맨 윗줄에서 확인할 것.

실행: .venv/bin/python -m backend.eval.stage2_improvement_compare
"""

import json

from backend.api.services.retrieval import _get_cached_embedder, _reciprocal_rank_fusion
from backend.db.connection import get_conn
from backend.db.loader import embed_texts
from backend.eval.domain_filter_compare import _dense_with_score, _sparse_with_score
from backend.eval.legalmalr_lite_compare import _FEWSHOT as _REFORM_FEWSHOT
from backend.eval.legalmalr_lite_compare import _SYSTEM as _REFORM_SYSTEM
from backend.eval.lightrag_compare import _N_QUERIES, LAWS_PATH, build_ground_truth
from backend.eval.local_llm import generate_json
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("stage2_improvement_compare.log")
EXAONE_REPORT = PROJECT_ROOT / "data/eval/raptor_lite_vs_rrf_report.json"
OUT_PATH = PROJECT_ROOT / "data/eval/stage2_improvement_report.json"

_POOL_SIZES = (8, 30, 50)
_CUTOFFS = (2, 5, 10, 20)
_RERANK_POOL = 30          # B안이 EXAONE에게 읽힐 후보 수
_RERANK_TEXT_LEN = 200     # 후보 1건당 본문 노출 길이(컨텍스트 관리)

_RERANK_SYSTEM = (
    "너는 한국 법률 전문가다. 아래 계약 조항의 불공정성을 다툴 때 **직접적인 근거 조문**이 "
    "되는 것을 후보 법조문 목록에서 관련도 높은 순으로 정확히 5개 골라라.\n"
    "조문의 실제 내용을 읽고 판단해라 — 단어가 겹치는지가 아니라, 그 조문이 이 계약 조항을 "
    "규율하는지를 봐라.\n"
    '반드시 JSON만 출력: {"top": [번호, 번호, 번호, 번호, 번호]}'
)


def _candidate_label(c: dict) -> str:
    md = c["metadata"] or {}
    title = md.get("article_title", "")
    head = f"{md.get('law_name', '')} 제{md.get('article_no', '')}조"
    if title:
        head = f"{head}({title})"
    return f"{head}: {c['text'][:_RERANK_TEXT_LEN].strip()}"


def _ranked(cur, embedder, query_text: str, laws: list[str], pool_k: int, _vec_cache: dict) -> list[dict]:
    """라우팅된 법령 파티션별로 pool_k씩 확보 → 실제 점수로 병합 → RRF 융합한 전체 순위."""
    if not laws or not query_text.strip():
        return []
    if query_text not in _vec_cache:
        vec = embed_texts(embedder, [query_text], prefix="query: ")[0]
        _vec_cache[query_text] = "[" + ",".join(repr(x) for x in vec) + "]"
    vec_literal = _vec_cache[query_text]

    all_dense, all_sparse = [], []
    for law_name in laws:
        all_dense.extend(_dense_with_score(cur, law_name, vec_literal, pool_k))
        all_sparse.extend(_sparse_with_score(cur, law_name, query_text, pool_k))

    dense_sorted  = [r[:4] for r in sorted(all_dense,  key=lambda r: r[4], reverse=True)]
    sparse_sorted = [r[:4] for r in sorted(all_sparse, key=lambda r: r[4], reverse=True)]
    return _reciprocal_rank_fusion(dense_sorted, sparse_sorted)


def _hit_at(ranked: list[dict], correct: set[tuple], k: int) -> bool:
    for c in ranked[:k]:
        md = c["metadata"] or {}
        if (md.get("law_name"), md.get("article_no")) in correct:
            return True
    return False


def _llm_rerank(clause: str, pool: list[dict]) -> list[dict]:
    """EXAONE에게 후보 조문 본문을 읽히고 상위 5개를 고르게 한 뒤, 나머지는 RRF 순서로 뒤에 붙인다."""
    if not pool:
        return pool
    listing = "\n".join(f"{i + 1}. {_candidate_label(c)}" for i, c in enumerate(pool))
    content = f"계약 조항:\n{clause[:800]}\n\n후보 법조문:\n{listing}"
    result = generate_json(_RERANK_SYSTEM, content, max_new_tokens=80)

    picked_idx, seen = [], set()
    for v in (result or {}).get("top", []):
        try:
            i = int(v) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(pool) and i not in seen:
            seen.add(i)
            picked_idx.append(i)
    if not picked_idx:
        return pool  # 파싱 실패 → RRF 순서 그대로(성능 저하 없이 폴백)
    return [pool[i] for i in picked_idx] + [c for i, c in enumerate(pool) if i not in seen]


def main(limit: int | None = None) -> None:
    recorded = {q["case_name"]: q for q in json.loads(EXAONE_REPORT.read_text())["per_query"]}
    law_recs = load_jsonl(LAWS_PATH)
    queries = build_ground_truth(law_recs, n_cases=_N_QUERIES)
    if limit:
        queries = queries[:limit]
    embedder = _get_cached_embedder()

    # 설정 이름 -> 적중 카운터
    hits: dict[str, int] = {}
    reform_ok = 0
    per_query = []

    with get_conn() as conn, conn.cursor() as cur:
        for i, q in enumerate(queries):
            rec = recorded.get(q["case_name"])
            laws = rec["predicted_laws"] if rec else []
            correct = set(map(tuple, q["correct_pairs"]))
            vec_cache: dict[str, str] = {}
            row = {"case_name": q["case_name"], "predicted_laws": laws, "routing_correct": bool(rec and rec["routing_correct"])}

            # --- A. 원문 쿼리, 풀 크기별 ---
            pools: dict[int, list[dict]] = {}
            for pk in _POOL_SIZES:
                pools[pk] = _ranked(cur, embedder, q["clause"], laws, pk, vec_cache)
                for k in _CUTOFFS:
                    name = f"A_pool{pk}_top{k}"
                    h = _hit_at(pools[pk], correct, k)
                    hits[name] = hits.get(name, 0) + h
                    row[name] = h

            # --- B. EXAONE listwise 재랭킹 (pool 30) ---
            reranked = _llm_rerank(q["clause"], pools[_RERANK_POOL][:_RERANK_POOL])
            for k in _CUTOFFS:
                name = f"B_llmrerank_top{k}"
                h = _hit_at(reranked, correct, k)
                hits[name] = hits.get(name, 0) + h
                row[name] = h

            # --- C. 쿼리 재구성(별도 호출) ---
            rf = generate_json(_REFORM_SYSTEM, f"계약 조항:\n{q['clause'][:800]}",
                               max_new_tokens=150, fewshot=_REFORM_FEWSHOT)
            reformed = ((rf or {}).get("reformed_query") or "").strip()
            reform_ok += bool(reformed)
            row["reformed_query"] = reformed
            for pk in (8, 30):
                rq = _ranked(cur, embedder, reformed, laws, pk, vec_cache) if reformed else []
                for k in _CUTOFFS:
                    name = f"C_reform_pool{pk}_top{k}"
                    h = _hit_at(rq, correct, k)
                    hits[name] = hits.get(name, 0) + h
                    row[name] = h

            per_query.append(row)
            if (i + 1) % 10 == 0:
                logger.info(
                    f"  진행 {i + 1}/{len(queries)} | 기준선(A_pool8_top5)={hits.get('A_pool8_top5', 0)} "
                    f"A30/5={hits.get('A_pool30_top5', 0)} B/5={hits.get('B_llmrerank_top5', 0)} "
                    f"C30/5={hits.get('C_reform_pool30_top5', 0)}"
                )

    n = len(queries)
    save_json({"n_queries": n, "reform_parsed_ok": reform_ok,
               "hits": hits, "hit_rate": {k: v / n for k, v in hits.items()},
               "per_query": per_query,
               "note": "라우팅은 raptor_lite_vs_rrf_report.json의 EXAONE 예측을 재사용(1단계 고정, 2단계만 변경)."},
              OUT_PATH)

    logger.info(f"========== 2단계 개선안 비교 (n={n}) ==========")
    logger.info(f"  재구성 파싱 성공: {reform_ok}/{n}")
    for name in sorted(hits, key=lambda x: -hits[x]):
        logger.info(f"  {name:<26} {hits[name]:>3}/{n} ({hits[name] / n * 100:5.1f}%)")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="쿼리 수 제한(스모크 테스트용)")
    main(limit=parser.parse_args().limit)
