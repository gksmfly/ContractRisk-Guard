# backend/eval/law_router_compare.py
"""
법령 라우팅 방식 비교 — LLM 라우터가 상수 기준선을 이기는가.

## 왜 만들었나

`raptor_lite_compare.py`는 로컬 EXAONE이 top-2 법령을 예측해 그 파티션만 검색하는
방식을 RRF(필터 없음)와 비교해 "8% → 33%, p<0.0001"이라는 결과를 냈다. 그런데
**상수 라우터와는 비교하지 않았다.**

이 평가셋은 FTC 불공정약관 의결서에서 만들었으므로 100건 **전부** 정답에 약관규제법이
들어 있다. 즉 "항상 약관규제법을 고른다"는 라우터의 정답률이 100%인데, EXAONE은
78%였다. 라우팅할 대상이 애초에 없는 문제였다.

## 무엇을 비교하나

| 구성 | 검색 파티션 |
|---|---|
| `rrf` | 필터 없음(법령 전체 경쟁) |
| `exaone` | EXAONE이 예측한 top-2 (기록된 예측 재사용 — LLM 호출 0) |
| `fixed_ykk` | 약관규제법 고정 |
| `fixed_ykk_min` | 약관규제법 + 민법 고정 |
| `ykk_plus_exaone` | 약관규제법 + EXAONE 예측(교체가 아니라 추가) |

`exaone` 구성은 `data/eval/raptor_lite_vs_rrf_report.json`의 `per_query.predicted_laws`를
그대로 읽는다 — 라우팅을 다시 돌리지 않으므로 GPU도 API도 쓰지 않고, 같은 예측 위에서
검색 조건만 바꿔 비교하게 되어 페어드 검정이 성립한다.

## 결과 (100건, 후보 20/법령)

    구성                     top-1  top-2  top-3  top-5  top-10  top-20
    약관규제법 고정             35%    43%    54%    66%     80%     88%
    약관규제법 + EXAONE 추가     18%    27%    32%    40%     52%     56%

    (후보 8/법령, top-5) RRF 18% · EXAONE 37% · 약관규제법 65% · 약관규제법+민법 23%

    약관규제법 고정 vs EXAONE: 고정만 맞음 28 / EXAONE만 맞음 0, McNemar p<0.00001

두 가지가 드러난다:

1. **LLM 라우팅이 상수에 완패한다.** EXAONE이 단독으로 맞힌 케이스가 한 건도 없다.
   조항 소재(전자상거래·방문판매 등)에 끌려 23건에서 다른 법을 골랐다.
2. **파티션을 하나만 더해도 크게 나빠진다.** 약관규제법(49조) 옆에 민법(1,337조)을
   붙이면 65% → 23%. 큰 파티션의 후보가 RRF 융합에서 정답을 top-K 밖으로 밀어낸다.
   EXAONE 예측을 *추가*만 해도 마찬가지다(40%).

## 한계

평가셋이 전부 FTC 약관 사건이라 약관규제법이 100% 정답이다. 따라서 이 결과는
"라우팅이 일반적으로 무용하다"가 아니라 **"이 도메인에서는 약관규제법이 항상
관련되므로 라우팅할 것이 없다"**로 읽어야 한다. 약관규제법이 정답이 아닌 사례가
포함된 평가셋이 생기면 다시 재야 한다.

반영: `backend/agents/retrieval_strategy_agent.py::_PRIMARY_LAW`,
      `backend/agents/evidence_selection_agent.py::_FINAL_K`

실행:
    PYTHONPATH=. EMBED_DEVICE=cuda:1 .venv/bin/python -m backend.eval.law_router_compare
"""

import json
import math

from backend.api.services.retrieval import _get_cached_embedder, _reciprocal_rank_fusion
from backend.db.connection import get_conn
from backend.db.loader import embed_texts
from backend.eval.domain_filter_compare import (
    _dense_with_score,
    _search_dense,
    _search_sparse,
    _sparse_with_score,
)
from backend.eval.lightrag_compare import _N_QUERIES, LAWS_PATH, build_ground_truth
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("law_router_compare.log")

RAPTOR_REPORT = PROJECT_ROOT / "data/eval/raptor_lite_vs_rrf_report.json"
OUT_PATH      = PROJECT_ROOT / "data/eval/law_router_compare_report.json"

YKK = "약관의 규제에 관한 법률"
MIN = "민법"

_CANDIDATE_K = 20   # 법령당 dense/sparse 후보 수
_KS = (1, 2, 3, 5, 10, 20)


_SUBSTANTIVE = (6, 14)   # 약관규제법에서 "이 조항이 불공정한가"를 정하는 실질 규범 구간


def _dense_range(cur, law_name: str, vec_literal: str, top_k: int, rng: tuple[int, int]) -> list[tuple]:
    """`_dense_with_score`와 같은 조건에 조 번호 범위 필터만 더한다."""
    cur.execute(
        """
        SELECT chunk_id, source, metadata, text, 1 - (embedding <=> %s::vector) AS score
        FROM chunks
        WHERE source = 'law' AND metadata->>'law_name' = %s
          AND (metadata->>'article_no') ~ '^[0-9]+$'
          AND (metadata->>'article_no')::int BETWEEN %s AND %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (vec_literal, law_name, rng[0], rng[1], vec_literal, top_k),
    )
    return cur.fetchall()


def _sparse_range(cur, law_name: str, query_text: str, top_k: int, rng: tuple[int, int],
                  threshold: float = 0.10) -> list[tuple]:
    """`_sparse_with_score`와 같은 조건(임계값 0.10, `%%` 연산자)에 조 범위 필터만 더한다."""
    cur.execute("SET pg_trgm.similarity_threshold = %s", (threshold,))
    cur.execute(
        """
        SELECT chunk_id, source, metadata, text, similarity(text, %s) AS score
        FROM chunks
        WHERE source = 'law' AND metadata->>'law_name' = %s AND text %% %s
          AND (metadata->>'article_no') ~ '^[0-9]+$'
          AND (metadata->>'article_no')::int BETWEEN %s AND %s
        ORDER BY score DESC
        LIMIT %s
        """,
        (query_text, law_name, query_text, rng[0], rng[1], top_k),
    )
    return cur.fetchall()


def _load_exaone_predictions() -> dict[str, list[str]]:
    """기록된 EXAONE 라우팅 예측을 읽는다(없으면 exaone 구성은 건너뛴다)."""
    if not RAPTOR_REPORT.exists():
        logger.warning(f"  {RAPTOR_REPORT} 없음 — EXAONE 구성은 제외한다")
        return {}
    rep = json.loads(RAPTOR_REPORT.read_text(encoding="utf-8"))
    return {r["case_name"]: r["predicted_laws"] for r in rep["per_query"]}


def _first_gold_rank(cur, embedder, query_text: str, laws: list[str] | None,
                     gold: set[tuple], candidate_k: int,
                     article_range: tuple[int, int] | None = None) -> int | None:
    """정답 조문이 RRF 순위에서 처음 등장하는 위치(1-base). 못 찾으면 None.

    `laws`가 None이면 파티션 필터 없이 법령 전체에서 검색한다(RRF 기준선).
    """
    vec = embed_texts(embedder, [query_text], prefix="query: ")[0]
    lit = "[" + ",".join(repr(x) for x in vec) + "]"

    if laws is None:
        # 필터 없음 — 법령 전체를 한 풀에서 경쟁시킨다(현행 production 기준선).
        ranked = _reciprocal_rank_fusion(
            _search_dense(cur, ["law"], lit, candidate_k),
            _search_sparse(cur, ["law"], query_text, candidate_k, 0.10),
        )
    else:
        # 파티션별로 따로 top-k를 뽑은 뒤 점수순으로 합친다 — 하나의 IN절로 묶으면
        # 큰 법령이 공통 top-k를 독점해 작은 법령의 조문이 아예 안 나온다.
        dense, sparse = [], []
        for law in laws:
            if article_range:
                dense.extend(_dense_range(cur, law, lit, candidate_k, article_range))
                sparse.extend(_sparse_range(cur, law, query_text, candidate_k, article_range))
            else:
                dense.extend(_dense_with_score(cur, law, lit, candidate_k))
                sparse.extend(_sparse_with_score(cur, law, query_text, candidate_k))
        ranked = _reciprocal_rank_fusion(
            [r[:4] for r in sorted(dense,  key=lambda x: x[4], reverse=True)],
            [r[:4] for r in sorted(sparse, key=lambda x: x[4], reverse=True)],
        )
    seq = [(c["metadata"].get("law_name"), c["metadata"].get("article_no")) for c in ranked]
    return next((i + 1 for i, p in enumerate(seq) if p in gold), None)


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """페어드 정확 검정. b가 개선한 건수 / a만 맞은 건수 / p값."""
    n_b = sum(1 for x, y in zip(a, b) if y and not x)
    n_c = sum(1 for x, y in zip(a, b) if x and not y)
    n = n_b + n_c
    if n == 0:
        return {"b_only": 0, "a_only": 0, "p_value": 1.0}
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(n_b, n_c) + 1)) / (2 ** n))
    return {"b_only": n_b, "a_only": n_c, "p_value": p}


def main() -> None:
    queries = build_ground_truth(load_jsonl(LAWS_PATH), n_cases=_N_QUERIES)
    exaone = _load_exaone_predictions()
    logger.info(f"  평가 쿼리 {len(queries)}건 | EXAONE 예측 기록 {len(exaone)}건")

    # 각 구성은 (파티션 선택 함수, 조 번호 범위) 쌍이다. 범위가 있으면 그 조문만 검색한다.
    configs: dict[str, tuple] = {
        "rrf":              (lambda q: None, None),          # 필터 없음
        "fixed_ykk":        (lambda q: [YKK], None),
        "fixed_ykk_min":    (lambda q: [YKK, MIN], None),
        "fixed_ykk_6_14":   (lambda q: [YKK], _SUBSTANTIVE),  # 실질 규범만
    }
    if exaone:
        configs["exaone"] = (lambda q: exaone.get(q["case_name"], []), None)
        configs["ykk_plus_exaone"] = (
            lambda q: [YKK] + [l for l in exaone.get(q["case_name"], []) if l != YKK], None
        )

    ranks: dict[str, list[int | None]] = {name: [] for name in configs}
    embedder = _get_cached_embedder()
    with get_conn() as conn, conn.cursor() as cur:
        for i, q in enumerate(queries, 1):
            gold = set(map(tuple, q["correct_pairs"]))
            for name, (pick, rng) in configs.items():
                laws = pick(q)
                if laws is not None and not laws:      # 라우팅이 아무것도 못 고른 경우
                    ranks[name].append(None)
                    continue
                ranks[name].append(
                    _first_gold_rank(cur, embedder, q["clause"], laws, gold, _CANDIDATE_K, rng)
                )
            if i % 20 == 0:
                logger.info(f"  진행 {i}/{len(queries)}")

    n = len(queries)
    hit = {name: {k: sum(1 for r in rs if r and r <= k) for k in _KS} for name, rs in ranks.items()}

    logger.info(f"========== 법령 라우팅 비교 (n={n}, 후보 {_CANDIDATE_K}/법령) ==========")
    logger.info("  " + f"{'구성':<20}" + "".join(f"top-{k:<6}" for k in _KS))
    for name in configs:
        logger.info("  " + f"{name:<20}" + "".join(f"{hit[name][k]:>4}%    " for k in _KS))

    tests = {}
    a5 = [bool(r and r <= 5) for r in ranks["fixed_ykk"]]
    b5 = [bool(r and r <= 5) for r in ranks["fixed_ykk_6_14"]]
    tests["fixed_ykk_6_14_vs_fixed_ykk@5"] = mcnemar(a5, b5)
    logger.info(f"  [top-5] 제6~14조만 vs 약관규제법 전체: {tests['fixed_ykk_6_14_vs_fixed_ykk@5']}")
    if "exaone" in ranks:
        for target in ("fixed_ykk", "fixed_ykk_6_14", "ykk_plus_exaone"):
            a = [bool(r and r <= 5) for r in ranks["exaone"]]
            b = [bool(r and r <= 5) for r in ranks[target]]
            tests[f"{target}_vs_exaone@5"] = mcnemar(a, b)
            logger.info(f"  [top-5] {target} vs exaone: {tests[f'{target}_vs_exaone@5']}")

    save_json({
        "n_queries": n, "candidate_k": _CANDIDATE_K, "ks": list(_KS),
        "hit_counts": hit, "mcnemar": tests,
        "first_gold_rank": {name: rs for name, rs in ranks.items()},
        "note": "exaone 구성은 raptor_lite_vs_rrf_report.json의 기록된 예측을 재사용 — LLM 호출 없음.",
    }, OUT_PATH)
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
