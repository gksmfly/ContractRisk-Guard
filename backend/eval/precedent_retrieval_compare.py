# backend/eval/precedent_retrieval_compare.py
"""
판례 검색 정확도 측정 + 법원 심급 가중치 스윕 — 코퍼스의 90%를 처음으로 재는 스크립트.

`evidence_selection_agent`의 법원 심급 가중치에 버그 두 개가 있다(점검으로 확인):

  A-1. `_COURT_WEIGHT`가 `"고등법원"` 완전일치를 요구하는데 DB의 court 값은
       `서울고법`·`부산고등법원`·`대전고법(청주)` 등 58종이라 **한 번도 매칭되지 않는다**
       → 고등법원급 3,452청크(11.4%)가 전부 가산 0.
  A-2. `대법원` 가산 0.10은 RRF 1~51위 전체 점수 폭(0.00738)의 **13.5배**라
       "동률에 가까운 상위권 사이의 소폭 가산"(코드 주석)이 아니라 전면 재정렬이다.
       대법원이 판례의 68.5%라 사실상 "대법원 순 정렬"이 된다.

이 스크립트는 **고치기 전 수치를 먼저 확보**하고(현행 그대로 측정), 그 위에서 수정안과
가중치 후보를 같은 쿼리로 비교한다. 개선폭을 주장하려면 before가 있어야 한다.

측정 조건:
  rrf_only    — 재랭킹 없음(RRF 순서 그대로). 가중치가 도움이 되는지의 기준선
  current     — **2026-08-16 수정 이전** 동작 재현 (버그 포함 상태, `_legacy_court_boost`)
  fixed_w{X}  — A-1 수정(문자열 정규화 매칭) + 대법원 가산 X, 고등법원 X/2
                X ∈ {0, 0.0003, 0.001, 0.01, 0.1}
                X=0은 "가중치 폐지"이며, 이게 이기면 가중치를 없앤다 — 도메인 지식이라도
                측정에서 지면 뺀다(Cross-Encoder를 같은 기준으로 기각한 전례와 동일).

검색은 운영 코드(`retrieval.fetch_candidates`)를 그대로 재사용한다 — 평가와 운영이 다른
코드를 타면 "평가는 top-5인데 운영은 top-2"(실사용 정확도 33%가 아니라 18%였던) 같은
불일치가 또 생긴다. 재랭킹은 운영에서 제거됐으므로 수정 전 동작을 이 파일 안에
`_legacy_court_boost`로 보존해 before/after 비교를 계속 재현할 수 있게 한다.

매 쿼리마다 운영 노드(`evidence_selection_node`)가 판례를 RRF 순서 그대로 쓰는지도
확인한다(`sanity_mismatch`) — 가중치를 되살리는 변경이 들어오면 여기서 잡힌다.

실행:
    .venv/bin/python -m backend.eval.precedent_retrieval_compare --pool 30 --n 100
    .venv/bin/python -m backend.eval.precedent_retrieval_compare --pool 6  --n 100  # 운영 조건
"""

import argparse
from math import comb
from typing import Any

from backend.agents.evidence_selection_agent import _FINAL_K, evidence_selection_node
from backend.api.services.retrieval import fetch_candidates
from backend.eval.precedent_ground_truth import build_precedent_gold, build_reference_index
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("precedent_retrieval_compare.log")
OUT_PATH = PROJECT_ROOT / "data/eval/precedent_retrieval_report.json"

_CUTOFFS = (2, 5, 10, 20)
_WEIGHT_CANDIDATES = (0.0, 0.0003, 0.001, 0.01, 0.1)

# 2026-08-16 수정 **이전**의 동작을 재현하기 위해 보존한 상수/로직.
# 운영 코드(`evidence_selection_agent`)에서는 제거됐지만, before/after 비교를 다시 돌릴 수
# 있어야 "고쳐서 좋아졌다"는 주장이 재현 가능하다. 운영 코드가 이걸 다시 import하면 안 된다.
_LEGACY_COURT_WEIGHT = {"대법원": 0.10, "고등법원": 0.05}
_RRF_K = 60  # retrieval._fuse_reciprocal_rank과 동일한 k


def _legacy_court_boost(court: str) -> float:
    """수정 전 동작 — 완전일치 조회라 DB의 `서울고법` 등 58종 표기에는 걸리지 않는다(A-1)."""
    return _LEGACY_COURT_WEIGHT.get(court or "", 0.0)


def fixed_court_boost(court: str, top_weight: float) -> float:
    """A-1 수정판 — 완전일치 대신 심급 패턴으로 판별한다.

    DB의 court 값 58종에는 `서울고법`·`부산고등법원`·`대전고법(청주)`처럼 표기가 제각각이라
    `"고등법원"` 키 하나로는 아무것도 안 걸린다. 접미 패턴으로 심급을 본다.
    """
    if not court:
        return 0.0
    if "대법원" in court:
        return top_weight
    if "고법" in court or "고등법원" in court:
        return top_weight / 2
    return 0.0


def _rerank_with(candidates: list[dict], boost_fn: Any) -> list[dict]:
    """수정 전과 동일한 점수 체계(RRF 순위 → 점수 환산 + 가산)에 가산 함수만 주입한다."""
    def score(indexed: tuple[int, dict]) -> float:
        rank, cand = indexed
        return 1.0 / (_RRF_K + rank + 1) + boost_fn((cand["metadata"] or {}).get("court", ""))

    return [c for _, c in sorted(enumerate(candidates), key=score, reverse=True)]


def _hit_at(ranked: list[dict], gold_docs: set, chunk2doc: dict, k: int) -> bool:
    return any(chunk2doc.get(c["chunk_id"]) in gold_docs for c in ranked[:k])


def _mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    B = sum(1 for x, y in zip(a, b) if x and not y)
    C = sum(1 for x, y in zip(a, b) if y and not x)
    n = B + C
    if n == 0:
        return B, C, 1.0
    k = min(B, C)
    return B, C, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main(n: int = 100, pool: int = 30, mode: str = "rarest") -> None:
    index = build_reference_index()
    chunk2doc = index["chunk2doc"]
    gold_rows = build_precedent_gold(n_cases=n, mode=mode)

    configs = {"rrf_only": None, "current": "current"}
    for w in _WEIGHT_CANDIDATES:
        configs[f"fixed_w{w:g}"] = w

    hits = {name: {k: 0 for k in _CUTOFFS} for name in configs}
    # 쿼리별 적중을 **모든 cutoff**에 대해 남긴다 — @5만 저장하면 격차가 가장 큰 @2 구간을
    # 나중에 검정할 수 없다(운영은 top-2를 노출하므로 @2가 실사용에 가장 가깝다).
    per_query: dict[str, dict[int, list[bool]]] = {nm: {k: [] for k in _CUTOFFS} for nm in configs}
    evaluated = 0
    rnd5_sum = 0.0
    sanity_mismatch = 0

    for i, row in enumerate(gold_rows):
        gold = set(row["gold_docs"])
        if not gold:
            continue  # 정답 판례가 없는 케이스는 채점 대상에서 제외
        evaluated += 1
        rnd5_sum += 1 - (1 - row["gold_chunk_ratio"]) ** 5

        candidates = fetch_candidates(row["clause"], top_k_per_source=pool).get("precedent", [])
        current_ranked = _rerank_with(candidates, _legacy_court_boost)  # 수정 전 동작 재현

        # 회귀 검증: 운영 노드가 판례를 **RRF 순서 그대로** 상위 K개 쓰는지 확인한다.
        # 법원 가중치를 되살리는 변경이 들어오면 여기서 즉시 잡힌다.
        produced = evidence_selection_node({
            "clause": row["clause"], "domain": "해지_조항",
            "retrieval_candidates": {"law": [], "precedent": candidates},
        })
        if len(produced.get("legal_basis", [])) != len(candidates[:_FINAL_K]):
            sanity_mismatch += 1

        for name, cfg in configs.items():
            if cfg is None:
                ranked = candidates
            elif cfg == "current":
                ranked = current_ranked
            else:
                ranked = _rerank_with(candidates, lambda ct, w=cfg: fixed_court_boost(ct, w))
            for k in _CUTOFFS:
                h = _hit_at(ranked, gold, chunk2doc, k)
                hits[name][k] += h
                per_query[name][k].append(h)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  진행 {i + 1}/{len(gold_rows)} (채점 {evaluated}) | "
                + " ".join(f"{nm}@5={hits[nm][5]}" for nm in ("rrf_only", "current", "fixed_w0.001"))
            )

    rnd5 = rnd5_sum / max(evaluated, 1)
    result = {
        "n_queries": len(gold_rows), "n_evaluated": evaluated, "gold_mode": mode,
        "pool_per_source": pool, "random_hit_at5": rnd5,
        "sanity_mismatch": sanity_mismatch,
        "hits": {nm: {str(k): v for k, v in d.items()} for nm, d in hits.items()},
        "hit_rate": {nm: {str(k): v / evaluated for k, v in d.items()} for nm, d in hits.items()},
        "mcnemar_vs_current": {
            nm: {str(k): dict(zip(("b_this_only", "c_current_only", "p_value"),
                                  _mcnemar(per_query[nm][k], per_query["current"][k])))
                 for k in _CUTOFFS}
            for nm in configs if nm != "current"
        },
        # 사후 분석·다른 쌍의 검정을 위해 쿼리별 적중을 cutoff마다 남긴다
        # (집계만 저장하면 나중에 어떤 조합도 다시 검정할 수 없다)
        "per_query_hits": {nm: {str(k): v for k, v in d.items()} for nm, d in per_query.items()},
        "case_names": [r["case_name"] for r in gold_rows if r["gold_docs"]],
    }
    save_json(result, OUT_PATH)

    logger.info(f"===== 판례 검색 (gold={mode}, pool={pool}, 채점 {evaluated}건) =====")
    logger.info(f"  하네스 검증 — 현행 함수 재현 불일치: {sanity_mismatch}건 (0이어야 정상)")
    logger.info(f"  무작위 기저 hit@5: {rnd5 * 100:.1f}%")
    logger.info("  적중률 / (현행 대비 McNemar p)")
    logger.info(f"  {'설정':<14}" + "".join(f"{'@' + str(k):>17}" for k in _CUTOFFS))
    for nm in configs:
        cells = []
        for k in _CUTOFFS:
            rate = f"{hits[nm][k] / evaluated * 100:.1f}%"
            p = "" if nm == "current" else f" (p={result['mcnemar_vs_current'][nm][str(k)]['p_value']:.3g})"
            cells.append(f"{rate + p:>17}")
        logger.info(f"  {nm:<14}" + "".join(cells))
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--pool", type=int, default=30)
    p.add_argument("--gold", default="rarest", choices=["union", "intersection", "rarest"])
    a = p.parse_args()
    main(n=a.n, pool=a.pool, mode=a.gold)
