# eval/label/fbcheck_binary.py
"""2진 FB-Check 확증 ① — GPT-4o 라벨, 사건 단위 준거 (사전 등록: eval/paper_plan.md §7-16, 2026-10-10, API 0)

E1과 같은 모집단·채점(`label_ablation`)에서 2진 규칙(등급 무시)과 기존 규칙(CLEAN)을 비교한다.

    .venv/bin/python -m eval.label.fbcheck_binary
"""

import json

import numpy as np

from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json
from eval.label.label_ablation import _doc_of, _labels, _ref_by_doc, _score_rows, filter_diff_ci, random_keep_baseline, score_vector

logger = load_logger("fbcheck_binary.log")
CLEAN = PROJECT_ROOT / "data/fb_check/clean.jsonl"
NOISE = PROJECT_ROOT / "data/fb_check/noise.jsonl"
OUT = PROJECT_ROOT / "data/eval/label/fbcheck_binary.json"
MARGIN_PP = -2.0


def binary_keep(r: dict) -> bool:
    """§7-16 2진 규칙. 등급은 보지 않는다."""
    f, v = set(r.get("forward_articles") or []), set(r.get("verify_articles") or [])
    if not f:
        return not v
    return bool(r.get("snippet_exists")) and bool(f & v)


def main() -> None:
    ref = _ref_by_doc()
    _score_rows._ref = ref
    rows = []
    for r in load_jsonl(CLEAN) + load_jsonl(NOISE):
        doc = _doc_of(r.get("chunk_id", ""))
        if r.get("source") != "ftc_case" or doc not in ref:
            continue
        rows.append({"doc": doc, "forward": _labels(r.get("forward_articles")),
                     "orig": r.get("status") == "CLEAN", "bin": binary_keep(r)})
    scores, has = score_vector(rows, "forward")
    docs = [r["doc"] for r in rows]
    k_orig = np.array([r["orig"] for r in rows])
    k_bin = np.array([r["bin"] for r in rows])
    rng = np.random.default_rng(42)

    main1 = random_keep_baseline(scores, has, k_bin, rng, groups=docs)
    main2 = filter_diff_ci(scores, has, k_bin, k_orig, rng, groups=docs)
    orig_strat = random_keep_baseline(scores, has, k_orig, rng, groups=docs)
    bin_unstrat = random_keep_baseline(scores, has, k_bin, rng)
    ok1 = main1["perm_p"] < 0.05
    ok2 = main2["ci95_pp"][0] > MARGIN_PP
    rep = {"rule": "eval/paper_plan.md §7-16 ①", "labeler": "gpt-4o (운영 FB-Check)",
           "n": {"records": len(rows), "forward_nonempty": int(has.sum()),
                 "kept_orig_scored": int((k_orig & has).sum()), "kept_bin_scored": int((k_bin & has).sum())},
           "F1_pp": {"all": round(float(scores[has].mean()) * 100, 2),
                     "orig": round(float(scores[k_orig & has].mean()) * 100, 2),
                     "bin": round(float(scores[k_bin & has].mean()) * 100, 2)},
           "주1_bin_vs_docstrat_random": main1, "주2_bin_minus_orig": main2, "margin_pp": MARGIN_PP,
           "보고_orig_vs_docstrat_random": orig_strat, "보고_bin_vs_unstrat_random": bin_unstrat,
           "verdict": "2진 규칙 확증(GPT-4o)" if (ok1 and ok2) else f"미확증 (주1 {'통과' if ok1 else '실패'}, 주2 {'통과' if ok2 else '실패'})"}
    save_json(rep, OUT)
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
