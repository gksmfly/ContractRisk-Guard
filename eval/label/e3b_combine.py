# eval/label/e3b_combine.py
"""FB-Check + CleanLab 결합 — 탐색 (사전 등록: eval/paper_plan.md §7-5, 2026-10-07, API 0)

주 판정은 **수와 무관한** 질문이다: FB-Check CLEAN 안에서 CleanLab이 남긴 것과 제거한 것의 F1 차(문서 층화 순열).
결합 방식 중 주 결과는 합집합(둘 중 하나라도 의심하면 제거)이며, 결과를 보고 바꾸지 않는다.

    .venv/bin/python -m eval.label.e3b_combine
"""

import json

import numpy as np

from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json
from eval.label.label_ablation import (
    _doc_of,
    _f1,
    _labels,
    _ref_by_doc,
    filter_diff_ci,
    random_keep_baseline,
)

logger = load_logger("e3b_combine.log")

CLEAN   = PROJECT_ROOT / "data/fb_check/clean.jsonl"
NOISE   = PROJECT_ROOT / "data/fb_check/noise.jsonl"
REMOVED = PROJECT_ROOT / "data/eval/label/e3_removed_ids.json"
OUT     = PROJECT_ROOT / "data/eval/label/e3b_combine.json"
BOOT_N  = 10_000


def _within(scores, has, pop, keep_sub, docs, rng) -> dict:
    """pop 안에서 keep_sub가 남긴 것 vs 버린 것. 문서 층화 순열로 p값(남긴 쪽 F1이 같은 문서 무작위보다 높은가)."""
    idx = np.where(pop)[0]
    s, h, k = scores[idx], has[idx], keep_sub[idx]
    d = [docs[i] for i in idx]
    kept, dropped = s[k & h], s[~k & h]
    r = random_keep_baseline(s, h, k, rng, BOOT_N, groups=d)
    return {"n_pop": int(len(idx)), "n_kept": int(k.sum()), "n_dropped": int((~k).sum()),
            "f1_kept_pp": round(float(kept.mean()) * 100, 2), "f1_dropped_pp": round(float(dropped.mean()) * 100, 2),
            "kept_minus_dropped_pp": round(float(kept.mean() - dropped.mean()) * 100, 2),
            "docstrat_gain_pp": round(r["diff_pp"], 2), "docstrat_perm_p": r["perm_p"],
            "verdict": "보완 정보 있음" if (r["perm_p"] < 0.05 and r["diff_pp"] > 0) else "미판정"}


def main() -> None:
    ref = _ref_by_doc()
    rows = []
    for path, status in ((CLEAN, "CLEAN"), (NOISE, "NOISE")):
        for r in load_jsonl(path):
            if r.get("source") == "ftc_case" and _doc_of(r.get("chunk_id", "")) in ref:
                rows.append({"chunk_id": r["chunk_id"], "doc": _doc_of(r["chunk_id"]), "status": status,
                             "forward": _labels(r.get("forward_articles"))})
    cl_removed = set(json.loads(REMOVED.read_text(encoding="utf-8"))["cleanlab_removed"])
    docs = [r["doc"] for r in rows]
    has = np.array([bool(r["forward"]) for r in rows])
    scores = np.array([_f1(r["forward"], ref[r["doc"]]) if r["forward"] else 0.0 for r in rows])
    fb_keep = np.array([r["status"] == "CLEAN" for r in rows])
    cl_keep = np.array([r["chunk_id"] not in cl_removed for r in rows])
    assert int((~cl_keep).sum()) == len(cl_removed), "CleanLab 제거 ID가 모집단과 맞지 않는다"
    rng = np.random.default_rng(42)

    union_keep = fb_keep & cl_keep          # 둘 중 하나라도 의심하면 제거
    inter_keep = fb_keep | cl_keep          # 둘 다 의심할 때만 제거
    overlap = int((~fb_keep & ~cl_keep).sum())

    def vs_random(keep):
        a = random_keep_baseline(scores, has, keep, rng, BOOT_N)
        b = random_keep_baseline(scores, has, keep, rng, BOOT_N, groups=docs)
        return {"n_keep": a["n_keep"], "n_scored": a["n_scored"], "f1_pp": round(a["mean_f1"] * 100, 2),
                "gain_vs_random_pp": round(a["diff_pp"], 2), "gain_docstrat_pp": round(b["diff_pp"], 2),
                "docstrat_p": b["perm_p"]}

    rep = {
        "rule": "eval/paper_plan.md §7-5",
        "population": f"ftc_case & 근거 있음 {len(rows)} (FB-Check CLEAN {int(fb_keep.sum())}, CleanLab 제거 {len(cl_removed)})",
        "overlap": {"both_removed": overlap, "cl_removed_also_fb_noise_pct": round(overlap / len(cl_removed) * 100, 1)},
        "★_within_fb_clean_cleanlab": _within(scores, has, fb_keep, cl_keep, docs, rng),
        "within_cl_kept_fbcheck": _within(scores, has, cl_keep, fb_keep, docs, rng),
        "filters": {"fbcheck": vs_random(fb_keep), "cleanlab": vs_random(cl_keep),
                    "union(주)": vs_random(union_keep), "intersection": vs_random(inter_keep)},
        "union_minus_fbcheck_docclust": filter_diff_ci(scores, has, union_keep, fb_keep, rng, 5000, groups=docs),
        "note": "합집합−FB-Check 차는 남긴 수가 달라 참고만. 판정은 ★(수와 무관)",
    }
    save_json(rep, OUT)
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
