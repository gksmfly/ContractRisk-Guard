# eval/analysis/label_problem_types.py
"""라벨 문제 유형별 비율 — 잘못된 라벨 / 누락된 라벨 / 혼합된 라벨 (보고용, 사전 등록 아님, 2026-10-13, 학습·API 0)

AI Hub 5,150건(위반 2,575 + 공정 2,575, 조항 단위 전문가 정답)에 대해 Qwen 첫 판정 라벨을 정답과 맞춰 유형으로 나눈다.
FB-Check가 각 유형을 얼마나 걸러내는지(§7-14 제거율)를 함께 적어 "기존 방법으로 해결하기 어려운 유형"을 본다.

    .venv/bin/python -m eval.analysis.label_problem_types
"""

import json

import numpy as np

from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("label_problem_types.log")
OUT = PROJECT_ROOT / "data/eval/analysis/label_problem_types.json"
rng = np.random.default_rng(42)


def ci(groups, fn, n=2000):
    idx = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    cl = [np.array(v) for v in idx.values()]
    b = [fn(np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])) for _ in range(n)]
    return [round(float(x) * 100, 1) for x in np.percentile(b, [2.5, 97.5])]


def kind(F: set, ref: set, viol: bool) -> str:
    if not viol:
        return "정상(공정·빈 라벨)" if not F else "잘못된 라벨(위반 아닌데 위반 라벨)"
    if not F:
        return "누락된 라벨(위반인데 빈 라벨)"
    if F == ref:
        return "정상(조 라벨 일치)"
    if not (F & ref):
        return "혼합: 조 전부 틀림(위반은 맞음)"
    extra, miss = F - ref, ref - F
    if extra and miss:
        return "혼합: 맞는 조 + 틀린 조 + 빠진 조"
    return "혼합: 맞는 조 + 틀린 조 덧붙임" if extra else "혼합: 맞는 조만 있고 일부 빠짐"


def main() -> None:
    rows = load_jsonl(PROJECT_ROOT / "data/eval/label/fbcheck_aihub_raw.jsonl")
    K = []
    for r in rows:
        F, ref = set(r["forward"]), set(r["ref"])
        K.append(kind(F, ref, r["label"] == 1))
    n = len(rows)
    groups = [r["key"].rsplit("#", 1)[0] for r in rows]
    cnt = {}
    for k, r in zip(K, rows):
        d = cnt.setdefault(k, {"n": 0, "removed": 0})
        d["n"] += 1
        d["removed"] += r["status"] != "CLEAN"
    types = {}
    for k, d in sorted(cnt.items(), key=lambda x: -x[1]["n"]):
        m = np.array([kk == k for kk in K])
        types[k] = {"n": d["n"], "pct_of_all": round(d["n"] / n * 100, 1),
                    "ci95": ci(groups, lambda i, mm=m: mm[i].mean()),
                    "fbcheck_removed_pct": round(d["removed"] / d["n"] * 100, 1)}
    viol = [r for r in rows if r["label"] == 1]
    fair = [r for r in rows if r["label"] == 0]
    # 라벨(조) 단위: 붙은 조의 정확도, 정답 조의 회수
    attached = [(a in set(r["ref"])) for r in rows for a in r["forward"]]
    ref_total = sum(len(r["ref"]) for r in viol)
    ref_hit = sum(len(set(r["forward"]) & set(r["ref"])) for r in viol)
    rep = {"rule": "보고용 분석 (사전 등록 아님)", "n_records": n, "n_violation": len(viol), "n_fair": len(fair),
           "record_types": types,
           "violation_clauses": {"exact_article_match_pct": round(sum(set(r["forward"]) == set(r["ref"]) for r in viol) / len(viol) * 100, 1),
                                 "empty_label_pct": round(sum(not r["forward"] for r in viol) / len(viol) * 100, 1)},
           "fair_clauses": {"wrong_violation_label_pct": round(sum(bool(r["forward"]) for r in fair) / len(fair) * 100, 1)},
           "label_level": {"attached_articles": len(attached), "correct_pct": round(float(np.mean(attached)) * 100, 1),
                           "true_articles": ref_total, "recovered_pct": round(ref_hit / ref_total * 100, 1)}}
    save_json(rep, OUT)
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
