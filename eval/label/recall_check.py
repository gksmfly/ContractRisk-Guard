# eval/label/recall_check.py
"""누락 검사(Recall-Check) 가능성 확인 — 데이터 수준 오류 (사전 등록: eval/paper_plan.md §7-19, 2026-10-12, 학습·API 0)

FB-Check는 붙은 라벨만 검사한다. 빈 라벨을 다른 계열 모델(인코더)로 교차 확인해 놓친 위반을 되살리는지 본다.
§7-14 행(Qwen forward + FB-Check) 중 AI Hub 평가 절반 파일만, 인코더 예측은 aihub_eval 캐시를 쓴다.

    .venv/bin/python -m eval.label.recall_check
"""

import json

import numpy as np

from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json
from eval.label.fbcheck_aihub import RAW, population
from eval.model.aihub_eval import CACHE, testset

logger = load_logger("recall_check.log")
SPLIT = PROJECT_ROOT / "data/eval/model/aihub_split.json"
OUT = PROJECT_ROOT / "data/eval/label/recall_check.json"
PRIMARY_CACHE, OTHER = "CNA_s7", ["CNA_s42", "CNA_s123"]
BOOT = 5000


def _cluster_ci(groups: list[str], fn) -> list[float]:
    idx: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    cl = [np.array(v) for v in idx.values()]
    rng = np.random.default_rng(42)
    b = [fn(np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])) for _ in range(BOOT)]
    return [round(float(x) * 100, 2) for x in np.percentile(b, [2.5, 97.5])]


def evaluate(cache: str, rows: list[dict], enc_flag_all: np.ndarray) -> dict:
    y = np.array([r["label"] for r in rows])
    g = [r["key"].rsplit("#", 1)[0] for r in rows]
    lab0 = np.array([bool(r["forward"]) for r in rows])           # P0 라벨: 위반 지목 여부
    keep1 = np.array([r["status"] == "CLEAN" for r in rows])      # P1 유지
    enc = enc_flag_all
    lab2 = lab0 | (~lab0 & enc)                                   # P2: 빈 라벨 + 인코더 위반 → 위반

    def err(i, keep, lab):
        k = keep[i]
        return float((lab[i][k] != (y[i][k] == 1)).mean())

    def fp(i, keep, lab):
        k = keep[i] & (y[i] == 0)
        return float(lab[i][k].mean()) if k.any() else 0.0

    def fn(i, keep, lab):
        k = keep[i] & (y[i] == 1)
        return float((~lab[i][k]).mean()) if k.any() else 0.0

    all_ = np.arange(len(rows))
    allkeep = np.ones(len(rows), bool)
    P = {"P0": (allkeep, lab0), "P1": (keep1, lab0), "P2": (keep1, lab2)}
    summ = {k: {"kept": int(kp.sum()), "error_pp": round(err(all_, kp, lb) * 100, 2),
                "false_alarm_pp": round(fp(all_, kp, lb) * 100, 2), "missed_pp": round(fn(all_, kp, lb) * 100, 2)}
            for k, (kp, lb) in P.items()}
    d = err(all_, keep1, lab2) - err(all_, keep1, lab0)
    ci = _cluster_ci(g, lambda i: err(i, keep1, lab2) - err(i, keep1, lab0))
    # 보조: 빈 라벨(P1 유지) 중 인코더 위반의 정밀도 − 기저율
    empty = keep1 & ~lab0

    def prec_minus_base(i):
        e = empty[i]
        f = e & enc[i]
        return (float(y[i][f].mean()) if f.any() else 0.0) - float(y[i][e].mean())
    ci2 = _cluster_ci(g, prec_minus_base)
    flagged = empty & enc
    return {"cache": cache, "pipelines": summ,
            "주_P2_minus_P1_error_pp": round(d * 100, 2), "주_ci95_pp": ci, "주_pass": ci[1] < 0,
            "보조_flag_precision_pp": round(float(y[flagged].mean()) * 100, 2) if flagged.any() else None,
            "보조_base_rate_pp": round(float(y[empty].mean()) * 100, 2),
            "보조_precision_minus_base_ci95_pp": ci2, "보조_pass": ci2[0] > 0,
            "n_empty_kept": int(empty.sum()), "n_flagged": int(flagged.sum()),
            "recovered_missed": int((flagged & (y == 1)).sum()), "missed_total_in_P1": int((empty & (y == 1)).sum()),
            "new_false_alarms": int((flagged & (y == 0)).sum())}


def main() -> None:
    items = testset()
    pos = {(it["group"], it["text"]): i for i, it in enumerate(items)}
    raw = {r["key"]: r for r in load_jsonl(RAW)}
    train_files = set(json.loads(SPLIT.read_text(encoding="utf-8"))["train_files"])
    rows, tidx = [], []
    for p in population():
        if p["group"] in train_files:
            continue
        r = raw[p["key"]]
        rows.append(r)
        tidx.append(pos[(p["group"], p["text"])])
    tidx = np.array(tidx)

    def flags(cache: str) -> np.ndarray:
        z = np.load(CACHE / f"{cache}.npz")
        return (z["P"][tidx] >= z["thr"]).any(1)

    res = {c: evaluate(c, rows, flags(c)) for c in [PRIMARY_CACHE, *OTHER]}
    main_ = res[PRIMARY_CACHE]
    verdict = ("누락 검사가 데이터 오류를 줄인다" if main_["주_pass"] else "미판정") + \
              (" · 교차 확인이 무작위 이상" if main_["보조_pass"] else " · 교차 확인 무작위와 구분 안 됨")
    rep = {"rule": "eval/paper_plan.md §7-19", "n_rows": len(rows), "n_violation": int(sum(r["label"] for r in rows)),
           "primary": main_, "other_seeds": {c: res[c] for c in OTHER}, "verdict": verdict}
    save_json(rep, OUT)
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
