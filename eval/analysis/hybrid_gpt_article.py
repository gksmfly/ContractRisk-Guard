# eval/analysis/hybrid_gpt_article.py
"""KoELECTRA 게이트 + GPT 조 안내 조합 — 탐색 (2026-10-07, API 0)

GPT는 조 식별에서 KoELECTRA보다 약 5~7%p 앞서지만(§7-4) 조항 지목은 동률이고 300배 느리며 유료다.
그래서 **KoELECTRA가 짚은 조항에만 GPT를 붙이는** 조합이 얼마나 나은지 이미 있는 GPT 실행으로 잰다.
gold로 고른 값은 하나도 없다(KoELECTRA는 배포 임계값, GPT는 임계값 없음). 사전 등록이 아니라 탐색이다.

    방식   검토 필요 판정                 조 안내
    K      KoELECTRA                      KoELECTRA                      (현행)
    G      GPT                            GPT
    H1     KoELECTRA                      짚힌 조항만 GPT, GPT가 비우면 KoELECTRA
    H2     KoELECTRA ∧ GPT                GPT
    H3     KoELECTRA ∨ GPT                GPT, 비면 KoELECTRA

GPT 실행: 10-07 1·2회차 + FB-Check v5 forward (gold 255와 홀드아웃 300을 모두 가진 실행만). 실행별로 재고 평균한다.

    .venv/bin/python -m eval.analysis.hybrid_gpt_article
"""

import json
import re

import numpy as np

from backend.training.retrain import eval_sets
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json
from eval.analysis.confound_articles import _f1
from eval.model.e8b_matched import COMMON, _groups, _load

logger = load_logger("hybrid_gpt_article.log")

HOLDOUT = PROJECT_ROOT / "backend/training/negative_holdout_docs.json"
REPEAT  = PROJECT_ROOT / "data/eval/teacher_repeat"
OUT     = PROJECT_ROOT / "data/eval/analysis/hybrid_gpt_article.json"
REAL    = PROJECT_ROOT / "data/eval/real_tos"


def _gpt_runs(gold, neg) -> dict[str, tuple[list, list]]:
    runs = {}
    for k in (1, 2):
        rows = {r["key"]: r for r in load_jsonl(REPEAT / f"run{k}.jsonl")}
        runs[f"10-07 #{k}"] = ([frozenset(a for a in rows[f"gold|{i}"]["articles"] if a in COMMON) for i in range(len(gold))],
                               [frozenset(a for a in rows[f"neg|{r['chunk_id']}"]["articles"] if a in COMMON) for r in neg])
    norm = lambda t: re.sub(r"\s+", "", t or "")  # noqa: E731
    fb_rows = [r for p in ("clean", "noise") for r in load_jsonl(PROJECT_ROOT / f"data/fb_check/{p}.jsonl")]
    by_text = {norm(r["text"]): r for r in fb_rows if r.get("source") == "ftc_case"}
    by_id = {r["chunk_id"]: r for r in fb_rows}
    runs["FB-Check v5"] = ([frozenset(a for a in by_text[norm(g["text"])]["forward_articles"] if a in COMMON) for g in gold],
                           [frozenset(a for a in (by_id[r["chunk_id"]].get("forward_articles") or []) if a in COMMON) for r in neg])
    return runs


def _combine(mode: str, k: frozenset, g: frozenset) -> frozenset:
    if mode == "K":
        return k
    if mode == "G":
        return g
    if mode == "H1":
        return (g or k) if k else frozenset()
    if mode == "H2":
        return g if (k and g) else frozenset()
    if mode == "H3":
        return g or k
    raise ValueError(mode)


def main() -> None:
    gold, neg = eval_sets(json.loads(HOLDOUT.read_text(encoding="utf-8")))
    G = [frozenset(a for a in g["articles"] if a in COMMON) for g in gold]
    P, Pn, thr = _load("article_v2")
    Kg = [frozenset(a for a, b in zip(COMMON, row) if b) for row in (P >= thr)]
    Kn = [frozenset(a for a, b in zip(COMMON, row) if b) for row in (Pn >= thr)]
    runs = _gpt_runs(gold, neg)
    g_groups = _groups([g["doc_id"] for g in gold])
    n_groups = _groups([":".join(str(r["chunk_id"]).split(":")[:2]) for r in neg])

    def metrics(mode: str, Tg: list, Tn: list) -> dict[str, np.ndarray]:
        pred_g = [_combine(mode, k, t) for k, t in zip(Kg, Tg)]
        pred_n = [_combine(mode, k, t) for k, t in zip(Kn, Tn)]
        flagged = np.array([bool(p) for p in pred_g])
        f1 = np.array([_f1(p, y) if p else 0.0 for p, y in zip(pred_g, G)])
        # GPT 호출 비율: 이 방식이 GPT를 불러야 하는 조항 비율(gold+홀드아웃 기준, 실제 약관은 아래에서 따로)
        calls = {"K": 0.0, "G": 1.0, "H1": float(np.mean([bool(k) for k in Kg + Kn])),
                 "H2": float(np.mean([bool(k) for k in Kg + Kn])), "H3": 1.0}[mode]
        return {"recall": flagged.astype(float), "fpr": np.array([bool(p) for p in pred_n], dtype=float),
                "f1": f1, "f1_flagged": f1[flagged], "calls": calls}

    modes = ["K", "G", "H1", "H2", "H3"]
    per = {m: [metrics(m, Tg, Tn) for Tg, Tn in runs.values()] for m in modes}
    rng = np.random.default_rng(42)

    def diff_ci(a: str, b: str, key: str, groups) -> dict:
        D = [x[key] - y[key] for x, y in zip(per[a], per[b])]
        m = np.mean(D, axis=0)
        boot = [m[np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(3000)]
        se = float(np.sqrt(np.var(boot) + np.var([d.mean() for d in D], ddof=1) / len(D)))
        pt = float(m.mean())
        lo, hi = pt - 1.96 * se, pt + 1.96 * se
        better_low = key == "fpr"
        verdict = "미판정" if lo <= 0 <= hi else (f"{a} 우위" if (lo > 0) != better_low else f"{b} 우위")
        return {"diff_pp": round(pt * 100, 2), "ci95_pp": [round(lo * 100, 2), round(hi * 100, 2)], "verdict": verdict}

    summary = {m: {"recall": round(float(np.mean([r["recall"].mean() for r in per[m]])) * 100, 1),
                   "holdout_fpr": round(float(np.mean([r["fpr"].mean() for r in per[m]])) * 100, 1),
                   "article_f1_all": round(float(np.mean([r["f1"].mean() for r in per[m]])) * 100, 1),
                   "article_f1_flagged": round(float(np.mean([r["f1_flagged"].mean() for r in per[m]])) * 100, 1),
                   "gpt_call_share": round(per[m][0]["calls"] * 100, 1)} for m in modes}
    contrasts = {f"{a}-K": {"recall": diff_ci(a, "K", "recall", g_groups), "fpr": diff_ci(a, "K", "fpr", n_groups),
                            "article_f1": diff_ci(a, "K", "f1", g_groups)} for a in ("G", "H1", "H2", "H3")}
    contrasts["H1-G"] = {"article_f1": diff_ci("H1", "G", "f1", g_groups), "recall": diff_ci("H1", "G", "recall", g_groups),
                         "fpr": diff_ci("H1", "G", "fpr", n_groups)}

    # 실제 약관에서의 GPT 호출 비율(비용 근사): H1·H2는 KoELECTRA가 짚은 조항만 GPT로 보낸다
    from backend.training.retrain import predict
    real = [c["text"] for f in sorted(REAL.glob("*.json")) for c in json.loads(f.read_text(encoding="utf-8"))["clauses"]]
    Pr, thr_r, _ = predict(PROJECT_ROOT / "models/article_v2", real, 0)
    real_share = float((Pr >= thr_r).any(1).mean())

    rep = {"note": "탐색 — gold로 고른 값 없음. GPT 3회 실행 평균, CI는 문서 군집 + 실행 분산",
           "n_gold": len(gold), "n_holdout": len(neg), "gpt_runs": list(runs),
           "summary": summary, "contrasts": contrasts,
           "real_tos_gpt_call_share_for_H1": round(real_share * 100, 1), "real_tos_n_clauses": len(real)}
    save_json(rep, OUT)
    for m, s in summary.items():
        logger.info(f"  {m:<3} 재현 {s['recall']:5.1f}% 오경보 {s['holdout_fpr']:5.1f}% 조F1(전체) {s['article_f1_all']:5.1f}% "
                    f"조F1(짚은 것) {s['article_f1_flagged']:5.1f}% GPT 호출 {s['gpt_call_share']:5.1f}%")
    for c, v in contrasts.items():
        logger.info(f"  {c:<6} " + " | ".join(f"{k} {x['diff_pp']:+.1f} {x['ci95_pp']} {x['verdict']}" for k, x in v.items()))
    logger.info(f"  실제 약관에서 H1이 GPT로 보내는 조항 비율: {real_share * 100:.1f}% ({len(real)}조항)")


if __name__ == "__main__":
    main()
