# eval/model/fresh_revalidation.py
"""새 시험 데이터로 교체 재검증 (사전 등록: eval/paper_plan.md §7-3, 2026-10-06)

    --prepare   새 시험 문서를 고정 규칙으로 뽑고, 그 문서를 뺀 학습 라벨 파일 두 개를 만든다
                data/eval/fresh/split.json · labels_NC.jsonl · labels_NRSN.jsonl
    --analyze   N-C(KoELECTRA) vs N-RSN(RoBERTa + 회사 말투 사본), 3시드 앙상블끼리 새 시험셋에서 판정

학습은 `eval.model.e8_downstream_ablation --arms NC NRSN`.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json, save_jsonl

logger = load_logger("fresh_revalidation.log")

FRESH   = PROJECT_ROOT / "data/eval/fresh"
CLEAN   = PROJECT_ROOT / "data/fb_check/clean.jsonl"
NOISE   = PROJECT_ROOT / "data/fb_check/noise.jsonl"
SN      = PROJECT_ROOT / "data/eval/style_aug/labels_SN.jsonl"
HOLDOUT = PROJECT_ROOT / "backend/training/negative_holdout_docs.json"
SEED_SELECT = 20261006
FTC_SHARE, N_SC = 0.20, 12


def _doc(cid: str) -> str:
    return ":".join(str(cid).split("#")[0].split(":")[:2])


def prepare() -> None:
    from backend.training.train_article import load_ftc_gold
    FRESH.mkdir(parents=True, exist_ok=True)
    held = set(json.loads(HOLDOUT.read_text(encoding="utf-8")))
    gold_docs = {g["doc_id"] for g in load_ftc_gold("all")}
    rows = load_jsonl(CLEAN) + load_jsonl(NOISE)
    ftc_docs = sorted({_doc(r["chunk_id"]) for r in rows if r["source"] == "ftc_case"
                       and _doc(r["chunk_id"]).split(":")[1] not in gold_docs})
    sc_docs = sorted({_doc(r["chunk_id"]) for r in rows if r["source"] == "standard_contract"
                      and _doc(r["chunk_id"]) not in held})
    rng = random.Random(SEED_SELECT)
    f_ftc = sorted(rng.sample(ftc_docs, round(len(ftc_docs) * FTC_SHARE)))
    f_sc = sorted(rng.sample(sc_docs, N_SC))
    fresh = set(f_ftc) | set(f_sc)
    save_json({"seed": SEED_SELECT, "ftc_docs": f_ftc, "sc_docs": f_sc,
               "pool": {"ftc_non_gold": len(ftc_docs), "sc_non_holdout": len(sc_docs)}}, FRESH / "split.json")

    nc = [r for r in load_jsonl(CLEAN) if _doc(r["chunk_id"]) not in fresh]
    nrsn = [r for r in load_jsonl(SN) if _doc(r["chunk_id"]) not in fresh]
    save_jsonl(nc, FRESH / "labels_NC.jsonl")
    save_jsonl(nrsn, FRESH / "labels_NRSN.jsonl")
    # 시험셋: 출처로만 정의 (라벨 무관). 같은 텍스트는 한 번만
    seen, pos, neg = set(), [], []
    for r in rows:
        d = _doc(r["chunk_id"])
        if r["text"] in seen or d not in fresh:
            continue
        seen.add(r["text"])
        (pos if r["source"] == "ftc_case" else neg).append({"chunk_id": r["chunk_id"], "doc": d, "text": r["text"]})
    save_jsonl(pos, FRESH / "test_pos.jsonl")
    save_jsonl(neg, FRESH / "test_neg.jsonl")
    stats = {"fresh_ftc_docs": len(f_ftc), "fresh_sc_docs": len(f_sc), "test_pos": len(pos), "test_neg": len(neg),
             "NC": len(nc), "NRSN": len(nrsn), "removed_from_SN": len(load_jsonl(SN)) - len(nrsn)}
    save_json(stats, FRESH / "prepare_stats.json")
    logger.info(json.dumps(stats, ensure_ascii=False))


def analyze(gpu: int) -> None:
    from backend.training.retrain import predict
    pos, neg = load_jsonl(FRESH / "test_pos.jsonl"), load_jsonl(FRESH / "test_neg.jsonl")
    sets = {"pos": [r["text"] for r in pos], "neg": [r["text"] for r in neg]}
    groups = {"pos": [r["doc"] for r in pos], "neg": [r["doc"] for r in neg]}

    def run(dirs: list[Path]) -> dict:
        per_seed, probs = [], {k: [] for k in sets}
        thrs = []
        for d in dirs:
            fl = {}
            for k, texts in sets.items():
                P, thr, _ = predict(d, texts, gpu)
                probs[k].append(P)
                fl[k] = (P >= thr).any(1)
            thrs.append(thr)
            per_seed.append(fl)
        thr = np.mean(thrs, axis=0)
        ens = {k: (np.mean(v, axis=0) >= thr).any(1) for k, v in probs.items()}
        return {"ens": ens, "seeds": per_seed}

    models = {"N-C": [PROJECT_ROOT / f"models/_e8_NC_s{s}" for s in (42, 7, 123)],
              "N-RSN": [PROJECT_ROOT / f"models/_e8_NRSN_s{s}" for s in (42, 7, 123)],
              "article_v2": [PROJECT_ROOT / "models/article_v2"]}
    res = {k: run(v) for k, v in models.items()}

    def contrast(x: str, y: str, k: str) -> dict:
        a, b = res[x], res[y]
        diff = a["ens"][k].astype(float) - b["ens"][k].astype(float)
        idx: dict[str, list[int]] = {}
        for i, g in enumerate(groups[k]):
            idx.setdefault(g, []).append(i)
        cl = [np.array(v) for v in idx.values()]
        rng = np.random.default_rng(42)
        boot = [diff[np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])].mean() for _ in range(3000)]
        sa = np.var([s[k].mean() for s in a["seeds"]], ddof=1) / len(a["seeds"]) if len(a["seeds"]) > 1 else 0.0
        sb = np.var([s[k].mean() for s in b["seeds"]], ddof=1) / len(b["seeds"]) if len(b["seeds"]) > 1 else 0.0
        se = float(np.sqrt(np.var(boot) + sa + sb))
        pt = float(diff.mean())
        return {"x": round(float(a["ens"][k].mean()) * 100, 2), "y": round(float(b["ens"][k].mean()) * 100, 2),
                "diff_pp": round(pt * 100, 2), "ci95_pp": [round((pt - 1.96 * se) * 100, 2), round((pt + 1.96 * se) * 100, 2)]}

    c_fpr = contrast("N-RSN", "N-C", "neg")
    c_rec = contrast("N-RSN", "N-C", "pos")
    reason = c_fpr["ci95_pp"][1] < 0
    harm = c_rec["ci95_pp"][1] < 0
    rep = {"rule": "eval/paper_plan.md §7-3", "n": {"test_pos": len(pos), "test_neg": len(neg),
                                                   "pos_docs": len(set(groups["pos"])), "neg_docs": len(set(groups["neg"]))},
           "ensemble_rates": {m: {"recall": round(float(r["ens"]["pos"].mean()) * 100, 2),
                                  "fpr": round(float(r["ens"]["neg"].mean()) * 100, 2),
                                  "seeds_recall": [round(float(s["pos"].mean()) * 100, 1) for s in r["seeds"]],
                                  "seeds_fpr": [round(float(s["neg"].mean()) * 100, 1) for s in r["seeds"]]} for m, r in res.items()},
           "①_fpr_RSN_minus_C": c_fpr, "②_recall_RSN_minus_C": c_rec,
           "report_vs_article_v2": {"fpr": contrast("N-RSN", "article_v2", "neg"), "recall": contrast("N-RSN", "article_v2", "pos"),
                                    "note": "article_v2는 옛 데이터로 학습해 새 시험 문서 일부를 봤을 수 있다 — 현행 쪽에 유리"},
           "decision": {"reason_fpr_reduced": reason, "harm_recall": harm, "recommend_swap": reason and not harm}}
    save_json(rep, PROJECT_ROOT / "data/eval/model/fresh_revalidation.json")
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    if args.prepare:
        prepare()
    if args.analyze:
        analyze(args.gpu)


if __name__ == "__main__":
    main()
