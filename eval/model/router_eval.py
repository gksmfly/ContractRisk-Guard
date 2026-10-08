# eval/model/router_eval.py
"""문서 종류별 기준 분리 (라우터) — 한 모델 두 기준 (사전 등록: eval/paper_plan.md §7-12, 2026-10-08, API 0)

라우터(TF-IDF+LR)가 조항을 "회사 약관" / "표준계약서"로 가르고, CN 모델에 각각 CNA 임계값 / CN 원래 dev 임계값을 쓴다.
같은 시드 C와 비교한다.

    .venv/bin/python -m eval.model.router_eval --gpu 0                 # §7-12 조항 단위
    .venv/bin/python -m eval.model.router_eval --gpu 0 --doc-level     # §7-13 문서 단위 (문서 내 p 평균 ≥ 0.5)
"""

import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from backend.training.retrain import eval_sets, predict
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json
from eval.model.aihub_eval import testset

logger = load_logger("router_eval.log")

HOLD  = PROJECT_ROOT / "backend/training/negative_holdout_docs.json"
SPLIT = PROJECT_ROOT / "data/eval/model/aihub_split.json"
OUT   = PROJECT_ROOT / "data/eval/model/router_eval.json"
SEEDS = [42, 7, 123]
_doc = lambda cid: ":".join(str(cid).split(":")[:2])  # noqa: E731


def train_router(items: list[dict], train_files: set[str]):
    held = set(json.loads(HOLD.read_text(encoding="utf-8")))
    company = [it["text"] for it in items if it["group"] in train_files]
    standard = [r["text"] for r in load_jsonl(PROJECT_ROOT / "data/fb_check/clean.jsonl")
                if r["source"] == "standard_contract" and _doc(r["chunk_id"]) not in held]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=50_000, sublinear_tf=True)
    X = vec.fit_transform(company + standard)
    y = np.array([1] * len(company) + [0] * len(standard))
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X, y)
    return (lambda texts: clf.predict_proba(vec.transform(texts))[:, 1]), {"company": len(company), "standard": len(standard)}


def route(probs: np.ndarray, docs: list[str], doc_level: bool) -> np.ndarray:
    if not doc_level:
        return probs >= 0.5
    by: dict[str, list[int]] = {}
    for i, d in enumerate(docs):
        by.setdefault(d, []).append(i)
    out = np.zeros(len(probs), dtype=bool)
    for idx in by.values():
        out[idx] = probs[idx].mean() >= 0.5
    return out


def flags(model: str, seed: int, texts: list[str], route: np.ndarray, gpu: int) -> np.ndarray:
    if model == "C":
        P, thr, _ = predict(PROJECT_ROOT / f"models/_e8_C_s{seed}", texts, gpu)
        return (P >= thr).any(1)
    P, thr_dev, _ = predict(PROJECT_ROOT / f"models/_e8_CN_s{seed}", texts, gpu)
    m = json.loads((PROJECT_ROOT / f"models/_e8_CNA_s{seed}/metrics.json").read_text(encoding="utf-8"))
    thr_co = np.array([m["thresholds"][a] for a in m["article_names"]])
    thr = np.where(route[:, None], thr_co[None, :], thr_dev[None, :])
    return (P >= thr).any(1)


def contrast(x: list[np.ndarray], y: list[np.ndarray], groups: list[str], lower_better: bool) -> dict:
    d = np.mean(x, axis=0).astype(float) - np.mean(y, axis=0).astype(float)
    idx: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    cl = [np.array(v) for v in idx.values()]
    rng = np.random.default_rng(42)
    b = [d[np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])].mean() for _ in range(3000)]
    se = float(np.sqrt(np.var(b) + np.var([v.mean() for v in x], ddof=1) / len(x) + np.var([v.mean() for v in y], ddof=1) / len(y)))
    pt = float(d.mean())
    lo, hi = pt - 1.96 * se, pt + 1.96 * se
    verdict = "미판정" if lo <= 0 <= hi else ("RT 우위" if (hi < 0) == lower_better else "C 우위")
    return {"C": round(float(np.mean([v.mean() for v in y])) * 100, 2), "RT": round(float(np.mean([v.mean() for v in x])) * 100, 2),
            "diff_pp": round(pt * 100, 2), "ci95_pp": [round(lo * 100, 2), round(hi * 100, 2)], "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--doc-level", action="store_true")
    args = ap.parse_args()
    items = testset()
    train_files = set(json.loads(SPLIT.read_text(encoding="utf-8"))["train_files"])
    route_fn, router_n = train_router(items, train_files)

    test = [it for it in items if it["group"] not in train_files]
    gold, neg = eval_sets(json.loads(HOLD.read_text(encoding="utf-8")))
    real_rows = [(c["text"], f.stem) for f in sorted((PROJECT_ROOT / "data/eval/real_tos").glob("*.json"))
                 for c in json.loads(f.read_text(encoding="utf-8"))["clauses"]]
    real = [t for t, _ in real_rows]
    sets = {
        "aihub_pos": ([it["text"] for it in test if it["label"] == 1], [it["group"] for it in test if it["label"] == 1]),
        "aihub_neg": ([it["text"] for it in test if it["label"] == 0], [it["group"] for it in test if it["label"] == 0]),
        "gold": ([g["text"] for g in gold], [g["doc_id"] for g in gold]),
        "holdout": ([r["text"] for r in neg], [_doc(r["chunk_id"]) for r in neg]),
        "real_tos": (real, [d for _, d in real_rows]),
    }
    # 군집(문서) id가 곧 라우팅 문서 단위다. 실제 약관은 회사 파일 단위
    routes = {k: route(route_fn(v[0]), v[1], args.doc_level) for k, v in sets.items()}
    fl = {m: {k: [flags(m, s, v[0], routes[k], args.gpu) for s in SEEDS] for k, v in sets.items()} for m in ("C", "RT")}

    res = {
        "주_aihub_fpr": contrast(fl["RT"]["aihub_neg"], fl["C"]["aihub_neg"], sets["aihub_neg"][1], lower_better=True),
        "해①_aihub_recall": contrast(fl["RT"]["aihub_pos"], fl["C"]["aihub_pos"], sets["aihub_pos"][1], lower_better=False),
        "해②_gold_recall": contrast(fl["RT"]["gold"], fl["C"]["gold"], sets["gold"][1], lower_better=False),
        "해③_holdout_fpr": contrast(fl["RT"]["holdout"], fl["C"]["holdout"], sets["holdout"][1], lower_better=True),
        "보고_real_tos_flag": contrast(fl["RT"]["real_tos"], fl["C"]["real_tos"], sets["real_tos"][1], lower_better=True),
    }
    main_ok = res["주_aihub_fpr"]["ci95_pp"][1] < 0
    harm = (res["해①_aihub_recall"]["ci95_pp"][1] < 0 or res["해②_gold_recall"]["ci95_pp"][1] < 0
            or res["해③_holdout_fpr"]["ci95_pp"][0] > 0)
    rep = {"rule": "eval/paper_plan.md §7-13" if args.doc_level else "eval/paper_plan.md §7-12",
           "doc_level": args.doc_level, "router_train_n": router_n,
           "routed_to_company_pct": {k: round(float(v.mean()) * 100, 1) for k, v in routes.items()},
           "results": res, "decision": {"main": main_ok, "harm": harm, "adopt": main_ok and not harm}}
    save_json(rep, OUT.with_name("router_eval_doc.json") if args.doc_level else OUT)
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
