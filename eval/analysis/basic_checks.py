# eval/analysis/basic_checks.py
"""세 가지 기본 질문을 기존 예측·라벨로 다시 잰다 (보고용, 사전 등록 아님, 2026-10-13, 학습·API 0)

① 조 식별이 상수 기준선보다 나은가 — AI Hub 평가 절반(위반 조항 단일 조 정답)에서 배포 계열 모델 대 상수
② FB-Check가 단순 규칙("High 라벨만 남김")보다 나은가 — GPT-4o 라벨(E1 모집단)에서 같은 비교
③ 필터 학습 효과 — 이미 5번 잰 값을 한 표로 모은다(새 계산 없음)

    .venv/bin/python -m eval.analysis.basic_checks
"""

import json
from collections import Counter

import numpy as np

from backend.training.retrain import f1_rows
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json
from eval.label.label_ablation import _doc_of, _f1, _labels, _ref_by_doc
from eval.model.aihub_eval import CACHE, COMMON, testset

logger = load_logger("basic_checks.log")
OUT = PROJECT_ROOT / "data/eval/analysis/basic_checks.json"
rng = np.random.default_rng(42)


def cluster_ci(groups, fn, n=3000):
    idx = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    cl = [np.array(v) for v in idx.values()]
    b = [fn(np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])) for _ in range(n)]
    return [round(float(x) * 100, 2) for x in np.percentile(b, [2.5, 97.5])]


def check1() -> dict:
    items = testset()
    train_files = set(json.loads((PROJECT_ROOT / "data/eval/model/aihub_split.json").read_text(encoding="utf-8"))["train_files"])
    single = np.array([it["label"] == 1 and len(it["articles"]) == 1 and it["articles"][0] in COMMON for it in items])
    ev = np.array([it["group"] not in train_files for it in items])
    tr = np.array([it["group"] in train_files for it in items])
    sel = np.where(single & ev)[0]
    gold = np.array([[a in items[i]["articles"] for a in COMMON] for i in sel])
    groups = [items[i]["group"] for i in sel]
    # 상수: 학습 절반 위반 조항에서 가장 흔한 조 top-k를 항상 예측 (테스트를 보지 않고 고른다)
    freq = Counter(a for it, t, s in zip(items, tr, single) if t and s for a in it["articles"])
    consts = {k: [a for a, _ in freq.most_common(k)] for k in (1, 2, 3)}
    const_f1 = {}
    for k, arts in consts.items():
        pred = np.tile(np.array([a in arts for a in COMMON]), (len(sel), 1))
        const_f1[k] = f1_rows(pred, gold)
    out = {"n_eval_single_article_positives": int(len(sel)), "n_files": len(set(groups)),
           "constants": {f"top{k}": {"articles": consts[k], "f1_pp": round(float(v.mean()) * 100, 2)} for k, v in const_f1.items()},
           "models": {}}
    best_const = max(const_f1.values(), key=lambda v: v.mean())
    best_k = max(const_f1, key=lambda k: const_f1[k].mean())
    for name in ("C", "CN"):
        f1s, ptd = [], []
        for s in (42, 7, 123):
            z = np.load(CACHE / f"{name}_s{s}.npz")
            pred = z["P"][sel] >= z["thr"]
            f1s.append(f1_rows(pred, gold))
            ptd.append(pred.any(1))
        f1 = np.mean(f1s, axis=0)
        pt = np.mean(ptd, axis=0) > 0.5          # 시드 과반이 지목한 조항
        d = f1 - best_const
        out["models"][name] = {
            "f1_pp": round(float(f1.mean()) * 100, 2), "pointed_pct": round(float(pt.mean()) * 100, 1),
            "minus_best_const_pp": round(float(d.mean()) * 100, 2), "ci95": cluster_ci(groups, lambda i, d=d: d[i].mean()),
            "pointed_only_f1_pp": round(float(f1[pt].mean()) * 100, 2),
            "pointed_only_const_pp": round(float(best_const[pt].mean()) * 100, 2),
            "pointed_only_diff_ci95": cluster_ci([g for g, p in zip(groups, pt) if p], lambda i, d=d[pt]: d[i].mean())}
    out["best_const"] = f"top{best_k}"
    return out


def check2() -> dict:
    ref = _ref_by_doc()
    rows = []
    for r in load_jsonl(PROJECT_ROOT / "data/fb_check/clean.jsonl") + load_jsonl(PROJECT_ROOT / "data/fb_check/noise.jsonl"):
        doc = _doc_of(r.get("chunk_id", ""))
        if r.get("source") != "ftc_case" or doc not in ref:
            continue
        f = _labels(r.get("forward_articles"))
        rows.append({"doc": doc, "f": f, "clean": r.get("status") == "CLEAN", "grade": r.get("forward_label"),
                     "span": bool(r.get("snippet_exists")), "s": _f1(f, ref[doc]) if f else 0.0})
    has = np.array([bool(r["f"]) for r in rows])
    s = np.array([r["s"] for r in rows])
    docs = [r["doc"] for r in rows]
    rules = {"필터 없음": np.ones(len(rows), bool),
             "FB-Check(CLEAN)": np.array([r["clean"] for r in rows]),
             "High만 남김": np.array([r["grade"] == "High" for r in rows]),
             "High 또는 Medium": np.array([r["grade"] in ("High", "Medium") for r in rows]),
             "근거 문구가 원문에 있음": np.array([r["span"] for r in rows])}
    out = {"n_records": len(rows), "n_labeled": int(has.sum()), "rules": {}}
    for name, keep in rules.items():
        k = keep & has
        out["rules"][name] = {"kept_labeled": int(k.sum()), "f1_pp": round(float(s[k].mean()) * 100, 2)}
    # 같은 수를 문서 층화로 무작위로 남긴 기준선 대비 이득 (규칙별)
    by = {}
    for i, d in enumerate(docs):
        by.setdefault(d, []).append(i)
    blocks = [(np.array(v)) for v in by.values()]
    for name, keep in rules.items():
        if name == "필터 없음":
            continue
        base = np.empty(2000)
        for b in range(2000):
            m = np.zeros(len(rows), bool)
            for ids in blocks:
                kk = int(keep[ids].sum())
                if kk:
                    m[rng.choice(ids, kk, replace=False)] = True
            base[b] = s[m & has].mean()
        f = s[keep & has].mean()
        out["rules"][name]["gain_vs_docstrat_random_pp"] = round(float(f - base.mean()) * 100, 2)
        out["rules"][name]["perm_p"] = float((base >= f).mean())
    # FB-Check 대 High만: 문서 군집 부트스트랩 차이
    a, b2 = rules["FB-Check(CLEAN)"], rules["High만 남김"]
    out["fbcheck_minus_high_pp"] = round((s[a & has].mean() - s[b2 & has].mean()) * 100, 2)
    out["fbcheck_minus_high_ci95"] = cluster_ci(docs, lambda i: s[i][a[i] & has[i]].mean() - s[i][b2[i] & has[i]].mean())
    return out


def main() -> None:
    rep = {"rule": "보고용 재확인 (사전 등록 아님)", "1_조식별_대_상수(AI Hub 평가 절반)": check1(),
           "2_FBCheck_대_단순규칙(GPT-4o E1 모집단)": check2()}
    save_json(rep, OUT)
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
