# eval/model/aihub_eval.py
"""AI Hub 외부 대형 평가 (사전 등록: eval/paper_plan.md §7-8, 2026-10-07, API 0)

평가셋: AI Hub 약관 중 우리 데이터 전체·gold와 공백·기호 제거 30자 조각을 공유하지 않는 조항
        위반("불리" + 제6~14조 표시) 2,575 · 공정("유리") 6,097 (§24)
지표:   주 AUROC(위반 vs 공정, r = max(p/thr)) · 보조 재현 80%에서의 오경보 · 조문 1개 위반의 조 F1(배포 임계값)
CI:     AI Hub 파일 단위 군집 부트스트랩 + 시드 분산
판정:   D−B · MV−RMV · MV−C 각각 CI 하한 > 0 → "개선"

    .venv/bin/python -m eval.model.aihub_eval --gpu 0                  # §7-8 (3시드)
    .venv/bin/python -m eval.model.aihub_eval --gpu 0 --mode seeds10   # §7-9 가: D·B 시드 10개
    .venv/bin/python -m eval.model.aihub_eval --gpu 0 --mode seeds20   # §7-15: 시드 20개 (학습은 aihub_seeds20.py)
    .venv/bin/python -m eval.model.aihub_eval --gpu 0 --mode cn        # §7-9 나: AI Hub 평가 절반에서 CN vs C
"""

import argparse
import glob
import json
import os
import re

import numpy as np

from backend.labeling.articles import ARTICLE_IDS
from backend.training.retrain import f1_rows, predict
from backend.training.train_article import load_ftc_gold
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json
from eval.model.e8b_matched import COMMON

logger = load_logger("aihub_eval.log")

CACHE = PROJECT_ROOT / "data/eval/model/aihub_cache"
OUT   = PROJECT_ROOT / "data/eval/model/aihub_eval.json"
ARMS  = ["A", "B", "C", "D", "CL", "MV", "RMV"]
SEEDS = [42, 7, 123]
PRIMARY = [("D", "B", "FB-Check 필터 vs 같은 수 무작위 (forward)"),
           ("MV", "RMV", "다중 검증자 vs 같은 수 무작위 (agreed)"),
           ("MV", "C", "다중 검증자 vs 현행 FB-Check (agreed)")]
REPORT = [("C", "A", "현행 vs 필터 없음"), ("CL", "D", "CleanLab vs FB-Check"), ("B", "A", "무작위 vs 필터 없음(가짜 처치에 가까움)")]
_norm = lambda t: re.sub(r"[\s\W_]+", "", t or "")  # noqa: E731


def _sh(t: str, step: int = 1) -> set[str]:
    t = _norm(t)
    return {t[i:i + 30] for i in range(0, max(1, len(t) - 29), step)}


def testset() -> list[dict]:
    ours = [r["text"] for p in ("clean", "noise") for r in load_jsonl(PROJECT_ROOT / f"data/fb_check/{p}.jsonl")]
    ours += [g["text"] for g in load_ftc_gold("all")]
    banned = set().union(*(_sh(t) for t in ours))
    out = []
    for f in sorted({os.path.realpath(x) for x in glob.glob(str(PROJECT_ROOT / "data/raw/aihub_tos/**/*.json"), recursive=True)}):
        d = json.load(open(f, encoding="utf-8"))
        adv = d.get("dvAntageous")
        arts = sorted({f"제{m}조" for law in d.get("relateLaword", []) for m in re.findall(r"제(\d+)조", law)
                       if f"제{m}조" in ARTICLE_IDS})
        for t in d.get("clauseArticle", []):
            if len((t or "").strip()) < 30 or (_sh(t, step=3) & banned):
                continue
            if adv == "2" and arts:
                out.append({"text": t, "label": 1, "articles": arts, "group": os.path.basename(f)})
            elif adv == "1":
                out.append({"text": t, "label": 0, "articles": [], "group": os.path.basename(f)})
    return out


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def metrics(P: np.ndarray, thr: np.ndarray, items: list[dict], idx: np.ndarray) -> dict:
    lab = np.array([it["label"] for it in items])[idx]
    r = (P[idx] / thr).max(1)
    pos, neg = r[lab == 1], r[lab == 0]
    cut = np.sort(pos)[::-1][int(np.ceil(0.80 * len(pos))) - 1]
    single = np.array([it["label"] == 1 and len(it["articles"]) == 1 and it["articles"][0] in COMMON for it in items])[idx]
    gold = np.array([[a in it["articles"] for a in COMMON] for it in items])[idx][single]
    return {"auroc": _auroc(pos, neg), "fpr@80": float((neg >= cut).mean()),
            "deploy_recall": float((pos >= 1).mean()), "deploy_fpr": float((neg >= 1).mean()),
            "article_f1": float(f1_rows(P[idx][single] >= thr, gold).mean())}


def load_probs(name: str, d, texts: list[str], gpu: int):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}.npz"
    if f.exists():
        z = np.load(f)
        return z["P"], z["thr"]
    P, thr, names = predict(d, texts, gpu)
    cols = [list(names).index(a) for a in COMMON]
    P, thr = P[:, cols], thr[cols]
    np.savez_compressed(f, P=P, thr=thr)
    return P, thr


MODES = {
    "full":    {"arms": ARMS, "seeds": SEEDS, "primary": PRIMARY, "report": REPORT, "out": OUT, "half": False},
    "seeds10": {"arms": ["D", "B"], "seeds": [42, 7, 123, 11, 22, 33, 44, 55, 66, 77],
                "primary": [("D", "B", "FB-Check 필터 vs 같은 수 무작위 (forward), 시드 10개")], "report": [],
                "out": OUT.with_name("aihub_eval_seeds10.json"), "half": False},
    "seeds20": {"arms": ["D", "B"], "seeds": [42, 7, 123, 11, 22, 33, 44, 55, 66, 77,
                                              88, 99, 101, 202, 303, 404, 505, 606, 707, 808],
                "primary": [("D", "B", "FB-Check 필터 vs 같은 수 무작위 (forward), 시드 20개 — 최종 (§7-15)")], "report": [],
                "out": OUT.with_name("aihub_eval_seeds20.json"), "half": False},
    "binary":  {"arms": ["D2", "B2", "D"], "seeds": [42, 7, 123, 11, 22, 33, 44, 55, 66, 77],
                "primary": [("D2", "B2", "2진 FB-Check 필터 vs 같은 수 무작위 (forward), 시드 10개 (§7-16 ②)")],
                "report": [("D2", "D", "2진 규칙 vs 기존 규칙 (보고)")],
                "out": OUT.with_name("aihub_eval_binary.json"), "half": False},
    "soft":    {"arms": ["S", "A", "D"], "seeds": [42, 7, 123, 11, 22, 33, 44, 55, 66, 77],
                "primary": [("S", "A", "FB-Check v2(확신도) vs 필터 없음, 시드 10개 (§7-17)"),
                            ("S", "D", "FB-Check v2 vs 기존 FB-Check (§7-17 보조)")],
                "report": [("A", "D", "필터 없음 vs 기존 FB-Check (보고)")],
                "out": OUT.with_name("aihub_eval_soft.json"), "half": False},
    "ds":      {"arms": ["DS", "D", "A"], "seeds": [42, 7, 123, 11, 22, 33, 44, 55, 66, 77],
                "primary": [("DS", "D", "표준계약서 NOISE만 제외 vs 기존 FB-Check — H1 조 F1 (§7-18)"),
                            ("DS", "A", "표준계약서 NOISE만 제외 vs 필터 없음 — H2 AUROC (§7-18)")],
                "report": [],
                "out": OUT.with_name("aihub_eval_ds.json"), "half": False},
    "adapter": {"arms": ["LA", "A", "LD", "D"], "seeds": [42, 7, 123, 11, 22, 33, 44, 55, 66, 77],
                "primary": [("LA", "A", "LoRA 대 전체 파인튜닝, 필터 없음 데이터 (§7-26 주 판정 1)"),
                            ("LD", "D", "LoRA 대 전체 파인튜닝, FB-Check 데이터 (§7-26 주 판정 2)")],
                "report": [("LD", "LA", "LoRA 위에서 FB-Check 대 필터 없음 (보고)")],
                "out": OUT.with_name("aihub_eval_adapter.json"), "half": False},
    "labeltype": {"arms": ["OQ", "OFP", "OFN", "OMX", "OG"], "seeds": [42, 7, 123, 11, 22, 33, 44, 55, 66, 77],
                  "primary": [("OG", "OQ", "총 영향(게이트): 전부 정답 대 Qwen 라벨 (§7-27 주 판정 1)"),
                              ("OFN", "OFP", "누락 수정 대 잘못된 라벨 수정 (§7-27 주 판정 2, 게이트 통과 시에만)")],
                  "report": [("OFP", "OQ", "잘못된 라벨만 고침 대 Qwen 라벨"), ("OFN", "OQ", "누락만 고침 대 Qwen 라벨"),
                             ("OMX", "OQ", "혼합만 고침 대 Qwen 라벨")],
                  "out": OUT.with_name("aihub_eval_labeltype.json"), "half": True},
    "cnc":     {"arms": ["C", "CNC"], "seeds": SEEDS,
                "primary": [("CNC", "C", "CN 임계값 dev 재조정 vs 현행 (AI Hub 평가 절반, §7-10)")], "report": [],
                "out": OUT.with_name("aihub_eval_cnc.json"), "half": True},
    "cna":     {"arms": ["C", "CNA"], "seeds": SEEDS,
                "primary": [("CNA", "C", "CN 임계값 AI Hub 학습 절반 재조정 vs 현행 (AI Hub 평가 절반, §7-11)")], "report": [],
                "out": OUT.with_name("aihub_eval_cna.json"), "half": True},
    "cn":      {"arms": ["C", "CN"], "seeds": SEEDS,
                "primary": [("CN", "C", "AI Hub 공정 조항 1,500건 추가 vs 현행 (AI Hub 평가 절반)")], "report": [],
                "out": OUT.with_name("aihub_eval_cn.json"), "half": True},
}
SPLIT = PROJECT_ROOT / "data/eval/model/aihub_split.json"


def harm_check(gpu: int, cand: str = "CN") -> dict:
    """§7-9 나 판정 2의 기존 규칙: gold 255 조항 재현 · 홀드아웃 12문서 오경보, CN − C (시드 평균, 군집 부트스트랩 + 시드 분산)."""
    from backend.training.retrain import eval_sets
    gold, neg = eval_sets(json.loads((PROJECT_ROOT / "backend/training/negative_holdout_docs.json").read_text(encoding="utf-8")))
    res = {}
    for a in ("C", cand):
        rec, fpr = [], []
        for s in SEEDS:
            d = PROJECT_ROOT / f"models/_e8_{a}_s{s}"
            P, thr, _ = predict(d, [g["text"] for g in gold], gpu)
            Pn, *_ = predict(d, [r["text"] for r in neg], gpu)
            rec.append((P >= thr).any(1).astype(float))
            fpr.append((Pn >= thr).any(1).astype(float))
        res[a] = (rec, fpr)
    out = {}
    for k, gi, grp in (("gold_recall", 0, [g["doc_id"] for g in gold]),
                       ("holdout_fpr", 1, [":".join(str(r["chunk_id"]).split(":")[:2]) for r in neg])):
        x, y = res[cand][gi], res["C"][gi]
        d = np.mean(x, axis=0) - np.mean(y, axis=0)
        idx: dict[str, list[int]] = {}
        for i, g in enumerate(grp):
            idx.setdefault(g, []).append(i)
        cl = [np.array(v) for v in idx.values()]
        rng = np.random.default_rng(42)
        b = [d[np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])].mean() for _ in range(3000)]
        se = float(np.sqrt(np.var(b) + np.var([v.mean() for v in x], ddof=1) / 3 + np.var([v.mean() for v in y], ddof=1) / 3))
        pt = float(d.mean())
        out[k] = {"C": round(float(np.mean([v.mean() for v in y])) * 100, 2), cand: round(float(np.mean([v.mean() for v in x])) * 100, 2),
                  "diff_pp": round(pt * 100, 2), "ci95_pp": [round((pt - 1.96 * se) * 100, 2), round((pt + 1.96 * se) * 100, 2)]}
    out["harm"] = out["gold_recall"]["ci95_pp"][1] < 0 or out["holdout_fpr"]["ci95_pp"][0] > 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--mode", choices=list(MODES), default="full")
    args = ap.parse_args()
    cfg = MODES[args.mode]
    ARMS_, SEEDS_, PRIMARY_, REPORT_, OUT_ = cfg["arms"], cfg["seeds"], cfg["primary"], cfg["report"], cfg["out"]
    items = testset()
    texts = [it["text"] for it in items]        # 예측 캐시는 전체 평가셋 기준 — 절반 평가는 인덱스로 고른다
    if cfg["half"]:
        train_files = set(json.loads(SPLIT.read_text(encoding="utf-8"))["train_files"])
        keep_idx = np.array([i for i, it in enumerate(items) if it["group"] not in train_files])
    else:
        keep_idx = np.arange(len(items))
    logger.info(f"평가셋: 위반 {sum(i['label'] for i in items)} · 공정 {sum(1 - i['label'] for i in items)}")
    probs = {}
    for a in ARMS_:
        for s in SEEDS_:
            d = PROJECT_ROOT / f"models/_e8_{a}_s{s}"
            # 디스크 때문에 가중치를 지운 모델도 예측 캐시가 있으면 쓴다 (§7-15)
            if (d / "model.safetensors").exists() or (CACHE / f"{a}_s{s}.npz").exists():
                probs[(a, s)] = load_probs(f"{a}_s{s}", d, texts, args.gpu)
    probs[("article_v2", 0)] = load_probs("article_v2", PROJECT_ROOT / "models/article_v2", texts, args.gpu)

    all_idx = keep_idx
    per = {k: metrics(P, thr, items, all_idx) for k, (P, thr) in probs.items()}
    keys = list(next(iter(per.values())))
    arms = {}
    for a in ARMS_ + ["article_v2"]:
        ms = [per[k] for k in per if k[0] == a]
        if ms:
            arms[a] = {m: {"mean": round(float(np.mean([x[m] for x in ms])), 4),
                           "seeds": [round(x[m], 4) for x in ms]} for m in keys}

    # 군집 부트스트랩 (AI Hub 파일 단위)
    gidx: dict[str, list[int]] = {}
    for i in keep_idx:
        gidx.setdefault(items[i]["group"], []).append(int(i))
    groups = [np.array(v) for v in gidx.values()]
    rng = np.random.default_rng(42)
    boots = {a: {m: [] for m in keys} for a in ARMS_ if any(k[0] == a for k in probs)}
    for _ in range(args.n_boot):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        for a in boots:
            ms = [metrics(P, thr, items, idx) for (aa, _), (P, thr) in probs.items() if aa == a]
            for m in keys:
                boots[a][m].append(np.mean([x[m] for x in ms]))

    def contrast(x: str, y: str) -> dict:
        row = {}
        for m in keys:
            d = np.array(boots[x][m]) - np.array(boots[y][m])
            sx = np.var(arms[x][m]["seeds"], ddof=1) / len(arms[x][m]["seeds"])
            sy = np.var(arms[y][m]["seeds"], ddof=1) / len(arms[y][m]["seeds"])
            se = float(np.sqrt(d.var() + sx + sy))
            pt = arms[x][m]["mean"] - arms[y][m]["mean"]
            lo, hi = pt - 1.96 * se, pt + 1.96 * se
            lower_better = m.endswith("fpr") or m.startswith("fpr")
            v = "미판정" if lo <= 0 <= hi else (f"{x} 우위" if (lo > 0) != lower_better else f"{y} 우위")
            row[m] = {"diff_pp": round(pt * 100, 2), "ci95_pp": [round(lo * 100, 2), round(hi * 100, 2)], "verdict": v}
        return row

    primary = {f"{x}-{y}": {"desc": dsc, **contrast(x, y)} for x, y, dsc in PRIMARY_ if x in boots and y in boots}
    report = {f"{x}-{y}": {"desc": dsc, **contrast(x, y)} for x, y, dsc in REPORT_ if x in boots and y in boots}
    decision = {k: v["auroc"]["verdict"] for k, v in primary.items()}
    sub = [items[i] for i in keep_idx]
    rep = {"rule": {"full": "eval/paper_plan.md §7-8", "seeds20": "eval/paper_plan.md §7-15", "binary": "eval/paper_plan.md §7-16 ②", "soft": "eval/paper_plan.md §7-17", "ds": "eval/paper_plan.md §7-18", "adapter": "eval/paper_plan.md §7-26", "labeltype": "eval/paper_plan.md §7-27"}.get(args.mode, "eval/paper_plan.md §7-9"), "mode": args.mode,
           "seeds": SEEDS_, "n": {"pos": int(sum(i["label"] for i in sub)), "neg": int(sum(1 - i["label"] for i in sub)),
                                  "pos_single_article": int(sum(i["label"] == 1 and len(i["articles"]) == 1 for i in sub)),
                                  "files": len(groups)},
           "arms": arms, "primary": primary, "report": report, "decision_auroc": decision}
    if args.mode in ("cn", "cnc", "cna"):
        rep["harm_gold_holdout"] = harm_check(args.gpu, {"cn": "CN", "cnc": "CNC", "cna": "CNA"}[args.mode])
    save_json(rep, OUT_)
    for a, v in arms.items():
        logger.info(f"  {a:<10} AUROC {v['auroc']['mean']:.4f} 오경보@80 {v['fpr@80']['mean']*100:5.1f}% "
                    f"조F1 {v['article_f1']['mean']*100:5.1f}% 배포 재현 {v['deploy_recall']['mean']*100:5.1f}% 배포 오경보 {v['deploy_fpr']['mean']*100:5.1f}%")
    for k, v in {**primary, **report}.items():
        logger.info(f"  {k:<8} " + " | ".join(f"{m} {x['diff_pp']:+.2f} {x['ci95_pp']} {x['verdict']}" for m, x in v.items() if m != "desc"))
    logger.info(f"  판정(AUROC): {decision}")


if __name__ == "__main__":
    main()
