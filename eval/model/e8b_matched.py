# eval/model/e8b_matched.py
"""E8b — 하류 학습 ablation 재분석: 공통 헤드 · 같은 재현율에서 비교 (P1-1, 2026-09-30)

## 왜 다시 재나 — E8(09-28)의 결함 셋

1. **팔마다 자기 dev 임계값(동작점)에서 비교했다.** 재현과 오경보는 임계값 하나로 맞바꿔지는
   지표라 동작점이 다르면 우열이 아니라 위치 차이를 잰 것이다(규칙 9).
2. **C만 두 가지가 동시에 달랐다.** 라벨 소스(agreed vs forward)와 음성 비율(빈 라벨
   59.2% vs 44~45%). → 팔 D(CLEAN 레코드 + forward 라벨)로 분해한다.
3. **forward 팔에만 제13조 헤드가 생긴다.** forward 라벨에서는 제13조가 min_support(5)를
   넘지만 dev에 양성이 거의 없어 임계값이 0.05로 무너진다(B s42, D s42). 이 헤드 하나가
   B의 오경보 30.44%, D s42의 94.7%를 만든다. gold 255에는 제13조가 0건이고 배포 모델은
   이 조를 접는다. → **8개 공통 조(제6·7·8·9·10·11·12·14조) 헤드만** 쓴다.

## 지표 (전부 공통 8개 헤드)

- 배포 동작점: 자기 dev 임계값 그대로의 재현·오경보·조 F1 (참고)
- **임계값 무관 순위 점수** r = max_a p_a / thr_a (배포 임계값에 곱하는 배율을 훑는 것과 동치,
  `threshold_match_sweep`과 같은 틀). gold(위반) vs 홀드아웃(비위반) AUROC
- **같은 재현율에서 오경보**: 재현 80 / 85 / 90%가 되는 r 문턱에서의 홀드아웃 오경보
- 같은 재현율 90%에서의 조 F1(참고 — 조 F1은 gold 255로 판정 불가, 필요 n≈1,296)

## 판정 (사전 등록, paper_plan.md §P1-1)

팔 간 차는 시드 평균끼리, **gold 문서와 홀드아웃 문서를 각각 군집 재표집**한 부트스트랩
95% CI로 낸다(문턱은 재표집마다 다시 잡는다). CI가 0을 넘으면 유의, 포함하면 미판정 + MDE.

    필터 효과        D − A   (레코드 선택만 다름. 라벨 소스 forward 동일)
    양 vs 질         D − B   (같은 수 · 같은 라벨 소스, 무작위 vs FB-Check)
    라벨 소스 효과   C − D   (같은 레코드, agreed vs forward)
    종합(옛 E8)      C − A · C − B
    전문가 라벨      X3352 − C (같은 수) · X − C · X − A

⚠ 홀드아웃은 표준계약서 12문서 ~300조항이다. 문서 단위 재표집이라 오경보 CI가 넓다 —
MDE를 같이 적는다.

## 실행

    python -m eval.model.e8b_matched --gpu 1            # 예측 캐시 + 분석
    python -m eval.model.e8b_matched --analyze-only     # 캐시만으로 분석
"""

import argparse
import json
from pathlib import Path

import numpy as np

from backend.training.retrain import eval_sets, f1_rows, predict
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("e8b_matched.log")

HOLDOUT_PATH = PROJECT_ROOT / "backend/training/negative_holdout_docs.json"
CACHE_DIR    = PROJECT_ROOT / "data/eval/model/e8b_cache"
OUT_PATH     = PROJECT_ROOT / "data/eval/model/e8b_matched.json"

COMMON  = ["제6조", "제7조", "제8조", "제9조", "제10조", "제11조", "제12조", "제14조"]
ARMS    = ["A", "B", "C", "D", "X", "X3352", "X1000", "XC", "CL", "G", "K", "RB"]
SEEDS   = [42, 7, 123]
TARGETS = [0.80, 0.85, 0.90]
N_BOOT  = 2000
SEED    = 42

CONTRASTS = [
    ("RB", "C", "KLUE-RoBERTa-base vs KoELECTRA-base (같은 데이터·설정, 백본만 다름)"),
    ("CL", "D", "CleanLab vs FB-Check (같은 forward 라벨, 공정위 조항 선택 규칙만 다름)"),
    ("CL", "A", "CleanLab 필터 효과"),
    ("K", "A", "Consistency만의 필터 효과"),
    ("G", "A", "Grounding만의 필터 효과"),
    ("K", "D", "Grounding 추가분 (D = K − snippet 33건)"),
    ("D", "A", "필터 효과 (레코드 선택만 다름)"),
    ("D", "B", "양 vs 질 (같은 수·같은 라벨 소스)"),
    ("C", "D", "라벨 소스 효과 (agreed vs forward, 같은 레코드)"),
    ("C", "A", "종합 — 옛 E8 C vs A"),
    ("C", "B", "종합 — 옛 E8 C vs B"),
    ("XC", "C", "전문가 라벨 vs FB-Check (같은 실제 학습 수)"),
    ("X3352", "C", "전문가 라벨 3,352 vs FB-Check (실제 학습 1,945 — AI Hub가 1.7배 많음)"),
    ("X", "C", "전문가 라벨 전량 vs FB-Check"),
    ("X", "A", "전문가 라벨 전량 vs 필터 없음"),
    ("X1000", "X3352", "전문가 라벨 학습 곡선 (1,000 vs 3,352)"),
]


def _model_dir(arm: str, seed: int) -> Path:
    return PROJECT_ROOT / "models" / f"_e8_{arm}_s{seed}"


# ── 예측 캐시 ────────────────────────────────────────────────────────────────

def cache_predictions(gpu: int) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    holdout = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
    gold, neg = eval_sets(holdout)
    targets = [(f"{a}_s{s}", _model_dir(a, s)) for a in ARMS for s in SEEDS]
    targets.append(("article_v2", PROJECT_ROOT / "models/article_v2"))
    for name, d in targets:
        out = CACHE_DIR / f"{name}.npz"
        if out.exists() or not (d / "metrics.json").exists():
            continue
        P, thr, names = predict(d, [g["text"] for g in gold], gpu)
        Pn, *_ = predict(d, [r["text"] for r in neg], gpu)
        np.savez_compressed(out, P=P, Pn=Pn, thr=thr, names=np.array(names))
        logger.info(f"  캐시: {name} ({len(names)}개 헤드)")


# ── 지표 ─────────────────────────────────────────────────────────────────────

def _load(name: str):
    z = np.load(CACHE_DIR / f"{name}.npz")
    names = list(z["names"])
    cols = [names.index(a) for a in COMMON]
    return z["P"][:, cols], z["Pn"][:, cols], z["thr"][cols]


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    # 동점 평균 순위
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _metrics(P, Pn, thr, labels, gi: np.ndarray, ni: np.ndarray) -> dict:
    """gi/ni: 재표집 인덱스(항등이면 원본)."""
    P, Pn, lab = P[gi], Pn[ni], labels[gi]
    pred = P >= thr
    rg = (P / thr).max(1)
    rn = (Pn / thr).max(1)
    out = {"deploy_recall": float(pred.any(1).mean()), "deploy_fpr": float((Pn >= thr).any(1).mean()),
           "deploy_f1": float(f1_rows(pred, lab).mean()), "auroc": _auroc(rg, rn)}
    srt = np.sort(rg)[::-1]
    for t in TARGETS:
        k = int(np.ceil(t * len(srt))) - 1          # 재현 ≥ t가 되는 가장 높은 문턱
        cut = srt[k]
        out[f"fpr@{int(t*100)}"] = float((rn >= cut).mean())
        if t == 0.90:
            s = cut                                   # 배포 임계값 × s 로 맞춘 것과 같다
            out["f1@90"] = float(f1_rows(P >= thr * s, lab).mean())
    return out


def _groups(ids: list[str]) -> list[np.ndarray]:
    idx: dict[str, list[int]] = {}
    for i, g in enumerate(ids):
        idx.setdefault(g, []).append(i)
    return [np.array(v) for v in idx.values()]


def analyze() -> None:
    holdout = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
    gold, neg = eval_sets(holdout)
    labels = np.array([[a in g.get("articles", []) for a in COMMON] for g in gold])
    g_groups = _groups([g["doc_id"] for g in gold])
    n_groups = _groups([":".join(str(r["chunk_id"]).split(":")[:2]) for r in neg])
    logger.info(f"=== E8b === gold {len(gold)}({len(g_groups)}문서) · 홀드아웃 {len(neg)}({len(n_groups)}문서)")

    models = {a: [f"{a}_s{s}" for s in SEEDS if (CACHE_DIR / f"{a}_s{s}.npz").exists()] for a in ARMS}
    models = {a: v for a, v in models.items() if v}
    data = {m: _load(m) for ms in models.values() for m in ms}
    ref = _load("article_v2") if (CACHE_DIR / "article_v2.npz").exists() else None

    gi0, ni0 = np.arange(len(gold)), np.arange(len(neg))
    per_model = {m: _metrics(*data[m], labels, gi0, ni0) for m in data}
    keys = list(next(iter(per_model.values())).keys())

    arms = {}
    for a, ms in models.items():
        arms[a] = {k: {"mean": float(np.mean([per_model[m][k] for m in ms])),
                       "sd": float(np.std([per_model[m][k] for m in ms])),
                       "seeds": [round(per_model[m][k], 4) for m in ms]} for k in keys}
        arms[a]["n_seeds"] = len(ms)

    # ── 군집 부트스트랩: 재표집마다 모든 모델을 같은 인덱스로 다시 잰다 ──
    rng = np.random.default_rng(SEED)
    boot = {a: {k: [] for k in keys} for a in models}
    for _ in range(N_BOOT):
        gi = np.concatenate([g_groups[j] for j in rng.integers(0, len(g_groups), len(g_groups))])
        ni = np.concatenate([n_groups[j] for j in rng.integers(0, len(n_groups), len(n_groups))])
        for a, ms in models.items():
            vals = [_metrics(*data[m], labels, gi, ni) for m in ms]
            for k in keys:
                boot[a][k].append(np.mean([v[k] for v in vals]))

    contrasts = {}
    for x, y, desc in CONTRASTS:
        if x not in models or y not in models:
            continue
        row = {"desc": desc}
        for k in keys:
            d = np.array(boot[x][k]) - np.array(boot[y][k])
            lo, hi = np.percentile(d, [2.5, 97.5])
            pt = arms[x][k]["mean"] - arms[y][k]["mean"]
            # ★ 학습 변동 포함 CI (2026-10-04). 위 부트스트랩은 **테스트 표본만** 재표집해 학습 무작위성
            # (dev 분할·체크포인트·임계값 선택)을 빼먹는다. 가짜 처치 G(A − 33건)가 A와 "유의하게"
            # 다르게 나와서 드러났다. 시드 평균의 학습 분산 (s²_x + s²_y)/n 을 더한 정규 근사 CI를 판정에 쓴다
            sx = np.var(arms[x][k]["seeds"], ddof=1) if len(arms[x][k]["seeds"]) > 1 else 0.0
            sy = np.var(arms[y][k]["seeds"], ddof=1) if len(arms[y][k]["seeds"]) > 1 else 0.0
            se = float(np.sqrt(d.var() + sx / len(arms[x][k]["seeds"]) + sy / len(arms[y][k]["seeds"])))
            tlo, thi = pt - 1.96 * se, pt + 1.96 * se
            row.setdefault("_train", {})[k] = {
                "ci95_pp": [round(tlo * 100, 2), round(thi * 100, 2)], "mde80_pp": round(se * 2.8 * 100, 2),
                "verdict": ("미판정" if tlo <= 0 <= thi else
                            f"{x} 우위" if (tlo > 0) != k.startswith(("fpr", "deploy_fpr")) else f"{y} 우위")}
            row[k] = {"diff_pp": round(pt * 100, 2), "ci95_pp": [round(float(lo) * 100, 2), round(float(hi) * 100, 2)],
                      "mde80_pp": round(float(d.std()) * 2.8 * 100, 2),
                      # 차 = x − y. 오경보 계열은 낮을수록 좋으므로 "어느 팔이 나은가"로 적는다
                      "verdict": ("미판정" if lo <= 0 <= hi else
                                  f"{x} 우위" if (lo > 0) != k.startswith(("fpr", "deploy_fpr")) else f"{y} 우위")}
        contrasts[f"{x}-{y}"] = row

    report = {
        "experiment": "E8b",
        "heads": COMMON,
        "population": f"gold clean {len(gold)} · 음성 홀드아웃 {len(neg)}({len(n_groups)}문서)",
        "boot": f"gold·홀드아웃 문서 군집 재표집 {N_BOOT}회, 문턱 재표집마다 재설정",
        "constant": {"auroc": 0.5, "f1_note": "조 F1 상수(제6·8·9조) 36.1% — gold 255"},
        "reference_article_v2": _metrics(*ref, labels, gi0, ni0) if ref else None,
        "arms": arms,
        "contrasts": contrasts,
        "per_model": per_model,
    }
    save_json(report, OUT_PATH)

    logger.info(f"  {'팔':<7}{'n':>2}  {'dep재현':>7}{'dep오경보':>9}{'AUROC':>8}{'FPR@80':>8}{'FPR@85':>8}{'FPR@90':>8}{'F1@90':>7}")
    for a, s in arms.items():
        logger.info(f"  {a:<7}{s['n_seeds']:>2}  {s['deploy_recall']['mean']*100:>6.1f}%{s['deploy_fpr']['mean']*100:>8.1f}%"
                    f"{s['auroc']['mean']:>8.3f}{s['fpr@80']['mean']*100:>7.1f}%{s['fpr@85']['mean']*100:>7.1f}%"
                    f"{s['fpr@90']['mean']*100:>7.1f}%{s['f1@90']['mean']*100:>6.1f}%")
    for c, r in contrasts.items():
        logger.info(f"  {c:<12} {r['desc']}")
        for k in ("deploy_recall", "deploy_fpr", "auroc", "fpr@80", "fpr@85", "fpr@90", "deploy_f1", "f1@90"):
            v, t = r[k], r["_train"][k]
            logger.info(f"      {k:<10} {v['diff_pp']:+6.2f}%p  테스트만 {v['ci95_pp']} {v['verdict']:<6} | "
                        f"학습변동 포함 {t['ci95_pp']} MDE {t['mde80_pp']} → {t['verdict']}")
    logger.info(f"저장: {OUT_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        cache_predictions(args.gpu)
    analyze()


if __name__ == "__main__":
    main()
