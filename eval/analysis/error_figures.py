# eval/analysis/error_figures.py
"""조별 혼동 행렬 + 재현–오경보 / PR 곡선 (오류 분석 절, 2026-10-04)

## 무엇을 그리나

1. **조별 혼동 행렬** — gold clean 255(사건당 근거 1개 = 정답 조 1개). 행 = 정답 조, 열 = 모델이 낸 조
   (multi-label이라 한 건이 여러 열에 들어갈 수 있다) + "지목 없음". KoELECTRA 배포 임계값과
   gpt-4o forward(teacher) 둘 다. 행 정규화 값도 같이 낸다
2. **재현–오경보 곡선 (조항 단위)** — gold 255(위반) vs 홀드아웃 300(표준계약서). 순위 점수
   r = max(p/thr)를 훑는다. **유병률과 무관**해서 이 프로젝트의 주 곡선이다
3. **PR 곡선 (조항 단위)** — 정밀도는 실제 약관의 위반 비율 π에 달려 있다(π는 미측정). 평가셋의
   위반 비율(255/555 = 46%)로 그리면 실제보다 훨씬 좋아 보이므로 **π = 5 / 10 / 15%를 가정해** 재현·오경보에서
   유도한다: precision = R·π / (R·π + FPR·(1−π))
4. **조별 PR 곡선 (조 식별)** — 각 조에 대해 양성 = 그 조가 정답인 gold, 음성 = 나머지 gold + 홀드아웃.
   조별 AP와 support를 같이 적는다

## 실행

    .venv/bin/python -m eval.analysis.error_figures

입력: `data/eval/model/e8b_cache/*.npz`(E8b 예측 캐시), `data/eval/label/teacher_on_gold_raw.jsonl`,
FB-Check 산출물(홀드아웃 조항의 gpt-4o forward). TF-IDF는 `e_base_baselines`와 같은 절차로 다시 학습한다.
출력: `data/eval/analysis/figures/*.png`, `data/eval/analysis/error_figures.json`
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from backend.training.retrain import eval_sets  # noqa: E402
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json  # noqa: E402
from eval.model.e8b_matched import CACHE_DIR, COMMON  # noqa: E402

logger = load_logger("error_figures.log")

HOLDOUT  = PROJECT_ROOT / "backend/training/negative_holdout_docs.json"
RAW      = PROJECT_ROOT / "data/eval/label/teacher_on_gold_raw.jsonl"
FB_PATHS = [PROJECT_ROOT / "data/fb_check/clean.jsonl", PROJECT_ROOT / "data/fb_check/noise.jsonl"]
FIG_DIR  = PROJECT_ROOT / "data/eval/analysis/figures"
OUT_PATH = PROJECT_ROOT / "data/eval/analysis/error_figures.json"

# 시각화 규칙: 범주색은 고정 순서, 텍스트는 텍스트 색, 연속값은 단일 색상(밝음→어두움)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f7fc", "#9cc0ec", "#2a78d6", "#0f3a70"])
PRIORS = [0.05, 0.10, 0.15]

plt.rcParams.update({
    "font.family": "WenQuanYi Zen Hei", "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


# ── 입력 ─────────────────────────────────────────────────────────────────────

def _load_cache(name: str):
    z = np.load(CACHE_DIR / f"{name}.npz")
    names = list(z["names"])
    cols = [names.index(a) for a in COMMON]
    return z["P"][:, cols], z["Pn"][:, cols], z["thr"][cols]


def _tfidf_probs(gold, neg):
    """e_base_baselines의 balanced + dev 임계값 TF-IDF를 같은 절차로 다시 학습한다."""
    from backend.eval.label.e_base_baselines import HEADS, _train_tfidf_lr, _training_split
    from backend.training.train_article import tune_thresholds
    assert HEADS == COMMON
    train, dev = _training_split()
    vec, clf, mlb = _train_tfidf_lr(train, balanced=True)
    dev_p = clf.predict_proba(vec.transform([r["text"] for r in dev]))
    dev_y = np.array([[a in r["articles"] for a in mlb.classes_] for r in dev], dtype=float)
    thr = tune_thresholds(dev_p, dev_y, list(mlb.classes_))
    return (clf.predict_proba(vec.transform([g["text"] for g in gold])),
            clf.predict_proba(vec.transform([r["text"] for r in neg])), thr)


# ── 1. 혼동 행렬 ─────────────────────────────────────────────────────────────

def confusion(pred_sets: list[set], true: list[str]) -> dict:
    cols = COMMON + ["지목 없음"]
    M = np.zeros((len(COMMON), len(cols)), dtype=int)
    for t, p in zip(true, pred_sets):
        i = COMMON.index(t)
        for a in p:
            M[i, COMMON.index(a)] += 1
        if not p:
            M[i, -1] += 1
    support = np.array([true.count(a) for a in COMMON])
    exact = np.array([sum(1 for t, p in zip(true, pred_sets) if t == a and p == {a}) for a in COMMON])
    # 제6조(일반원칙) 흡수: 정답이 6조가 아닌데 6조만 낸 건
    only6 = sum(1 for t, p in zip(true, pred_sets) if t != "제6조" and p == {"제6조"})
    non6 = sum(1 for t in true if t != "제6조")
    return {"cols": cols, "rows": COMMON, "matrix": M.tolist(), "support": support.tolist(),
            "hit_rate": [round(M[i, i] / support[i], 4) if support[i] else None for i in range(len(COMMON))],
            "exact_only_true": exact.tolist(),
            "absorbed_into_art6": {"n": only6, "of_non6": non6, "rate": round(only6 / max(non6, 1), 4)},
            "mean_preds_per_item": round(float(np.mean([len(p) for p in pred_sets])), 3)}


def plot_confusions(cms: dict[str, dict], path) -> None:
    fig, axes = plt.subplots(1, len(cms), figsize=(6.4 * len(cms), 5.6), constrained_layout=True)
    for ax, (title, cm) in zip(np.atleast_1d(axes), cms.items()):
        M = np.array(cm["matrix"], dtype=float)
        sup = np.array(cm["support"], dtype=float)
        R = M / np.maximum(sup[:, None], 1)                      # 행 정규화: 정답 조 건수 대비
        ax.imshow(R, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
        ax.grid(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j]:
                    ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=8.5,
                            color=SURFACE if R[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(cm["cols"])), cm["cols"], rotation=45, ha="right")
        ax.set_yticks(range(len(cm["rows"])), [f"{r} (n={int(s)})" for r, s in zip(cm["rows"], sup)])
        ax.set_xlabel("모델이 낸 조 (한 건이 여러 칸에 들어갈 수 있음)")
        ax.set_ylabel("정답 조 (공정위 근거_법령)")
        a6 = cm["absorbed_into_art6"]
        ax.set_title(f"{title}\n제6조 흡수 {a6['n']}/{a6['of_non6']}건 · 건당 평균 {cm['mean_preds_per_item']}개 조",
                     fontsize=11, loc="left", color=INK)
    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.01)
    cb.set_label("정답 조 건수 대비 비율 (칸 숫자 = 건수)", color=INK2)
    fig.suptitle("조별 혼동 행렬 — gold clean 255 (사건당 근거 1개)", x=0.01, ha="left", fontsize=13)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ── 2·3. 조항 단위 곡선 ──────────────────────────────────────────────────────

def clause_curve(P, Pn, thr) -> dict:
    rg, rn = (P / thr).max(1), (Pn / thr).max(1)
    cuts = np.unique(np.concatenate([rg, rn, [np.inf]]))[::-1]
    rec = np.array([(rg >= c).mean() for c in cuts])
    fpr = np.array([(rn >= c).mean() for c in cuts])
    deploy = {"recall": float((rg >= 1).mean()), "fpr": float((rn >= 1).mean())}
    return {"recall": rec, "fpr": fpr, "deploy": deploy}


def precision_at(rec, fpr, pi):
    den = rec * pi + fpr * (1 - pi)
    return np.where(den > 0, rec * pi / np.maximum(den, 1e-12), np.nan)


def plot_clause_curves(curves: dict, points: dict, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    ax = axes[0]
    for k, (name, c) in enumerate(curves.items()):
        ax.plot(c["fpr"] * 100, c["recall"] * 100, color=SERIES[k], lw=2, label=name,
                drawstyle="steps-post")
        ax.plot(c["deploy"]["fpr"] * 100, c["deploy"]["recall"] * 100, "o", ms=8, color=SERIES[k],
                mec=SURFACE, mew=2)
    for k, (name, (f, r)) in enumerate(points.items(), start=len(curves)):
        ax.plot(f * 100, r * 100, "D", ms=9, color=SERIES[k], mec=SURFACE, mew=2, label=name)
    ax.set_xlim(0, 40)
    ax.set_ylim(40, 100)
    ax.set_xlabel("홀드아웃 오경보율 (%) — 표준계약서 300조항")
    ax.set_ylabel("조항 재현 (%) — gold 255")
    ax.set_title("재현–오경보 (유병률 무관) · 동그라미 = 각자의 배포 임계값", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    styles = {0.05: ":", 0.10: "-", 0.15: "--"}
    for k, (name, c) in enumerate(list(curves.items())[:2]):          # 두 모델만 — 선이 겹치지 않게
        for pi in PRIORS:
            pr = precision_at(c["recall"], c["fpr"], pi)
            ax.plot(c["recall"] * 100, pr * 100, color=SERIES[k], lw=2 if pi == 0.10 else 1.3,
                    ls=styles[pi], label=f"{name} · π={int(pi * 100)}%")
    ax.set_xlim(40, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("조항 재현 (%)")
    ax.set_ylabel("정밀도 (%) — 가정한 위반 비율 π에서 유도")
    ax.set_title("PR 곡선 (π 가정) · 실제 약관의 π는 미측정", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle("조항 단위 곡선 — 위반 조항을 짚었나", x=0.01, ha="left", fontsize=13)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ── 4. 조별 PR (조 식별) ─────────────────────────────────────────────────────

def article_pr(P, Pn, true: list[str]) -> dict:
    out = {}
    for j, a in enumerate(COMMON):
        s = np.concatenate([P[:, j], Pn[:, j]])
        y = np.concatenate([[t == a for t in true], np.zeros(len(Pn), bool)])
        order = np.argsort(-s, kind="stable")
        ys = y[order]
        tp = np.cumsum(ys)
        prec = tp / np.arange(1, len(ys) + 1)
        rec = tp / max(y.sum(), 1)
        ap = float((prec * ys).sum() / max(y.sum(), 1))
        out[a] = {"precision": prec, "recall": rec, "ap": ap, "support": int(y.sum()),
                  "base_rate": float(y.mean())}
    return out


def plot_article_pr(pr: dict, path) -> None:
    """조별 칸 나누기(2×4). 한 판에 8개를 겹치면 읽을 수 없다. 점선 = 기저율(무작위 분류기의 정밀도)."""
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), sharex=True, sharey=True, constrained_layout=True)
    for k, (ax, (a, d)) in enumerate(zip(axes.ravel(), pr.items())):
        ax.plot(d["recall"] * 100, d["precision"] * 100, color=SERIES[0], lw=2)
        ax.axhline(d["base_rate"] * 100, color=INK2, lw=1, ls=":")
        ax.set_title(f"{a}  AP {d['ap'] * 100:.0f}%  (n={d['support']})", loc="left", fontsize=10.5)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 102)
        if k % 4 == 0:
            ax.set_ylabel("정밀도 (%)")
        if k >= 4:
            ax.set_xlabel("재현 (%)")
    fig.suptitle("조별 PR — KoELECTRA article_v2 · 양성 = 그 조가 정답인 gold, 음성 = 나머지 gold + 홀드아웃 "
                 "· 점선 = 기저율", x=0.01, ha="left", fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    gold, neg = eval_sets(json.loads(HOLDOUT.read_text(encoding="utf-8")))
    true = []
    for g in gold:
        arts = [a for a in g["articles"] if a in COMMON]
        assert len(arts) == 1, f"clean 지층인데 정답 조가 1개가 아니다: {g['articles']}"
        true.append(arts[0])

    P, Pn, thr = _load_cache("article_v2")
    k_sets = [{a for a, b in zip(COMMON, row) if b} for row in (P >= thr)]
    raw = {r["key"]: r for r in load_jsonl(RAW)}
    t_sets = [{a for a in raw[g["doc_id"] + "|" + g["text"][:40]]["articles"] if a in COMMON} for g in gold]
    cms = {"KoELECTRA article_v2 (배포 임계값)": confusion(k_sets, true),
           "gpt-4o forward (teacher)": confusion(t_sets, true)}
    plot_confusions(cms, FIG_DIR / "confusion_articles.png")

    tP, tPn, tthr = _tfidf_probs(gold, neg)
    curves = {"KoELECTRA article_v2": clause_curve(P, Pn, thr),
              "TF-IDF+LR (balanced)": clause_curve(tP, tPn, tthr)}
    for arm, label in (("C", "E8 C · FB-Check 학습 (3시드 평균)"), ("A", "E8 A · 필터 없음 학습 (3시드 평균)")):
        cs = [clause_curve(*_load_cache(f"{arm}_s{s}")) for s in (42, 7, 123)]
        grid = np.linspace(0, 1, 301)                              # 오경보 축에서 재현을 보간해 평균
        rec = np.mean([np.interp(grid, c["fpr"], c["recall"]) for c in cs], axis=0)
        curves[label] = {"recall": rec, "fpr": grid,
                         "deploy": {k: float(np.mean([c["deploy"][k] for c in cs])) for k in ("recall", "fpr")}}
    fb = {r["chunk_id"]: r for p in FB_PATHS for r in load_jsonl(p)}
    gpt_fpr = float(np.mean([bool([a for a in (fb[r["chunk_id"]].get("forward_articles") or []) if a in COMMON])
                             for r in neg]))
    gpt_rec = float(np.mean([bool(s) for s in t_sets]))
    points = {"gpt-4o forward (임계값 없음)": (gpt_fpr, gpt_rec)}
    plot_clause_curves(curves, points, FIG_DIR / "clause_recall_fpr_pr.png")

    apr = article_pr(P, Pn, true)
    plot_article_pr(apr, FIG_DIR / "article_pr_koelectra.png")

    def _summ(c):
        out = {"deploy": c["deploy"]}
        for r_t in (0.80, 0.85, 0.90):
            i = int(np.argmax(c["recall"] >= r_t))
            out[f"fpr@{int(r_t * 100)}"] = round(float(c["fpr"][i]), 4)
            out[f"precision@{int(r_t * 100)}"] = {f"pi{int(pi * 100)}": round(float(precision_at(
                c["recall"][i], c["fpr"][i], pi)), 4) for pi in PRIORS}
        return out

    report = {
        "population": f"gold clean {len(gold)} · 홀드아웃 {len(neg)}",
        "confusion": cms,
        "clause_curves": {k: _summ(c) for k, c in curves.items()},
        "gpt_point": {"recall": gpt_rec, "fpr": gpt_fpr,
                      "precision": {f"pi{int(pi * 100)}": round(float(precision_at(gpt_rec, gpt_fpr, pi)), 4)
                                    for pi in PRIORS}},
        "article_pr_koelectra": {a: {"ap": round(d["ap"], 4), "support": d["support"],
                                     "base_rate": round(d["base_rate"], 4)} for a, d in apr.items()},
        "note": "정밀도는 가정한 π에서 유도한 값 — 실제 약관 위반 비율은 미측정이라 절대값 인용 금지",
        "figures": [str(p.relative_to(PROJECT_ROOT)) for p in sorted(FIG_DIR.glob("*.png"))],
    }
    save_json(report, OUT_PATH)
    logger.info(json.dumps({k: v for k, v in report.items() if k != "confusion"}, ensure_ascii=False, indent=1))
    for t, cm in cms.items():
        logger.info(f"[{t}] 적중 {cm['hit_rate']} | 제6조 흡수 {cm['absorbed_into_art6']}")


if __name__ == "__main__":
    main()
