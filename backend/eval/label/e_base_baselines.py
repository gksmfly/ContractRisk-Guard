# backend/eval/label/e_base_baselines.py
"""
E_base — 기준선 보강: 규칙 기반 / TF-IDF+LR / (c는 E9 완료 후 Qwen zero-shot으로 추가)

## 목적
ICAIL 2025, Text to Trust 등 선행 연구의 표준 비교 축을 채운다.
KoELECTRA(E6/E8)와 GPT 직접 추론(E_teacher)이 이 기준선을 얼마나 넘는지를 한 표로.

## 기준선 두 가지
(a) 규칙 기반 — 제6~14조 각 조항의 핵심 키워드가 clause 텍스트에 있으면 해당 조 지목
(b) TF-IDF + 로지스틱 회귀 — clean.jsonl agreed 라벨로 학습, gold clean 255에서 채점

## 평가 지표
- per-sample F1 (article_gold_eval과 동일한 방식)
- 상수 기준선(제6·8·9조 고정) 병기 필수
- 상수 대비 Δ와 부트스트랩 95% CI

## 판정 규칙 (사전 등록)
- Δ가 95% CI 포함 0이면 → "미판정, MDE ±4.3%p" 보고
- 규칙 기반은 특성상 임계값 없음 — 정밀 / 재현 / F1 모두 보고
- TF-IDF+LR 임계값: **dev split에서 조별 튜닝**(train_article.tune_thresholds와 같은 규칙).
  gold는 여전히 쓰지 않는다

## 2026-09-30 수정 (P1-3)
1. **TF-IDF 학습에서 gold 문서를 배제하지 않았다.** 학습 파이프라인(`train_article`)과 같은
   순서로 gold 문서 배제 → 음성 홀드아웃 분리 → 문서 단위 dev 분할을 한다
2. **임계값 0.5 고정은 허수아비였다.** 불균형 multi-label에서 희소 조는 0.5를 못 넘는다
   (`tune_thresholds` 주석). dev에서 조별로 고른다. 옛 0.5 값도 추적용으로 병기한다
3. **KoELECTRA 참조값 45.75%는 교차적합 값이었다** — 분포 맞는 보정셋이 있다는 가정 위의
   값이라 성능 주장에 쓰지 않기로 한 수치다. 배포 임계값 값(38.5%, 상수 대비 미판정)으로 바꾼다
4. 제품이 주장하는 축(조항 재현 · 홀드아웃 오경보)을 모든 기준선에 같이 잰다

실행:
    .venv/bin/python -m backend.eval.label.e_base_baselines
"""

import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from backend.labeling.articles import ARTICLE_IDS
from backend.training.retrain import eval_sets
from backend.training.train import split_by_document
from backend.training.train_article import (
    constant_baseline_f1,
    exclude_gold_documents,
    load_article_records,
    load_ftc_gold,
    split_negative_holdout,
    tune_thresholds,
)
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("e_base_baselines.log")
OUT_PATH = PROJECT_ROOT / "data/eval/label/e_base_baselines.json"

LABELED_PATH = PROJECT_ROOT / "data/fb_check/clean.jsonl"
HOLDOUT_PATH = PROJECT_ROOT / "backend/training/negative_holdout_docs.json"
GOLD_REPORT  = PROJECT_ROOT / "data/eval/gold/article_gold_report.json"
# article_v2와 같은 8개 헤드(제13조는 support 부족으로 접힘)
HEADS = ["제6조", "제7조", "제8조", "제9조", "제10조", "제11조", "제12조", "제14조"]

# ─── (a) 규칙 기반 키워드 ────────────────────────────────────────────────────
# 각 조항의 핵심 법률 용어. 더 넓게 잡으면 재현 오르고 정밀 내림.
# 이 목록 자체가 "약관규제법 조항 구분의 최소 기준선"이다.
_ARTICLE_KEYWORDS: dict[str, list[str]] = {
    "제6조": ["불공정", "고객에게 부당하게 불리", "신의성실", "공정성을 잃은"],
    "제7조": ["면책", "손해배상 책임", "고의 또는 중과실"],
    "제8조": ["손해배상액", "위약금", "과도", "예정"],
    "제9조": ["계약의 해제", "계약의 해지", "원상회복", "해지 위약금"],
    "제10조": ["채권양도", "양도 금지", "채무인수"],
    "제11조": ["고객의 항변권", "상계", "공제"],
    "제12조": ["소송 제기", "관할법원", "재판관할"],
    "제13조": ["대리인", "이행보조자"],
    "제14조": ["의사표시", "통지", "통보", "간주"],
}


def _rule_predict(text: str) -> list[str]:
    """텍스트에 해당 조항 키워드가 하나라도 있으면 그 조를 지목한다."""
    matched = []
    for article, kws in _ARTICLE_KEYWORDS.items():
        if any(kw in text for kw in kws):
            matched.append(article)
    return sorted(matched, key=lambda a: int(a.strip("제조")))


# ─── 공통 — 건별 F1 ──────────────────────────────────────────────────────────

def _per_sample_f1(preds: list[list[str]], labels: list[list[str]]) -> np.ndarray:
    f1s = []
    for p_set, y_set in zip(preds, labels):
        p = set(p_set)
        y = set(y_set)
        tp = len(p & y)
        pr = tp / len(p) if p else 0.0
        rc = tp / len(y) if y else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return np.array(f1s)


def _bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 9999) -> tuple[float, list]:
    """두 건별 F1 배열의 페어드 Δ와 95% CI (복원추출)."""
    rng = np.random.default_rng(42)
    n = len(a)
    obs_delta = float(a.mean() - b.mean())
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(float(a[idx].mean() - b[idx].mean()))
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return obs_delta, [lo, hi]


# ─── (b) TF-IDF + LR ─────────────────────────────────────────────────────────

def _train_tfidf_lr(train_records: list[dict], balanced: bool) -> tuple:
    texts  = [r["text"] for r in train_records]
    labels = [[a for a in r["articles"] if a in HEADS] for r in train_records]

    mlb = MultiLabelBinarizer(classes=HEADS)
    Y = mlb.fit_transform(labels)

    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), max_features=50_000, sublinear_tf=True,
    )
    X = vec.fit_transform(texts)

    clf = OneVsRestClassifier(
        LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                           class_weight="balanced" if balanced else None),
        n_jobs=-1,
    )
    clf.fit(X, Y)
    return vec, clf, mlb


def _clause_axes(pred_gold: list[list[str]], pred_neg: list[list[str]]) -> dict:
    """제품 축: gold 조항 재현(조가 하나라도 나왔나) · 홀드아웃 오경보(하나라도 나왔나)."""
    return {"clause_recall": round(float(np.mean([bool(p) for p in pred_gold])), 4),
            "holdout_fpr":   round(float(np.mean([bool(p) for p in pred_neg])), 4)}


def _training_split() -> tuple[list[dict], list[dict]]:
    """train_article과 같은 순서: gold 문서 배제 → 음성 홀드아웃 분리 → 문서 단위 dev 분할."""
    gold = load_ftc_gold(stratum="clean")
    records = load_article_records(LABELED_PATH, augment=False, label_source="agreed")
    records = exclude_gold_documents(records, gold)
    fixed = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
    records, _ = split_negative_holdout(records, len(fixed), 42, fixed)
    for r in records:
        r["has_article"] = bool([a for a in r["articles"] if a in HEADS])
    return split_by_document(records, 0.2, 42, stratify_key="has_article")


def _verdict(metric: str, lo: float, hi: float) -> str:
    """오경보(fpr*)는 낮을수록, 나머지는 높을수록 좋다. 차 = KoELECTRA − TF-IDF."""
    if lo <= 0 <= hi:
        return "미판정"
    koelectra_higher = lo > 0
    return "KoELECTRA 우위" if koelectra_higher != metric.startswith(("fpr", "deploy_fpr")) else "TF-IDF 우위"


def _matched_vs_koelectra(tfidf_probs: tuple, gold: list[dict], neg: list[dict]) -> dict:
    """TF-IDF(balanced, dev 임계값) vs article_v2를 **같은 재현율**에서 비교(규칙 9).

    article_v2 예측은 E8b 캐시(`data/eval/model/e8b_cache/article_v2.npz`, 같은 gold·홀드아웃 순서)를
    쓴다. 지표·재표집은 E8b와 같다(gold·홀드아웃 문서 군집 부트스트랩, 문턱 재설정).
    """
    from eval.model.e8b_matched import CACHE_DIR, COMMON, _groups, _load, _metrics

    if not (CACHE_DIR / "article_v2.npz").exists() or COMMON != HEADS:
        return {"note": "E8b 캐시 없음 — eval.model.e8b_matched 먼저 실행"}
    labels = np.array([[a in g["articles"] for a in HEADS] for g in gold])
    kP, kPn, kthr = _load("article_v2")
    tP, tPn, tthr = tfidf_probs
    g_groups = _groups([g["doc_id"] for g in gold])
    n_groups = _groups([":".join(str(r["chunk_id"]).split(":")[:2]) for r in neg])
    gi0, ni0 = np.arange(len(gold)), np.arange(len(neg))
    base = {"koelectra": _metrics(kP, kPn, kthr, labels, gi0, ni0),
            "tfidf": _metrics(tP, tPn, tthr, labels, gi0, ni0)}
    rng = np.random.default_rng(42)
    diffs: dict[str, list[float]] = {k: [] for k in base["tfidf"]}
    for _ in range(2000):
        gi = np.concatenate([g_groups[j] for j in rng.integers(0, len(g_groups), len(g_groups))])
        ni = np.concatenate([n_groups[j] for j in rng.integers(0, len(n_groups), len(n_groups))])
        a = _metrics(kP, kPn, kthr, labels, gi, ni)
        b = _metrics(tP, tPn, tthr, labels, gi, ni)
        for k in diffs:
            diffs[k].append(a[k] - b[k])
    out = {"koelectra": base["koelectra"], "tfidf": base["tfidf"], "koelectra_minus_tfidf": {}}
    for k, d in diffs.items():
        lo, hi = np.percentile(d, [2.5, 97.5])
        out["koelectra_minus_tfidf"][k] = {
            "diff_pp": round((base["koelectra"][k] - base["tfidf"][k]) * 100, 2),
            "ci95_pp": [round(float(lo) * 100, 2), round(float(hi) * 100, 2)],
            "verdict": _verdict(k, lo, hi)}
    logger.info(f"  같은 재현율 비교(KoELECTRA − TF-IDF): {out['koelectra_minus_tfidf']}")
    return out


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def run() -> dict:
    gold = load_ftc_gold(stratum="clean")           # 255건
    logger.info(f"gold 255 로드: {len(gold)}건")

    train_records, dev_records = _training_split()
    logger.info(f"훈련 {len(train_records)} / dev {len(dev_records)} (gold 문서·음성 홀드아웃 배제 후)")
    _, neg = eval_sets(json.loads(HOLDOUT_PATH.read_text(encoding="utf-8")))
    neg_texts = [r["text"] for r in neg]
    logger.info(f"음성 홀드아웃: {len(neg)}건")

    gold_labels = [r["articles"] for r in gold]
    gold_texts  = [r["text"]     for r in gold]

    # 상수 기준선 — 전체 gold에서 먼저 측정 (제6·8·9조 고정, top_n=3)
    names = list(ARTICLE_IDS)
    const_result = constant_baseline_f1(gold, names=names, k=3)
    const_mean = const_result["f1"]
    const_articles = const_result["articles"]
    logger.info(f"상수 기준선({const_articles}) mean F1: {const_mean:.4f}")
    const_f1s = np.array([
        _per_sample_f1([const_articles], [r["articles"]])[0] for r in gold
    ])

    # ── (a) 규칙 기반 ──────────────────────────────────────────────────────────
    rule_preds = [_rule_predict(t) for t in gold_texts]
    rule_f1s   = _per_sample_f1(rule_preds, gold_labels)
    rule_mean  = float(rule_f1s.mean())
    rule_delta, rule_ci = _bootstrap_ci(rule_f1s, np.array(const_f1s))

    # 정밀/재현 (매크로)
    precs, recs = [], []
    for p_set, y_set in zip(rule_preds, gold_labels):
        p, y = set(p_set), set(y_set)
        tp = len(p & y)
        precs.append(tp / len(p) if p else 0.0)
        recs.append(tp / len(y) if y else 0.0)
    rule_prec = float(np.mean(precs))
    rule_rec  = float(np.mean(recs))

    logger.info(
        f"(a) 규칙 기반: mean F1={rule_mean:.4f} Δ={rule_delta:+.4f} "
        f"CI=[{rule_ci[0]:.4f},{rule_ci[1]:.4f}]"
    )

    rule_axes = _clause_axes(rule_preds, [_rule_predict(t) for t in neg_texts])

    # ── (b) TF-IDF + LR ────────────────────────────────────────────────────────
    tfidf, probs = {}, {}
    for variant, balanced, tuned in (("fixed05", False, False),       # 옛 설정(추적용)
                                     ("dev_tuned", False, True),
                                     ("balanced_dev_tuned", True, True)):
        vec, clf, mlb = _train_tfidf_lr(train_records, balanced)
        if tuned:
            dev_p = clf.predict_proba(vec.transform([r["text"] for r in dev_records]))
            dev_y = np.array([[a in r["articles"] for a in mlb.classes_] for r in dev_records], dtype=float)
            thr = tune_thresholds(dev_p, dev_y, list(mlb.classes_))
        else:
            thr = np.full(len(mlb.classes_), 0.5)

        def _pred(texts: list[str]) -> list[list[str]]:
            pb = clf.predict_proba(vec.transform(texts)) >= thr
            return [[mlb.classes_[j] for j in range(len(mlb.classes_)) if pb[i, j]] for i in range(len(texts))]

        preds = _pred(gold_texts)
        f1s = _per_sample_f1(preds, gold_labels)
        probs[variant] = (clf.predict_proba(vec.transform(gold_texts)),
                          clf.predict_proba(vec.transform(neg_texts)), thr)
        delta, ci = _bootstrap_ci(f1s, np.array(const_f1s))
        tfidf[variant] = {
            "class_weight": "balanced" if balanced else None,
            "thresholds": {a: round(float(t), 2) for a, t in zip(mlb.classes_, thr)},
            "mean_f1": round(float(f1s.mean()), 4),
            "vs_constant_delta_pp": round(delta * 100, 2),
            "vs_constant_ci95_pp": [round(v * 100, 2) for v in ci],
            "verdict": "pass" if ci[0] > 0 else ("below" if ci[1] < 0 else "undecided"),
            **_clause_axes(preds, _pred(neg_texts)),
        }
        logger.info(f"(b) TF-IDF+LR [{variant}]: F1={f1s.mean():.4f} Δ={delta:+.4f} "
                    f"CI=[{ci[0]:.4f},{ci[1]:.4f}] {tfidf[variant]}")

    rep = json.loads(GOLD_REPORT.read_text(encoding="utf-8"))
    matched = _matched_vs_koelectra(probs["balanced_dev_tuned"], gold, neg)

    result = {
        "population": "FTC gold clean 255",
        "constant_baseline": {
            "articles": const_articles,
            "mean_f1": round(const_mean, 4),
        },
        "rule_based": {
            "mean_f1": round(rule_mean, 4),
            "macro_precision": round(rule_prec, 4),
            "macro_recall": round(rule_rec, 4),
            "vs_constant_delta_pp": round(rule_delta * 100, 2),
            "vs_constant_ci95_pp": [round(v * 100, 2) for v in rule_ci],
            "verdict": "pass" if rule_ci[0] > 0 else "undecided",
            "note": "임계값 없음. 키워드 hit → 조 지목",
            **rule_axes,
        },
        "tfidf_lr": {
            "train_n": len(train_records),
            "dev_n": len(dev_records),
            "train_source": "clean.jsonl agreed_articles (gold 문서·음성 홀드아웃 배제, 문서 단위 dev 분할)",
            **tfidf,
        },
        "matched_vs_koelectra": matched,
        "koelectra_reference": {
            "deploy": {"mean_f1": 0.385, "vs_constant": "+2.4%p [-4.0,+9.1] 미판정",
                       "clause_recall": 0.784, "holdout_fpr": 0.053},
            "crossfit_note": "45.8%(교차적합)는 분포 맞는 보정셋을 가정한 값 — 성능 주장 금지, 참고로만",
            "source": str(GOLD_REPORT.relative_to(PROJECT_ROOT)),
            "report_keys": sorted(rep)[:20],
        },
    }

    save_json(result, OUT_PATH)
    logger.info(f"결과 저장: {OUT_PATH}")

    # 요약 출력
    print("\n=== E_base 결과 요약 ===")
    print("모집단: gold clean 255")
    print(f"상수 기준선(제6·8·9조): {const_mean*100:.2f}%")
    print(f"(a) 규칙 기반:     {rule_mean*100:.2f}%  Δ={rule_delta*100:+.2f}%p  CI=[{rule_ci[0]*100:.2f},{rule_ci[1]*100:.2f}]")
    for k, v in tfidf.items():
        print(f"(b) TF-IDF+LR {k:<20} {v['mean_f1']*100:.2f}%  Δ={v['vs_constant_delta_pp']:+.2f}%p  "
              f"CI={v['vs_constant_ci95_pp']}  재현 {v['clause_recall']*100:.1f}%  오경보 {v['holdout_fpr']*100:.1f}%")
    print(f"규칙 기반 재현 {rule_axes['clause_recall']*100:.1f}%  오경보 {rule_axes['holdout_fpr']*100:.1f}%")
    print("KoELECTRA(배포 임계값): 38.5%  재현 78.4%  오경보 5.3%  (상수 대비 +2.4%p 미판정)")

    return result


if __name__ == "__main__":
    run()
