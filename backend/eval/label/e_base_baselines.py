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
- TF-IDF+LR 임계값: 0.5 고정 (튜닝 없음 — held-out gold 미사용 원칙)

실행:
    .venv/bin/python -m backend.eval.label.e_base_baselines
"""

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from backend.labeling.articles import ARTICLE_IDS
from backend.training.train_article import (
    _MAJORITY_ARTICLE,
    constant_baseline_f1,
    load_article_records,
    load_ftc_gold,
)
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("e_base_baselines.log")
OUT_PATH = PROJECT_ROOT / "data/eval/e_base_baselines.json"

LABELED_PATH = PROJECT_ROOT / "data/fb_check/clean.jsonl"

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

def _train_tfidf_lr(train_records: list[dict]) -> tuple:
    texts  = [r["text"] for r in train_records]
    labels = [r["articles"] for r in train_records]

    mlb = MultiLabelBinarizer(classes=list(ARTICLE_IDS))
    Y = mlb.fit_transform(labels)

    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), max_features=50_000, sublinear_tf=True,
    )
    X = vec.fit_transform(texts)

    clf = OneVsRestClassifier(
        LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
        n_jobs=-1,
    )
    clf.fit(X, Y)
    return vec, clf, mlb


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def run() -> dict:
    gold = load_ftc_gold(stratum="clean")           # 255건
    logger.info(f"gold 255 로드: {len(gold)}건")

    train_records = load_article_records(LABELED_PATH, augment=False, label_source="agreed")
    logger.info(f"훈련 데이터 로드: {len(train_records)}건 (clean.jsonl agreed)")

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
    precs, recs, f1s_per = [], [], []
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

    # ── (b) TF-IDF + LR ────────────────────────────────────────────────────────
    vec, clf, mlb = _train_tfidf_lr(train_records)
    X_test  = vec.transform(gold_texts)
    proba   = clf.predict_proba(X_test)
    thr     = 0.5                                    # 고정 (튜닝 없음)
    tfidf_preds_bin = (proba >= thr).astype(int)
    tfidf_preds = [
        [mlb.classes_[j] for j in range(len(mlb.classes_)) if tfidf_preds_bin[i, j]]
        for i in range(len(gold))
    ]
    tfidf_f1s  = _per_sample_f1(tfidf_preds, gold_labels)
    tfidf_mean = float(tfidf_f1s.mean())
    tfidf_delta, tfidf_ci = _bootstrap_ci(tfidf_f1s, np.array(const_f1s))

    logger.info(
        f"(b) TF-IDF+LR: mean F1={tfidf_mean:.4f} Δ={tfidf_delta:+.4f} "
        f"CI=[{tfidf_ci[0]:.4f},{tfidf_ci[1]:.4f}]"
    )

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
        },
        "tfidf_lr": {
            "threshold": thr,
            "train_n": len(train_records),
            "train_source": "clean.jsonl agreed_articles",
            "mean_f1": round(tfidf_mean, 4),
            "vs_constant_delta_pp": round(tfidf_delta * 100, 2),
            "vs_constant_ci95_pp": [round(v * 100, 2) for v in tfidf_ci],
            "verdict": "pass" if tfidf_ci[0] > 0 else "undecided",
            "note": "임계값 0.5 고정 (held-out gold 미사용 원칙)",
        },
        "koelectra_reference": {
            "mean_f1": 0.4575,
            "source": "reported_numbers.md §1 crossfit F1 (article_gold_report.json)",
            "note": "동일 gold 255 평가셋. 비교 기준점",
        },
    }

    save_json(result, OUT_PATH)
    logger.info(f"결과 저장: {OUT_PATH}")

    # 요약 출력
    print("\n=== E_base 결과 요약 ===")
    print(f"모집단: gold clean 255")
    print(f"상수 기준선(제6·8·9조): {const_mean*100:.2f}%")
    print(f"(a) 규칙 기반:     {rule_mean*100:.2f}%  Δ={rule_delta*100:+.2f}%p  CI=[{rule_ci[0]*100:.2f},{rule_ci[1]*100:.2f}]")
    print(f"(b) TF-IDF+LR:    {tfidf_mean*100:.2f}%  Δ={tfidf_delta*100:+.2f}%p  CI=[{tfidf_ci[0]*100:.2f},{tfidf_ci[1]*100:.2f}]")
    print(f"KoELECTRA(참조):   45.75%  (reported_numbers.md §1)")

    return result


if __name__ == "__main__":
    run()
