# backend/eval/source_signal_probe.py
"""
출처 판별이 '서식' 때문인가 '내용' 때문인가 — 정규화 단계를 올려가며 확인한다.

## 왜 재는가

`confound_analysis.py`에서 TF-IDF(char 2~4gram)가 출처를 79.7%로 맞혔다. 그런데 그게
**서식 차이**(공백·조항 번호 형식·페이지 마커·특수문자)라면 정규화만으로 크게 줄어든다.
반대로 **어휘·문체 차이**라면 정규화해도 안 줄어들고, 데이터를 새로 모으는 것 말고는
방법이 없다.

이 구분에 따라 다음 작업이 완전히 갈리므로 먼저 잰다:
  - 서식 때문 → 전처리 강화로 상당 부분 해결
  - 내용 때문 → 같은 종류 문서에서 High/Low가 모두 나오는 코퍼스를 새로 확보해야 함

## 방법

정규화를 단계적으로 올리며 매번 "TF-IDF + 로지스틱 회귀로 source 맞히기" 5-fold 정확도를
잰다. 마지막에 **무엇이 출처를 알려주는지** 상위 특징을 뽑아 눈으로 확인한다 — 수치만
보면 "왜"를 모른다.

기준선: 다수 출처 비율(그보다 낮으면 판별 불가라는 뜻).

실행: .venv/bin/python -m backend.eval.source_signal_probe
"""

import argparse
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("source_signal_probe.log")
OUT_PATH = PROJECT_ROOT / "data/eval/source_signal_report.json"

_PAGE = re.compile(r"\s*-\s*\d+\s*-\s*")
_ARTICLE = re.compile(r"제\s*\d+\s*조(\s*의\s*\d+)?")
_NUM = re.compile(r"\d+")
_CIRCLED = re.compile(r"[①-⑳㈀-㈎]")
_PUNCT = re.compile(r"[^\w\s가-힣]")


def norm_whitespace(t: str) -> str:
    return " ".join(_PAGE.sub(" ", t).split())


def norm_structure(t: str) -> str:
    """조항 번호·항 번호·숫자를 자리표시자로 — '제5조 ①' 같은 서식 단서를 지운다."""
    t = _ARTICLE.sub(" 제N조 ", t)
    t = _CIRCLED.sub(" ", t)
    return norm_whitespace(_NUM.sub(" 0 ", t))


def norm_punct(t: str) -> str:
    return norm_whitespace(_PUNCT.sub(" ", norm_structure(t)))


_LEVELS = [
    ("원문 그대로", lambda t: t),
    ("공백·페이지마커 정규화", norm_whitespace),
    ("+ 조항번호·숫자 마스킹", norm_structure),
    ("+ 특수문자 제거", norm_punct),
]


def probe(texts: list[str], sources: list[str], analyzer: str) -> float:
    kw = dict(ngram_range=(2, 4), max_features=20000, min_df=2) if analyzer == "char_wb" \
        else dict(ngram_range=(1, 2), max_features=20000, min_df=2)
    vec = TfidfVectorizer(analyzer=analyzer, **kw)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    return float(cross_val_score(clf, X, sources, cv=5, scoring="accuracy").mean())


def top_features(texts: list[str], sources: list[str], k: int = 12) -> dict:
    """무엇이 출처를 알려주는지 — 단어 단위로 상위 가중치 특징을 본다."""
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=20000, min_df=2)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, sources)
    names = np.array(vec.get_feature_names_out())
    coef = clf.coef_[0] if clf.coef_.shape[0] == 1 else clf.coef_
    if coef.ndim == 1:
        pos, neg = clf.classes_[1], clf.classes_[0]
        return {pos: names[np.argsort(coef)[-k:]][::-1].tolist(),
                neg: names[np.argsort(coef)[:k]].tolist()}
    return {c: names[np.argsort(coef[i])[-k:]][::-1].tolist() for i, c in enumerate(clf.classes_)}


def main(dataset: str = "clean") -> None:
    if dataset == "clean":
        rows = load_jsonl(PROJECT_ROOT / "data/fb_check/clean.jsonl")
        pairs = [(r.get("evidence_span") or r.get("text") or "", r.get("source")) for r in rows]
    else:
        spans = {r["chunk_id"]: r["evidence_span"] for r in load_jsonl(PROJECT_ROOT / "data/eval/evidence_span_cache.jsonl")}
        gt = load_jsonl(PROJECT_ROOT / "data/eval/ground_truth_3class.jsonl")
        pairs = [(spans[r["chunk_id"]], r["source"]) for r in gt if spans.get(r["chunk_id"])]

    # 중복 제거 — 두면 다수 출처가 과대표집돼 정확도가 부풀려진다
    seen: dict[str, str] = {}
    for t, s in pairs:
        if t and t.strip():
            seen.setdefault(t.strip(), s)
    texts, sources = list(seen.keys()), list(seen.values())
    counts = Counter(sources)
    majority = max(counts.values()) / len(sources)
    logger.info(f"  {dataset}: {len(texts)}건(중복 제거) / 출처 {dict(counts)}")
    logger.info(f"  다수 출처 비율(기준선): {majority * 100:.1f}%")

    results = {}
    logger.info(f"  {'정규화 단계':<26}{'char n-gram':>13}{'word n-gram':>13}")
    for name, fn in _LEVELS:
        normed = [fn(t) for t in texts]
        c = probe(normed, sources, "char_wb")
        w = probe(normed, sources, "word")
        results[name] = {"char": c, "word": w}
        logger.info(f"  {name:<26}{c * 100:>12.1f}%{w * 100:>12.1f}%")

    feats = top_features([norm_punct(t) for t in texts], sources)
    logger.info("  --- 최대 정규화 후에도 출처를 알려주는 단어 ---")
    for src, ws in feats.items():
        logger.info(f"    {src:<20} {', '.join(ws)}")

    raw = results["원문 그대로"]["char"]
    best_norm = results["+ 특수문자 제거"]["char"]
    drop = raw - best_norm
    verdict = ("서식 영향 큼 — 전처리로 상당 부분 완화 가능" if drop >= 0.10 else
               "서식이 아니라 어휘·내용 차이 — 전처리로는 해결 불가")
    logger.info(f"  → 정규화로 {drop * 100:+.1f}%p 변화 / 기준선 {majority * 100:.1f}%")
    logger.info(f"  → 판정: {verdict}")

    save_json({"dataset": dataset, "n": len(texts), "source_counts": dict(counts),
               "majority_baseline": majority, "levels": results,
               "top_features": feats, "verdict": verdict}, OUT_PATH)
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["clean", "eval"], default="clean")
    main(dataset=p.parse_args().dataset)
