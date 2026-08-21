# backend/eval/confound_analysis.py
"""
라벨-출처 교락 정량화 — 모델 점수 중 "코퍼스 판별"이 차지하는 몫을 잰다.

## 왜 필요한가

`backend/labeling/seed.py`가 *FTC 시정조치=High, 표준계약서=Low*로 라벨을 만든다. 그래서
학습 데이터에서 **출처만 보고 최빈 라벨을 찍으면 77.8%**가 나온다(FB-Check 이전 seed
기준으로는 90.2%). 두 코퍼스는 문체·서식·어휘가 완전히 달라서, 모델이 "위험한 조항인가"가
아니라 **"어느 코퍼스 문서인가"**를 학습해도 점수가 나온다.

이 교락이 v4의 성능 해석을 전부 흐린다:
  - 모델 축 네 개(데이터 확장·앙상블·2-class·백본 8종)가 전부 무효였던 것도,
    전부 같은 지름길을 똑같이 잘 배웠기 때문일 수 있다
  - **Medium만 두 코퍼스에 걸쳐 있어** 지름길이 안 통하고, 그래서 어떤 조합으로도
    정밀도가 12~13%에 고정됐다

## 세 가지 측정

1. **출처 전용 기준선** — 텍스트를 보지 않고 `source`만으로 최빈 라벨을 찍은 정확도.
   모델이 이 값을 못 넘으면 지름길만 배운 것이다.
2. **교락 없는 부분집합 성능** — 출처로 라벨이 결정되지 **않는** 샘플만 채점한다
   (`ftc_case`의 Low·Medium + `standard_contract`의 High·Medium). 여기서의 정확도가
   모델이 실제로 조항 내용을 이해한 정도에 가깝다.
3. **출처 판별 난이도** — TF-IDF + 로지스틱 회귀로 `source`를 맞혀본다. 신경망도 아닌
   단어 빈도만으로 높게 나오면, "코퍼스 판별은 표면 특징만으로 되는 쉬운 문제"라는 증거다.

## 주의: 중복 제거

이 데이터는 문서 간 텍스트 중복이 심하다(학습 55.4%, 평가 25.6%). 중복을 두고 세면
다수 클래스가 과대표집돼 기준선과 정확도가 모두 왜곡된다. 모든 수치를 **중복 제거 후**로
낸다(`--keep-duplicates`로 비교 가능).

평가는 `ground_truth_3class.jsonl` + evidence_span 캐시라 **OpenAI 비용 0**.

실행: .venv/bin/python -m backend.eval.confound_analysis
"""

import argparse
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from backend.eval.ensemble_compare import GT_PATH, SPAN_CACHE_PATH, predict_probs
from backend.model.electra import INV_RISK_MAP, RISK_MAP
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("confound_analysis.log")
OUT_PATH = PROJECT_ROOT / "data/eval/confound_report.json"
_LABELS = ["High", "Medium", "Low"]


def source_only_baseline(rows: list[dict]) -> tuple[float, dict]:
    """출처별 최빈 라벨을 찍었을 때의 정확도 — 텍스트를 전혀 안 보는 기준선."""
    by_src: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_src[r["source"]][r["risk_level"]] += 1
    correct = sum(c.most_common(1)[0][1] for c in by_src.values())
    table = {s: {"최빈라벨": c.most_common(1)[0][0], "분포": dict(c)} for s, c in by_src.items()}
    return correct / len(rows), table


def confound_free_mask(rows: list[dict]) -> np.ndarray:
    """출처가 라벨을 결정하지 '않는' 샘플 — 각 출처의 최빈 라벨이 아닌 것들."""
    by_src: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_src[r["source"]][r["risk_level"]] += 1
    majority = {s: c.most_common(1)[0][0] for s, c in by_src.items()}
    return np.array([r["risk_level"] != majority[r["source"]] for r in rows])


def source_separability(texts: list[str], sources: list[str]) -> float:
    """TF-IDF + 로지스틱 회귀로 출처를 얼마나 쉽게 맞히는가(5-fold 교차검증).

    신경망 없이 단어 빈도만으로 높은 정확도가 나오면, 코퍼스 판별은 표면 특징만으로
    가능한 쉬운 문제라는 뜻이다 — 모델이 risk를 배우는 대신 이걸 배울 유인이 크다.
    """
    if len(set(sources)) < 2:
        return float("nan")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=20000, min_df=2)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    return float(cross_val_score(clf, X, sources, cv=5, scoring="accuracy").mean())


def main(model_dir: str | None = None, keep_duplicates: bool = False) -> None:
    spans = {r["chunk_id"]: r["evidence_span"] for r in load_jsonl(SPAN_CACHE_PATH)}
    rows = [r for r in load_jsonl(GT_PATH) if spans.get(r["chunk_id"])]
    for r in rows:
        r["_text"] = spans[r["chunk_id"]].strip()

    n_raw = len(rows)
    if not keep_duplicates:
        seen: dict[str, dict] = {}
        for r in rows:
            seen.setdefault(r["_text"], r)
        rows = list(seen.values())
        logger.info(f"  텍스트 중복 제거: {n_raw} → {len(rows)}건 "
                    f"(중복을 두면 다수 클래스가 과대표집돼 기준선이 왜곡된다)")

    texts = [r["_text"] for r in rows]
    y = np.array([RISK_MAP[r["risk_level"]] for r in rows])
    sources = [r["source"] for r in rows]

    # --- 1. 기준선들 ---
    label_counts = Counter(r["risk_level"] for r in rows)
    majority_acc = max(label_counts.values()) / len(rows)
    src_acc, src_table = source_only_baseline(rows)
    sep = source_separability(texts, sources)

    # --- 2. 모델 성능: 전체 vs 교락 없는 부분집합 ---
    md = PROJECT_ROOT / (model_dir or "models/v4")
    probs = predict_probs(md, texts)
    pred = probs.argmax(axis=1)
    free = confound_free_mask(rows)
    overall = float((pred == y).mean())
    free_acc = float((pred[free] == y[free]).mean()) if free.any() else float("nan")
    # 교락 없는 부분집합 안에서의 기준선도 같이 내야 비교가 성립한다
    free_counts = Counter(np.array([r["risk_level"] for r in rows])[free].tolist())
    free_majority = max(free_counts.values()) / int(free.sum()) if free.any() else float("nan")

    result = {
        "n_eval": len(rows), "n_before_dedup": n_raw, "model": md.name,
        "label_counts": dict(label_counts),
        "majority_baseline": majority_acc,
        "source_only_baseline": src_acc,
        "source_table": src_table,
        "source_separability_tfidf": sep,
        "model_overall": overall,
        "confound_free": {"n": int(free.sum()), "counts": dict(free_counts),
                          "majority_baseline": free_majority, "model_accuracy": free_acc},
    }
    save_json(result, OUT_PATH)

    logger.info(f"===== 교락 분석 ({md.name}, n={len(rows)}) =====")
    logger.info(f"  라벨 분포: {label_counts.most_common()}")
    logger.info("  --- 텍스트를 안 보는 기준선 ---")
    logger.info(f"    무조건 최빈 라벨          : {majority_acc * 100:.1f}%")
    logger.info(f"    출처별 최빈 라벨          : {src_acc * 100:.1f}%  ← 출처만 알면 이만큼 맞힌다")
    logger.info(f"    TF-IDF로 출처 맞히기      : {sep * 100:.1f}%  ← 단어 빈도만으로 코퍼스 판별 가능한 정도")
    for s, t in src_table.items():
        logger.info(f"      {s:<20} 최빈={t['최빈라벨']:<7} {t['분포']}")
    logger.info("  --- 모델 ---")
    logger.info(f"    전체 정확도               : {overall * 100:.1f}%")
    logger.info(f"    교락 없는 {int(free.sum())}건에서    : {free_acc * 100:.1f}% "
                f"(그 안의 기준선 {free_majority * 100:.1f}%)")
    logger.info(f"      → 순이득 {(free_acc - free_majority) * 100:+.1f}%p "
                f"= 모델이 실제로 조항 내용을 이해한 정도")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=None)
    p.add_argument("--keep-duplicates", action="store_true")
    main(model_dir=p.parse_args().model_dir, keep_duplicates=p.parse_args().keep_duplicates)
