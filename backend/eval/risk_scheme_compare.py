# backend/eval/risk_scheme_compare.py
"""
3-class vs 2-class 라벨 체계 비교 — Medium을 빼면 High/Low 판별이 실제로 나아지는가.

## 왜 이 실험인가

`models/README.md`의 v4 성능(정확도 43.4%, Medium F1 0.10~0.14)을 여섯 세대(v5~v9)의
데이터 확장으로도 못 넘었고, 2026-08-16 측정에서 31가지 앙상블 조합으로도 못 넘었다
(전부 p>0.19). 같은 측정에서 **Medium 예측 정밀도가 모델 조합과 무관하게 12~13%에
고정**되는 것이 확인됐다 — 학습 실패가 아니라 라벨이 학습 가능한 신호를 담고 있지
않다는 뜻이다(라벨 소스별 Medium 개수도 60~159건으로 2.6배 편차).

그래서 남은 축은 "모델을 더 잘 만든다"가 아니라 **"문제 정의를 고친다"**다.

## 공정한 비교를 위한 통제

1. **기준선을 새로 학습한다.** 저장된 `models/v4`와 비교하면 안 된다 — v4는 학습 데이터가
   493건이던 시절 것이고(현재 `clean.jsonl`은 694건), 하이퍼파라미터도 다르다
   (v4: epochs 5/batch 16, `_seedexp`: epochs 10/batch 32). 라벨 체계 효과와 데이터·설정
   차이가 섞인다. → `models/_schemeexp/`에 3class·2class를 **동일 설정·동일 데이터·동일
   5시드**로 새로 학습해 비교한다.

2. **평가 대상을 맞춘다.** 정답이 High 또는 Low인 샘플만 채점한다 — 2class 모델은 Medium을
   애초에 못 내므로, Medium 정답 샘플을 포함하면 자동으로 지는 불공정한 비교가 된다.

3. **선택지 수를 맞춘다(핵심).** 3class 모델은 선택지가 3개라 그 자체로 불리하다. 그래서
   두 가지로 채점한다:
     - `3class_raw`      — Medium이라 답하면 오답 (실제 서비스에서 일어나는 일)
     - `3class_binarized` — High/Low 로짓만 놓고 argmax (**공정 통제**: 두 모델이 같은
       2지선다를 푼다. 여기서도 2class가 이기면 그건 순수하게 "학습 라벨에서 Medium을
       뺀 효과"다)

평가셋은 `ground_truth_3class.jsonl` 중 evidence_span이 있는 건(=프로덕션 Judgment Agent가
실제로 받는 입력 형태), span 캐시 재사용이라 **OpenAI 비용 0**.

실행: .venv/bin/python -m backend.eval.risk_scheme_compare
"""

import argparse
from math import comb

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from backend.eval.ensemble_compare import GT_PATH, SPAN_CACHE_PATH, predict_probs
from backend.model.electra import get_risk_scheme
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("risk_scheme_compare.log")

SCHEMEEXP_DIR = PROJECT_ROOT / "models/_schemeexp"
OUT_PATH = PROJECT_ROOT / "data/eval/risk_scheme_report.json"
_SEEDS = (1, 7, 42, 100, 123)
_BINARY_LABELS = ["High", "Low"]


def _mcnemar(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> tuple[int, int, float]:
    x = int(((a == y) & (b != y)).sum())
    c = int(((a != y) & (b == y)).sum())
    n = x + c
    if n == 0:
        return x, c, 1.0
    k = min(x, c)
    return x, c, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def _score(pred: np.ndarray, y: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "high_f1": float(f1_score(y, pred, average=None, labels=[0, 1], zero_division=0)[0]),
        "low_f1": float(f1_score(y, pred, average=None, labels=[0, 1], zero_division=0)[1]),
    }


def main(limit: int | None = None) -> dict:
    map3, _, _ = get_risk_scheme("3class")
    spans = {r["chunk_id"]: r["evidence_span"] for r in load_jsonl(SPAN_CACHE_PATH)}

    # High/Low 정답만 채점 대상 (통제 2)
    rows = [r for r in load_jsonl(GT_PATH)
            if spans.get(r["chunk_id"]) and r["risk_level"] in _BINARY_LABELS]
    if limit:
        rows = rows[:limit]
    texts = [spans[r["chunk_id"]] for r in rows]
    # 2class 인덱스 기준으로 정답을 만든다: High=0, Low=1
    y = np.array([0 if r["risk_level"] == "High" else 1 for r in rows])
    logger.info(f"  채점 대상 {len(rows)}건 (High {int((y == 0).sum())} / Low {int((y == 1).sum())}), OpenAI 호출 0")

    preds: dict[str, np.ndarray] = {}
    for seed in _SEEDS:
        d2 = SCHEMEEXP_DIR / f"2class_seed{seed}"
        if d2.exists():
            preds[f"2class_seed{seed}"] = predict_probs(d2, texts, num_risk_labels=2).argmax(axis=1)

        d3 = SCHEMEEXP_DIR / f"3class_seed{seed}"
        if d3.exists():
            p3 = predict_probs(d3, texts, num_risk_labels=3)
            # raw: Medium이면 오답이 되도록 정답에 없는 인덱스(2)로 둔다
            raw = np.where(p3.argmax(axis=1) == map3["Medium"], 2, 0)
            raw = np.where(p3.argmax(axis=1) == map3["Low"], 1, raw)
            preds[f"3class_raw_seed{seed}"] = raw
            # binarized: High/Low 확률만 비교 (통제 3)
            hl = p3[:, [map3["High"], map3["Low"]]]
            preds[f"3class_bin_seed{seed}"] = hl.argmax(axis=1)

    results = {k: _score(v, y) for k, v in preds.items()}

    def agg(prefix: str) -> dict | None:
        keys = [k for k in results if k.startswith(prefix)]
        if not keys:
            return None
        accs = [results[k]["accuracy"] for k in keys]
        f1s = [results[k]["macro_f1"] for k in keys]
        return {"n_seeds": len(keys),
                "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
                "accuracy_range": [float(min(accs)), float(max(accs))],
                "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s))}

    summary = {name: agg(name) for name in ("2class_seed", "3class_raw_seed", "3class_bin_seed")}

    # 시드별 짝지은 검정 — 같은 시드끼리 2class vs 3class(binarized)
    paired = {}
    for seed in _SEEDS:
        a, b = f"2class_seed{seed}", f"3class_bin_seed{seed}"
        if a in preds and b in preds:
            w, l, p = _mcnemar(preds[a], preds[b], y)
            paired[f"seed{seed}"] = {"b_2class_only": w, "c_3class_only": l, "p_value": p}

    save_json({"n_eval": len(rows), "summary": summary, "per_model": results,
               "mcnemar_2class_vs_3class_binarized": paired,
               "note": "동일 데이터·동일 하이퍼파라미터·동일 5시드로 새로 학습한 체크포인트끼리 비교. "
                       "저장된 models/v4는 학습 데이터·설정이 달라 비교 대상에서 제외."},
              OUT_PATH)

    logger.info(f"===== 라벨 체계 비교 (n={len(rows)}, High/Low만 채점) =====")
    logger.info(f"  {'설정':<20}{'정확도':>18}{'macro-F1':>12}")
    for name, label in (("2class_seed", "2class (Medium 제외)"),
                        ("3class_bin_seed", "3class (2지선다 통제)"),
                        ("3class_raw_seed", "3class (원본, Medium=오답)")):
        s = summary[name]
        if s:
            logger.info(f"  {label:<20}{s['accuracy_mean'] * 100:>10.1f}% ±{s['accuracy_std'] * 100:.1f}"
                        f"{s['macro_f1_mean']:>12.3f}")
    logger.info("  시드별 짝지은 McNemar (2class vs 3class-통제):")
    for k, v in paired.items():
        logger.info(f"    {k}: 2class승 {v['b_2class_only']} / 3class승 {v['c_3class_only']} / p={v['p_value']:.4g}")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    main(limit=p.parse_args().limit)
