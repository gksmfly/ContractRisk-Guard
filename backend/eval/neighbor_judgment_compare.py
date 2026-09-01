# backend/eval/neighbor_judgment_compare.py
"""
"비슷한 사례 보여주기"가 실제로 판단에 도움이 되는가 — 조항 추천 기능의 유일한 측정 가능한 축.

## 배경

Medium 등급 처리 방향으로 세 안이 열려 있다(A: 폐지 / B: "판단 보류"로 재정의 + 유사 사례
제시 / C: 현행 유지). B안은 "모델이 헷갈리는 조항은 등급을 억지로 매기지 말고, 검증된
유사 사례를 보여주고 사용자가 판단하게 하자"는 것이다.

그 전제가 참인지 먼저 재야 한다 — **이웃 조항들의 라벨이 실제 정답과 상관이 있는가.**
상관이 없으면 유사 사례를 보여주는 건 사용자를 오도할 뿐이다.

이 측정은 조항 추천 4종(대안 조항/유사 제재 사례/내부 연결/Medium 대체) 중 **정답
데이터로 평가할 수 있는 유일한 것**이며, 같은 임베딩 검색을 쓰는 나머지 기능들의
품질에 대한 선행 신호이기도 하다 — 여기서 이웃이 엉망이면 "대안 조항 추천"도 엉망이다.

## 누수 차단 (중요)

평가 대상(`ground_truth_3class.jsonl`)과 이웃 검색 대상(`clean_clauses`)은 출처가 겹친다.
자기 자신이 이웃으로 잡히면 정답을 그대로 베끼는 셈이라 수치가 무의미해진다. 그래서:
  - chunk_id 제외
  - **본문이 동일한 사본도 제외**(chunk_id만 막으면 중복 사본이 그대로 들어온다)
이 프로젝트는 과거에 doc-level leakage로 결론이 뒤집힌 전례가 있다(`models/README.md`).

## 비교 대상

  neighbor_k{K}  — 이웃 K개의 라벨 다수결 (동률이면 가장 가까운 이웃의 라벨)
  koelectra_v4   — 당시 프로덕션 모델 (같은 입력, 같은 평가셋)
                   ※ 지금 프로덕션은 `models/article_v1`(조 multi-label)이고 taxonomy가 다르다

실행: .venv/bin/python -m backend.eval.neighbor_judgment_compare
"""

import argparse
from collections import Counter
from math import comb

import numpy as np

from backend.api.services.retrieval import search_similar_clauses
from backend.eval.ensemble_compare import GT_PATH, SPAN_CACHE_PATH, predict_probs
from backend.model.electra import INV_RISK_MAP, RISK_MAP
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("neighbor_judgment_compare.log")
OUT_PATH = PROJECT_ROOT / "data/eval/neighbor_judgment_report.json"
_K_VALUES = (1, 3, 5, 10)


def _majority(neighbors: list[dict]) -> str | None:
    """이웃 라벨 다수결. 동률이면 가장 유사한 이웃(정렬상 첫 번째)의 라벨을 따른다."""
    if not neighbors:
        return None
    counts = Counter(n["risk_level"] for n in neighbors)
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return neighbors[0]["risk_level"]
    return top[0][0]


def _mcnemar(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> tuple[int, int, float]:
    x = int(((a == y) & (b != y)).sum())
    c = int(((a != y) & (b == y)).sum())
    n = x + c
    if n == 0:
        return x, c, 1.0
    k = min(x, c)
    return x, c, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main(limit: int | None = None) -> None:
    spans = {r["chunk_id"]: r["evidence_span"] for r in load_jsonl(SPAN_CACHE_PATH)}
    rows = [r for r in load_jsonl(GT_PATH) if spans.get(r["chunk_id"])]
    if limit:
        rows = rows[:limit]
    texts = [spans[r["chunk_id"]] for r in rows]
    y = np.array([RISK_MAP[r["risk_level"]] for r in rows])
    logger.info(f"  평가 {len(rows)}건 (High {int((y == 0).sum())} / Medium {int((y == 1).sum())} / Low {int((y == 2).sum())})")

    max_k = max(_K_VALUES)
    neighbor_labels: list[list[str]] = []
    leak_blocked = 0
    for i, r in enumerate(rows):
        # 자기 자신 + 동일 본문 사본을 모두 차단한다
        found = search_similar_clauses(
            texts[i], table="clean_clauses", top_k=max_k,
            exclude_chunk_id=r["chunk_id"],
            exclude_texts={texts[i], r.get("text", "")},
        )
        leak_blocked += sum(1 for n in found if n["similarity"] > 0.999)
        neighbor_labels.append([n["risk_level"] for n in found])
        if (i + 1) % 100 == 0:
            logger.info(f"  이웃 검색 {i + 1}/{len(rows)}")

    results: dict[str, dict] = {}
    preds: dict[str, np.ndarray] = {}
    for k in _K_VALUES:
        pred = []
        for labels in neighbor_labels:
            top = labels[:k]
            lab = _majority([{"risk_level": x} for x in top]) if top else "Low"
            pred.append(RISK_MAP.get(lab, RISK_MAP["Low"]))
        preds[f"neighbor_k{k}"] = np.array(pred)

    probs = predict_probs(PROJECT_ROOT / "models/v4", texts)
    preds["koelectra_v4"] = probs.argmax(axis=1)

    for name, p in preds.items():
        per_class = {
            INV_RISK_MAP[i]: {
                "predicted": int((p == i).sum()),
                "correct": int(((p == i) & (y == i)).sum()),
                "recall": float(((p == i) & (y == i)).sum() / max((y == i).sum(), 1)),
            }
            for i in sorted(INV_RISK_MAP)
        }
        results[name] = {"accuracy": float((p == y).mean()), "per_class": per_class}

    paired = {
        name: dict(zip(("b_this_only", "c_v4_only", "p_value"),
                       _mcnemar(p, preds["koelectra_v4"], y)))
        for name, p in preds.items() if name != "koelectra_v4"
    }

    save_json({"n_eval": len(rows), "identical_neighbors_seen": leak_blocked,
               "results": results, "mcnemar_vs_v4": paired}, OUT_PATH)

    logger.info(f"===== 이웃 기반 판단 vs KoELECTRA (n={len(rows)}) =====")
    logger.info(f"  유사도 0.999 초과 이웃(누수 의심): {leak_blocked}건")
    logger.info(f"  {'방식':<16}{'정확도':>9}{'Medium 예측':>13}{'Medium 정답':>12}{'vs v4 p':>10}")
    for name in list(preds):
        r = results[name]
        m = r["per_class"]["Medium"]
        p = "" if name == "koelectra_v4" else f"{paired[name]['p_value']:>10.4g}"
        logger.info(f"  {name:<16}{r['accuracy'] * 100:>8.1f}%{m['predicted']:>13}{m['correct']:>12}{p}")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    main(limit=p.parse_args().limit)
