# backend/eval/hybrid_ensemble.py
"""
KoELECTRA + 검색기반(v2) 앙상블이 둘 중 하나보다 나은지 확인 — 재학습(피처 융합) 전에
값싸게 신호를 먼저 확인하는 단계.

data/eval/ground_truth_3class.jsonl(860건)을 dev(35%)/test(65%)로 나눠, 앙상블 규칙은
dev에서만 설계·확인하고 최종 성능은 test에서만 보고한다 — 같은 데이터로 규칙을
만들고 채점하면 과적합이라 dev/test를 분리했다.

검색기반 v2는 confusion matrix상 Medium precision이 여전히 낮다(0.065~0.11 — Medium
예측 10건 중 9건 가까이 틀림). 1차 시도("검색기반이 Medium이라고 답하면 무조건
KoELECTRA로 대체")는 macro F1은 올랐지만(0.399→0.447) **어쩌다 맞춘 Medium까지
전부 버려서 Medium F1이 오히려 더 나빠지는(0.099→0.048) 부작용**이 있었다 — 맞았는지
틀렸는지 구분 안 하고 무조건 교체했기 때문.

개선: `retrieval_judge()`가 이제 (risk_level, neighbor_agreement)를 반환한다 —
neighbor_agreement는 KoE5 이웃 5개 중 최종 판정과 같은 risk_level을 가진 개수.
"검색기반이 Medium이라 답했는데 이웃들은 Medium을 별로 안 가리킨다"(agreement 낮음
= 헷징 의심, 진단했던 문제)면 그때만 KoELECTRA로 대체하고, "이웃들도 실제로
Medium을 가리킨다"(agreement 높음 = 근거 있는 판단)면 그대로 유지한다.

실행: python -m backend.eval.hybrid_ensemble
"""

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from backend.agents.judgment_agent import predict_articles
from backend.eval.retrieval_judgment import retrieval_judge
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

load_dotenv()
logger = load_logger("hybrid_ensemble.log")

GT_PATH           = PROJECT_ROOT / "data/eval/ground_truth_3class.jsonl"
SPAN_CACHE_PATH    = PROJECT_ROOT / "data/eval/evidence_span_cache.jsonl"
PRED_CACHE_PATH    = PROJECT_ROOT / "data/eval/retrieval_pred_cache.jsonl"
OUT_PATH           = PROJECT_ROOT / "data/eval/hybrid_ensemble_report.json"

_LABELS = ["High", "Medium", "Low"]


def _load_span_cache() -> dict[str, str]:
    if not SPAN_CACHE_PATH.exists():
        return {}
    return {r["chunk_id"]: r["evidence_span"] for r in load_jsonl(SPAN_CACHE_PATH)}


def _load_pred_cache() -> dict[str, dict]:
    """neighbor_agreement 없는 옛 캐시 항목(v1 규칙 실험 때 생성)은 무시 — 재호출해서 채운다."""
    if not PRED_CACHE_PATH.exists():
        return {}
    return {r["chunk_id"]: r for r in load_jsonl(PRED_CACHE_PATH) if "neighbor_agreement" in r}


def get_predictions(records: list[dict], client: OpenAI) -> list[dict]:
    """레코드마다 KoELECTRA·검색기반(v2, agreement 포함) 예측을 모두 구한다."""
    span_cache = _load_span_cache()
    pred_cache = _load_pred_cache()

    results = []
    PRED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PRED_CACHE_PATH, "a", encoding="utf-8") as f:
        for i, r in enumerate(records):
            span = span_cache.get(r["chunk_id"]) or r["text"]
            _, koelectra_pred, _ = predict_articles(span)

            cached = pred_cache.get(r["chunk_id"])
            if cached is not None:
                retrieval_pred, agreement = cached["risk_level"], cached["neighbor_agreement"]
            else:
                retrieval_pred, agreement = retrieval_judge(client, r["text"])
                if retrieval_pred is not None:
                    f.write(json.dumps({"chunk_id": r["chunk_id"], "risk_level": retrieval_pred, "neighbor_agreement": agreement}, ensure_ascii=False) + "\n")
                    f.flush()

            results.append({"chunk_id": r["chunk_id"], "true": r["risk_level"], "koelectra": koelectra_pred, "retrieval": retrieval_pred, "agreement": agreement})
            if (i + 1) % 50 == 0:
                logger.info(f"    예측 진행: {i + 1}/{len(records)}")
    return results


def rule_retrieval_only(p: dict) -> str:
    return p["retrieval"]


def rule_koelectra_only(p: dict) -> str:
    return p["koelectra"]


def rule_medium_fallback(p: dict) -> str:
    """검색기반이 Medium이면 맞았는지 안 따지고 무조건 KoELECTRA로 대체(1차 시도 — Medium 자체가 희생됨)."""
    return p["koelectra"] if p["retrieval"] == "Medium" else p["retrieval"]


def rule_medium_fallback_smart(p: dict) -> str:
    """검색기반이 Medium인데 이웃 근거가 약할 때만(agreement<2) KoELECTRA로 대체 — 근거 있는 Medium은 유지."""
    if p["retrieval"] == "Medium" and (p["agreement"] or 0) < 2:
        return p["koelectra"]
    return p["retrieval"]


_RULES = {
    "retrieval_only": rule_retrieval_only,
    "koelectra_only": rule_koelectra_only,
    "medium_fallback": rule_medium_fallback,
    "medium_fallback_smart": rule_medium_fallback_smart,
}


def _macro_f1(preds: list[dict], rule: Any) -> float:
    y_true = [p["true"] for p in preds if p["retrieval"] is not None]
    y_pred = [rule(p) for p in preds if p["retrieval"] is not None]
    report = classification_report(y_true, y_pred, labels=_LABELS, output_dict=True, zero_division=0)
    return report["macro avg"]["f1-score"], report


def main() -> None:
    logger.info("========== 하이브리드 앙상블 검증(dev/test 분리) ==========")
    records = load_jsonl(GT_PATH)
    dev, test = train_test_split(records, test_size=0.65, random_state=42, stratify=[r["risk_level"] for r in records])
    logger.info(f"dev {len(dev)}건 / test {len(test)}건")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    logger.info("--- dev 예측 생성 ---")
    dev_preds = get_predictions(dev, client)
    logger.info("--- dev에서 규칙별 macro F1 비교 ---")
    dev_scores = {}
    for name, rule in _RULES.items():
        f1, _ = _macro_f1(dev_preds, rule)
        dev_scores[name] = f1
        logger.info(f"  {name}: {f1:.3f}")

    best_rule_name = max(dev_scores, key=dev_scores.get)
    logger.info(f"  dev 기준 최고 규칙: {best_rule_name} ({dev_scores[best_rule_name]:.3f})")

    logger.info("--- test 예측 생성(최종 확인용, dev에서 규칙 확정 후 처음 사용) ---")
    test_preds = get_predictions(test, client)
    test_scores = {}
    test_reports = {}
    for name, rule in _RULES.items():
        f1, report = _macro_f1(test_preds, rule)
        test_scores[name] = f1
        test_reports[name] = report
        logger.info(f"  [test] {name}: macro F1={f1:.3f}")

    result = {
        "dev_size": len(dev), "test_size": len(test),
        "dev_scores": dev_scores, "best_rule_by_dev": best_rule_name,
        "test_scores": test_scores,
        "test_reports": test_reports,
    }
    save_json(result, OUT_PATH)
    logger.info(f"저장: {OUT_PATH}")
    logger.info("========== 완료 ==========")


if __name__ == "__main__":
    main()
