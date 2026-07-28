# backend/eval/compare_judgment.py
"""
KoELECTRA(v4) vs 검색기반(KoE5+GPT-4o-mini few-shot) 판단 방식 비교 — Phase B.

data/eval/ground_truth_3class.jsonl(Phase A, 1,052건)에 두 판단 경로를 동일하게
돌려 class별 precision/recall/F1을 비교한다.

KoELECTRA(v4)는 evidence_span(평균 40자대) 위주로 학습됐고, 원문 그대로 넣으면
정확도가 크게 떨어지는 게 이미 알려진 특성이다(models/README.md). 실제 프로덕션
(backend/agents/graph.py)에서도 Judgment Agent는 Analysis Agent가 뽑은
evidence_span을 받아서 판단한다 — 그래서 이 스크립트도 원문을 그대로 넣지 않고,
forward_labeling.run_forward()로 evidence_span만 먼저 뽑아(도메인·risk_level은
버림 — 그건 우리가 비교하려는 대상이라 섞으면 안 됨) KoELECTRA에 넣는다. 검색기반
경로는 원문이 evidence_span보다 오히려 살짝 나았다는 기존 실험 결과에 따라 원문
그대로 사용한다. 즉 두 경로 모두 "그 경로가 실제로 쓰는 입력 형태"로 공정하게 비교한다.

evidence_span 추출도 GPT-4o-mini 호출이 필요해서(원문 1,052건), 캐시 파일에
저장해 재실행 시 API를 다시 호출하지 않는다.

실행: python -m backend.eval.compare_judgment [--sample N]
"""

import argparse
import json
import os
import random
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import classification_report, confusion_matrix

from backend.agents.judgment_agent import electra_predict
from backend.eval.retrieval_judgment import retrieval_judge
from backend.fb_check.forward_labeling import run_forward
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

load_dotenv()
logger = load_logger("compare_judgment.log")

GT_PATH        = PROJECT_ROOT / "data/eval/ground_truth_3class.jsonl"
SPAN_CACHE_PATH = PROJECT_ROOT / "data/eval/evidence_span_cache.jsonl"
OUT_PATH        = PROJECT_ROOT / "data/eval/compare_judgment_report.json"

_LABELS = ["High", "Medium", "Low"]


def _load_span_cache() -> dict[str, str]:
    if not SPAN_CACHE_PATH.exists():
        return {}
    return {r["chunk_id"]: r["evidence_span"] for r in load_jsonl(SPAN_CACHE_PATH)}


def extract_evidence_spans(records: list[dict], client: OpenAI) -> dict[str, str]:
    """evidence_span만 필요 — run_forward의 domain/risk_level은 버린다(비교 대상 오염 방지).

    한 건씩 캐시 파일에 즉시 append한다 — 1,052건 중간에 API 오류로 끊겨도
    이미 처리된 건 재호출하지 않고 이어서 진행할 수 있다.
    """
    cache = _load_span_cache()
    missing = [r for r in records if r["chunk_id"] not in cache]
    logger.info(f"  evidence_span 캐시 적중 {len(records) - len(missing)}건 / 신규 추출 {len(missing)}건")

    SPAN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SPAN_CACHE_PATH, "a", encoding="utf-8") as f:
        for i, r in enumerate(missing):
            fwd = run_forward(client, r["text"])
            span = (fwd or {}).get("evidence_span") or ""
            cache[r["chunk_id"]] = span
            f.write(json.dumps({"chunk_id": r["chunk_id"], "evidence_span": span}, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                logger.info(f"    evidence_span 추출 진행: {i + 1}/{len(missing)}")

    return cache


def run_koelectra(records: list[dict], span_cache: dict[str, str]) -> list[str | None]:
    preds = []
    for r in records:
        query = span_cache.get(r["chunk_id"]) or r["text"]
        _, risk = electra_predict(query)
        preds.append(risk)
    return preds


def run_retrieval(records: list[dict], client: OpenAI) -> list[str | None]:
    preds = []
    for i, r in enumerate(records):
        risk_level, _agreement = retrieval_judge(client, r["text"])
        preds.append(risk_level)
        if (i + 1) % 50 == 0:
            logger.info(f"  검색기반 진행: {i + 1}/{len(records)}")
    return preds


def _report(y_true: list[str], y_pred: list[str | None], name: str) -> dict:
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    yt, yp = [t for t, _ in valid], [p for _, p in valid]
    report = classification_report(yt, yp, labels=_LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(yt, yp, labels=_LABELS).tolist()
    logger.info(f"=== {name} === macro F1: {report['macro avg']['f1-score']:.3f} (예측 실패 {len(y_true) - len(valid)}건 제외)")
    return {"classification_report": report, "confusion_matrix": cm, "labels": _LABELS, "dropped_no_prediction": len(y_true) - len(valid)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="일부만 샘플링해 비용/시간 가늠용")
    args = parser.parse_args()

    records = load_jsonl(GT_PATH)
    if args.sample:
        random.seed(42)
        records = random.sample(records, min(args.sample, len(records)))
    logger.info(f"평가 대상: {len(records)}건 | 분포: {dict(Counter(r['risk_level'] for r in records))}")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    y_true = [r["risk_level"] for r in records]

    logger.info("1단계: evidence_span 추출(KoELECTRA용, 캐시 사용)...")
    span_cache = extract_evidence_spans(records, client)

    logger.info("2단계: KoELECTRA(v4) 평가...")
    koelectra_preds = run_koelectra(records, span_cache)

    logger.info("3단계: 검색기반(KoE5+GPT-4o-mini) 평가...")
    retrieval_preds = run_retrieval(records, client)

    report = {
        "n_records": len(records),
        "class_distribution": dict(Counter(y_true)),
        "koelectra_v4": _report(y_true, koelectra_preds, "KoELECTRA(v4)"),
        "retrieval_based": _report(y_true, retrieval_preds, "검색기반(KoE5+GPT-4o-mini)"),
        "note": "High/Medium 라벨은 위반유형 개수 프록시(데이터셋 자체 한계) — data/eval/candidates/rebuild_report.json 참고. 기존 883/534건 재현율 평가와 직접 비교 불가.",
    }
    save_json(report, OUT_PATH)
    logger.info(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
