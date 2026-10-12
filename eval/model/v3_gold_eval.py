# eval/model/v3_gold_eval.py
"""article_v3의 gold 조 F1 — 문체 라우팅 포함 배포 상태 (보고용 측정, 2026-10-12, API 0)

기존 결과 파일에는 v3(= CN 시드 7)의 gold 조 F1이 없었다(학습 로그 한 줄뿐, 라우팅 전 값).
서비스가 실제로 쓰는 경로(`judgment_agent.predict_articles` + `route_document`)로 gold clean 255를 채점한다.

    JUDGMENT_DEVICE=cpu .venv/bin/python -m eval.model.v3_gold_eval

⚠ gold는 문서당 조항이 일부뿐이라, 라우터는 **그 문서의 gold 조항들만** 보고 판별한다(실제 서비스는 계약서 전체).
router_eval.py --doc-level 과 같은 방식이며 gold의 83.9%가 회사 약관형으로 분류된다.
"""

import json

import numpy as np

from backend.agents import judgment_agent as ja
from backend.training.retrain import eval_sets, f1_rows
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("v3_gold_eval.log")
OUT = PROJECT_ROOT / "data/eval/model/v3_gold_eval.json"
CONST_F1_PP = 36.1      # gold clean 255 상수 기준선 (reported_numbers.md 최상단 표, 09-05)


def _ci(vals: np.ndarray, groups: list[str]) -> list[float]:
    idx: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    cl = [np.array(v) for v in idx.values()]
    rng = np.random.default_rng(42)
    b = [vals[np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])].mean() for _ in range(5000)]
    return [round(float(x) * 100, 1) for x in np.percentile(b, [2.5, 97.5])]


def main() -> None:
    gold, _ = eval_sets(json.loads((PROJECT_ROOT / "backend/training/negative_holdout_docs.json").read_text(encoding="utf-8")))
    model = ja._get_electra()[0]
    names = list(model.article_names)
    labels = np.array([[a in g["articles"] for a in names] for g in gold])
    groups = [g["doc_id"] for g in gold]

    by_doc: dict[str, list[int]] = {}
    for i, d in enumerate(groups):
        by_doc.setdefault(d, []).append(i)
    style = np.array([None] * len(gold), dtype=object)
    for idx in by_doc.values():
        s = ja.route_document([gold[i]["text"] for i in idx])
        for i in idx:
            style[i] = s

    def run(styles) -> dict:
        pred = np.array([[a in ja.predict_articles(g["text"], doc_style=s) for a in names] for g, s in zip(gold, styles)])
        f1 = f1_rows(pred, labels)
        rec = pred.any(1).astype(float)
        return {"f1_pp": round(float(f1.mean()) * 100, 1), "f1_ci95": _ci(f1, groups),
                "recall_pp": round(float(rec.mean()) * 100, 1), "recall_ci95": _ci(rec, groups)}

    rep = {"rule": "보고용 측정 (사전 등록 아님) — 문서 군집 부트스트랩", "n_gold": len(gold), "n_docs": len(by_doc),
           "routed_company_pct": round(float(np.mean([s == "company" for s in style])) * 100, 1),
           "no_routing_default_thresholds": run([None] * len(gold)),
           "deployed_with_routing": run(list(style)),
           "constant_baseline_f1_pp": CONST_F1_PP}
    for k in ("no_routing_default_thresholds", "deployed_with_routing"):
        r = rep[k]
        r["vs_constant"] = "상수와 구분 안 됨" if r["f1_ci95"][0] <= CONST_F1_PP <= r["f1_ci95"][1] else \
            ("상수 초과" if r["f1_ci95"][0] > CONST_F1_PP else "상수 미만")
    save_json(rep, OUT)
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
