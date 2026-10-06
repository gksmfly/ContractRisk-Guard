# eval/model/swap_candidates.py
"""교체 재시도 후보 (a) — RB_s42 임계값 재조정 (사전 등록: eval/paper_plan.md §7 "재시도 사전 등록")

테스트 데이터를 보지 않는다. 쓰는 것은 학습 때와 같은 방식으로 재현한 **dev 분할**뿐이다.

    1. train_article과 같은 순서로 dev 분할을 재현한다 (gold 문서 배제 → 고정 음성 홀드아웃 분리 → 문서 단위 분할).
       RB_s42와 C_s42는 같은 라벨 파일·같은 시드라 분할이 같다. 건수(614)가 metrics.json과 다르면 멈춘다
    2. dev 표준계약서 음성(헤드 조가 하나도 없는 standard_contract 레코드)에서 C_s42의 오경보율 = 목표
    3. RB_s42 임계값 × s 의 dev 음성 오경보가 목표 이하가 되는 가장 작은 s (0.50~3.00, 0.01 간격, 임계값 상한 0.99)
    4. 가중치는 심볼릭 링크로 두고 metrics.json·thresholds.npy만 바꾼 체크포인트를 `models/_swap_RB_s42_cal/`에 만든다

실행:
    .venv/bin/python -m eval.model.swap_candidates --gpu 0
"""

import argparse
import json

import numpy as np

from backend.training.retrain import predict
from backend.training.train import split_by_document
from backend.training.train_article import (
    exclude_gold_documents,
    load_article_records,
    load_ftc_gold,
    split_negative_holdout,
)
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("swap_candidates.log")

RB   = PROJECT_ROOT / "models/_e8_RB_s42"
C    = PROJECT_ROOT / "models/_e8_C_s42"
OUT  = PROJECT_ROOT / "models/_swap_RB_s42_cal"
HOLD = PROJECT_ROOT / "backend/training/negative_holdout_docs.json"


def dev_split(metrics: dict) -> list[dict]:
    names = metrics["article_names"]
    recs = load_article_records(PROJECT_ROOT / "data/fb_check/clean.jsonl", augment=False, label_source="agreed")
    recs = exclude_gold_documents(recs, load_ftc_gold("clean"))
    fixed = json.loads(HOLD.read_text(encoding="utf-8"))
    recs, _ = split_negative_holdout(recs, len(fixed), 42, fixed)
    for r in recs:
        r["has_article"] = bool([a for a in r["articles"] if a in names])
    _, dev = split_by_document(recs, 0.2, metrics["train_config"].get("seed", 42), stratify_key="has_article")
    return dev


def dev_fpr(P: np.ndarray, thr: np.ndarray, neg: np.ndarray) -> float:
    return float((P[neg] >= thr).any(1).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    mr = json.loads((RB / "metrics.json").read_text(encoding="utf-8"))
    mc = json.loads((C / "metrics.json").read_text(encoding="utf-8"))
    assert mr["labels_sha256"] == mc["labels_sha256"] and mr["dev_samples"] == mc["dev_samples"]
    dev = dev_split(mr)
    if len(dev) != mr["dev_samples"]:
        raise SystemExit(f"dev 재현 실패: {len(dev)} ≠ {mr['dev_samples']} — 멈춘다")
    names = mr["article_names"]
    neg = np.array([r.get("source") == "standard_contract" and not r["has_article"] for r in dev])
    pos = np.array([r["has_article"] for r in dev])
    texts = [r["text"] for r in dev]

    Pc, thr_c, _ = predict(C, texts, args.gpu)
    target = dev_fpr(Pc, thr_c, neg)
    Pr, thr_r, _ = predict(RB, texts, args.gpu)
    rows = []
    for s in np.round(np.arange(0.50, 3.001, 0.01), 2):
        t = np.minimum(thr_r * s, 0.99)
        rows.append({"s": float(s), "dev_fpr": dev_fpr(Pr, t, neg),
                     "dev_recall": float((Pr[pos] >= t).any(1).mean())})
    ok = [r for r in rows if r["dev_fpr"] <= target + 1e-12]
    if not ok:
        raise SystemExit("목표 오경보에 도달하는 배율이 없다")
    best = min(ok, key=lambda r: r["s"])
    thr_new = np.minimum(thr_r * best["s"], 0.99)

    OUT.mkdir(exist_ok=True)
    for f in RB.iterdir():
        if f.name in ("metrics.json", "thresholds.npy"):
            continue
        link = OUT / f.name
        if not link.exists():
            link.symlink_to(f.resolve())
    m = dict(mr)
    m["thresholds"] = {a: float(t) for a, t in zip(names, thr_new)}
    m["swap_calibration"] = {"source": str(RB.relative_to(PROJECT_ROOT)), "scale": best["s"],
                             "target": "C_s42 dev 표준계약서 음성 오경보", "target_fpr": target,
                             "rule": "eval/paper_plan.md §7 재시도 사전 등록 (a)"}
    (OUT / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(OUT / "thresholds.npy", thr_new)

    rep = {"dev_n": len(dev), "dev_neg_n": int(neg.sum()), "dev_pos_n": int(pos.sum()),
           "C_s42_dev_fpr": target, "C_s42_dev_recall": float((Pc[pos] >= thr_c).any(1).mean()),
           "RB_s42_dev_fpr_orig": dev_fpr(Pr, thr_r, neg), "RB_s42_dev_recall_orig": float((Pr[pos] >= thr_r).any(1).mean()),
           "chosen": best, "thresholds_new": m["thresholds"]}
    save_json(rep, PROJECT_ROOT / "data/eval/swap_candidate_a.json")
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
