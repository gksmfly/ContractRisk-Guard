# eval/model/cn_calibrate.py
"""CN 임계값 재조정 — dev만 사용 (사전 등록: eval/paper_plan.md §7-10, 2026-10-08)

시드마다 CN의 dev 분할을 학습 때와 같이 재현하고, 같은 시드 C의 dev 위반 재현 이상이 되는 가장 보수적인 배율 s를 고른다.
가중치는 심볼릭 링크로 두고 임계값만 바꾼 체크포인트를 models/_e8_CNC_s{seed}에 만든다.

    .venv/bin/python -m eval.model.cn_calibrate --gpu 0                 # §7-10 dev 기준 → _e8_CNC_*
    .venv/bin/python -m eval.model.cn_calibrate --target aihub          # §7-11 AI Hub 학습 절반 위반 기준 → _e8_CNA_*
"""

import argparse
import json
from pathlib import Path

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

logger = load_logger("cn_calibrate.log")
HOLD = PROJECT_ROOT / "backend/training/negative_holdout_docs.json"
SEEDS = [42, 7, 123]


def dev_split(model_dir: Path) -> list[dict]:
    m = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    names = m["article_names"]
    recs = load_article_records(Path(m["labels_path"]) if m.get("labels_path") else None, augment=False,
                                label_source=m["label_source"])
    recs = exclude_gold_documents(recs, load_ftc_gold("clean"))
    fixed = json.loads(HOLD.read_text(encoding="utf-8"))
    recs, _ = split_negative_holdout(recs, len(fixed), 42, fixed)
    for r in recs:
        r["has_article"] = bool([a for a in r["articles"] if a in names])
    _, dev = split_by_document(recs, 0.2, m["train_config"]["seed"], stratify_key="has_article")
    if len(dev) != m["dev_samples"]:
        raise SystemExit(f"{model_dir.name}: dev 재현 실패 {len(dev)} ≠ {m['dev_samples']}")
    return dev


def dev_recall(model_dir: Path, dev: list[dict], gpu: int, scale: float = 1.0):
    pos = [r for r in dev if r["has_article"]]
    P, thr, _ = predict(model_dir, [r["text"] for r in pos], gpu)
    return P, thr


def write_ckpt(cn_dir: Path, out: Path, sc: float, info: dict) -> np.ndarray:
    m = json.loads((cn_dir / "metrics.json").read_text(encoding="utf-8"))
    names = m["article_names"]
    tn = np.array([m["thresholds"][a] for a in names])
    new_thr = np.minimum(tn * sc, 0.99)
    out.mkdir(exist_ok=True)
    for f in cn_dir.iterdir():
        if f.name not in ("metrics.json", "thresholds.npy") and not (out / f.name).exists():
            (out / f.name).symlink_to(f.resolve())
    m["thresholds"] = {a: float(t) for a, t in zip(names, new_thr)}
    m["cn_calibration"] = info
    (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(out / "thresholds.npy", new_thr)
    return new_thr


def calibrate_aihub() -> None:
    """§7-11: AI Hub 학습 절반 위반 조항(학습 미사용)에서 C의 재현을 목표로. 예측은 aihub_eval 캐시를 쓴다."""
    from eval.model.aihub_eval import CACHE, testset
    items = testset()
    train_files = set(json.loads((PROJECT_ROOT / "data/eval/model/aihub_split.json").read_text(encoding="utf-8"))["train_files"])
    cal = np.array([it["label"] == 1 and it["group"] in train_files for it in items])
    rep = {"rule": "eval/paper_plan.md §7-11", "n_calibration_pos": int(cal.sum()), "seeds": {}}
    for s in SEEDS:
        zc, zn = np.load(CACHE / f"C_s{s}.npz"), np.load(CACHE / f"CN_s{s}.npz")
        target = float((zc["P"][cal] >= zc["thr"]).any(1).mean())
        grid = [(float(sc), float((zn["P"][cal] >= np.minimum(zn["thr"] * sc, 0.99)).any(1).mean()))
                for sc in np.round(np.arange(0.05, 1.501, 0.01), 2)]
        ok = [g for g in grid if g[1] >= target - 1e-12]
        if not ok:
            raise SystemExit(f"seed {s}: 목표 재현 {target:.3f}에 도달하는 배율이 없다")
        sc, rec = max(ok, key=lambda g: g[0])
        write_ckpt(PROJECT_ROOT / f"models/_e8_CN_s{s}", PROJECT_ROOT / f"models/_e8_CNA_s{s}", sc,
                   {"scale": sc, "target_aihub_train_half_recall_from_C": target, "achieved": rec,
                    "rule": "eval/paper_plan.md §7-11"})
        rep["seeds"][s] = {"target_recall": round(target, 4),
                           "CN_recall_orig": round(float((zn["P"][cal] >= zn["thr"]).any(1).mean()), 4),
                           "scale": sc, "achieved": round(rec, 4)}
        logger.info(f"  seed {s}: {rep['seeds'][s]}")
    save_json(rep, PROJECT_ROOT / "data/eval/model/cn_calibrate_aihub.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--target", choices=["dev", "aihub"], default="dev")
    args = ap.parse_args()
    if args.target == "aihub":
        calibrate_aihub()
        return
    rep = {"rule": "eval/paper_plan.md §7-10", "seeds": {}}
    for s in SEEDS:
        c_dir, cn_dir = PROJECT_ROOT / f"models/_e8_C_s{s}", PROJECT_ROOT / f"models/_e8_CN_s{s}"
        dc, dn = dev_split(c_dir), dev_split(cn_dir)
        Pc, tc = dev_recall(c_dir, dc, args.gpu)
        target = float((Pc >= tc).any(1).mean())
        Pn, tn = dev_recall(cn_dir, dn, args.gpu)
        grid = []
        for sc in np.round(np.arange(0.05, 1.501, 0.01), 2):
            t = np.minimum(tn * sc, 0.99)
            grid.append((float(sc), float((Pn >= t).any(1).mean())))
        ok = [g for g in grid if g[1] >= target - 1e-12]
        if not ok:
            raise SystemExit(f"seed {s}: 목표 재현 {target:.3f}에 도달하는 배율이 없다")
        sc, rec = max(ok, key=lambda g: g[0])
        new_thr = np.minimum(tn * sc, 0.99)
        out = PROJECT_ROOT / f"models/_e8_CNC_s{s}"
        out.mkdir(exist_ok=True)
        for f in cn_dir.iterdir():
            if f.name not in ("metrics.json", "thresholds.npy") and not (out / f.name).exists():
                (out / f.name).symlink_to(f.resolve())
        m = json.loads((cn_dir / "metrics.json").read_text(encoding="utf-8"))
        names = m["article_names"]
        m["thresholds"] = {a: float(t) for a, t in zip(names, new_thr)}
        m["cn_calibration"] = {"scale": sc, "target_dev_recall_from_C": target, "achieved_dev_recall": rec,
                               "rule": "eval/paper_plan.md §7-10"}
        (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(out / "thresholds.npy", new_thr)
        rep["seeds"][s] = {"C_dev_n_pos": int(sum(r["has_article"] for r in dc)), "CN_dev_n_pos": int(sum(r["has_article"] for r in dn)),
                           "target_recall": round(target, 4), "CN_dev_recall_orig": round(float((Pn >= tn).any(1).mean()), 4),
                           "scale": sc, "achieved": round(rec, 4)}
        logger.info(f"  seed {s}: {rep['seeds'][s]}")
    save_json(rep, PROJECT_ROOT / "data/eval/model/cn_calibrate.json")


if __name__ == "__main__":
    main()
