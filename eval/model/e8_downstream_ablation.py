# backend/eval/e8_downstream_ablation.py
"""E8 — 하류 학습 ablation (사전 등록, 미판정 예상)

## 설계 (paper_plan.md §E8)

팔: A (필터 없음, forward, CLEAN∪NOISE 2305건)
    B (무작위, forward, C와 같은 수 1473건)
    C (FB-Check, agreed, CLEAN 1473건) ← 현행
    (선택) D (CleanLab, forward, 1287건)
각 팔 3시드(42 · 7 · 123). 학습 설정은 article_v2와 동일.

판정: C−A · C−B 각각. CI가 0을 포함하면 "미판정, MDE ±4%p"로 보고.

## 실행

    # 데이터 파일 준비만:
    python -m eval.e8_downstream_ablation --prep-only

    # 학습 + 평가 (각 팔·시드 순차 실행, ~2-3일 소요):
    python -m eval.e8_downstream_ablation --arms A B C --gpu 0

    # 평가만 (모델 디렉터리 이미 있을 때):
    python -m eval.e8_downstream_ablation --eval-only --gpu 0

    # 결과 집계:
    python -m eval.e8_downstream_ablation --analyze
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json, save_jsonl

logger = load_logger("e8_downstream_ablation.log")

# ── 설정 ────────────────────────────────────────────────────────────────────

SEEDS = [42, 7, 123]
N_C   = 0     # 실행 시 clean.jsonl 크기로 동적 결정 (prepare_arms 참고)

CLEAN_PATH  = PROJECT_ROOT / "data/fb_check/clean.jsonl"
NOISE_PATH  = PROJECT_ROOT / "data/fb_check/noise.jsonl"
BASE_DIR    = PROJECT_ROOT / "models/article_v2"
HOLDOUT_PATH = PROJECT_ROOT / "backend/training/negative_holdout_docs.json"
ARM_DATA_DIR = PROJECT_ROOT / "data/eval/e8_arms"
MODEL_OUT    = PROJECT_ROOT / "models"
OUT_PATH     = PROJECT_ROOT / "data/eval/e8_downstream_ablation.json"

# arm B 무작위 선택 시드 (학습 시드와 분리)
B_SUBSET_SEED = 42


# ── 데이터 준비 ──────────────────────────────────────────────────────────────

def _doc_of(chunk_id: str) -> str:
    parts = str(chunk_id).split(":")
    return ":".join(parts[:2]) if len(parts) >= 3 else str(chunk_id)


def prepare_arms() -> dict[str, Path]:
    """각 팔의 라벨 JSONL 파일 생성. 이미 있으면 덮어쓰지 않는다."""
    ARM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    holdout_docs = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))

    clean  = load_jsonl(CLEAN_PATH)
    noise  = load_jsonl(NOISE_PATH)
    all_records = clean + noise

    logger.info(f"  원본: CLEAN {len(clean)} + NOISE {len(noise)} = {len(all_records)}")

    paths: dict[str, Path] = {}

    # ── Arm C: CLEAN 현행 (agreed labels) ────────────────────────────────────
    # 기존 clean.jsonl을 그대로 씀 — 재생성 불필요
    paths["C"] = CLEAN_PATH
    logger.info(f"  Arm C: {CLEAN_PATH} (기존, agreed, {len(clean)}건)")

    # ── Arm A: CLEAN∪NOISE (forward labels) ──────────────────────────────────
    arm_a_path = ARM_DATA_DIR / "arm_A_forward_all.jsonl"
    if not arm_a_path.exists():
        # forward_articles가 있는 레코드만 (CLEAN은 항상, NOISE도 있어야 함)
        a_recs = []
        for r in all_records:
            arts = r.get("forward_articles")
            if arts is None:
                continue
            # agreed_articles를 forward_articles 값으로 덮어써서 train_article이 forward를 쓰게 함
            # label_source=forward를 쓰면 r["forward_articles"]를 직접 읽으므로 원본을 그대로 저장
            a_recs.append(r)
        save_jsonl(a_recs, arm_a_path)
        logger.info(f"  Arm A: {arm_a_path} 저장 (forward, {len(a_recs)}건)")
    else:
        logger.info(f"  Arm A: {arm_a_path} 이미 있음")
    paths["A"] = arm_a_path

    # ── Arm B: 무작위 len(clean)건 (forward labels) ──────────────────────────
    n_c = len(clean)  # C와 같은 크기
    arm_b_path = ARM_DATA_DIR / "arm_B_forward_random.jsonl"
    if not arm_b_path.exists():
        a_recs = load_jsonl(arm_a_path)
        holdout_set = set(holdout_docs)

        # holdout docs는 반드시 포함 (chunk_id "source:doc" 단위)
        must_include = [r for r in a_recs
                       if _doc_of(r.get("chunk_id","")) in holdout_set]
        rest         = [r for r in a_recs
                       if _doc_of(r.get("chunk_id","")) not in holdout_set]

        rng = np.random.default_rng(B_SUBSET_SEED)
        n_random = n_c - len(must_include)
        if n_random < 0:
            logger.warning(f"  holdout({len(must_include)}) > n_c({n_c}) — 그냥 holdout만")
            b_recs = must_include
        else:
            selected = rng.choice(len(rest), size=min(n_random, len(rest)), replace=False)
            b_recs = must_include + [rest[i] for i in sorted(selected)]

        save_jsonl(b_recs, arm_b_path)
        logger.info(f"  Arm B: {arm_b_path} 저장 (forward, random {len(b_recs)}건, "
                    f"holdout docs {len(set(_doc_of(r.get('chunk_id','')) for r in must_include))}개 보장)")
    else:
        logger.info(f"  Arm B: {arm_b_path} 이미 있음")
    paths["B"] = arm_b_path

    return paths


# ── 학습 ─────────────────────────────────────────────────────────────────────

def _arm_label_source(arm: str) -> str:
    return "agreed" if arm == "C" else "forward"


def _model_dir(arm: str, seed: int) -> Path:
    return MODEL_OUT / f"_e8_{arm}_s{seed}"


def train_arm(arm: str, seed: int, labels_path: Path, gpu: int) -> Path:
    """하나의 팔·시드 학습. 이미 완료됐으면 건너뜀."""
    out_dir = _model_dir(arm, seed)
    done_flag = out_dir / "metrics.json"
    if done_flag.exists():
        logger.info(f"  Arm {arm} seed {seed}: {out_dir} 이미 있음 — 건너뜀")
        return out_dir

    metrics_v2 = json.loads((BASE_DIR / "metrics.json").read_text(encoding="utf-8"))
    tc = metrics_v2["train_config"]

    argv = [
        sys.executable, "-m", "backend.training.train_article",
        "--labels",   str(labels_path),
        "--model-dir", str(out_dir),
        "--epochs",    str(tc["epochs"]),
        "--batch-size", str(tc["batch_size"]),
        "--lr",         str(tc["lr"]),
        "--max-len",    str(tc["max_len"]),
        "--min-support", str(tc["min_support"]),
        "--seed",        str(seed),
        "--label-source", _arm_label_source(arm),
        "--negative-holdout", str(metrics_v2["negative_holdout_docs"]),
        "--negative-holdout-docs", str(HOLDOUT_PATH),
        "--gpu", str(gpu),
    ]
    if metrics_v2.get("negative_ratio") is not None:
        argv += ["--negative-ratio", str(metrics_v2["negative_ratio"])]

    logger.info(f"  Arm {arm} seed {seed}: 학습 시작 → {out_dir}")
    logger.info(f"    명령: {' '.join(argv[2:])}")
    result = subprocess.run(argv, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Arm {arm} seed {seed} 학습 실패 (returncode={result.returncode})")

    logger.info(f"  Arm {arm} seed {seed}: 완료")
    return out_dir


# ── 평가 ─────────────────────────────────────────────────────────────────────

def eval_model(model_dir: Path, gpu: int) -> dict:
    """단일 모델 평가 — gold 재현·홀드아웃 FPR·조 F1."""
    from backend.training.retrain import eval_sets, f1_rows, predict

    holdout_docs = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
    gold, neg    = eval_sets(holdout_docs)
    neg_groups   = [":".join(str(r["chunk_id"]).split(":")[:2]) for r in neg]

    P, thr, names = predict(model_dir, [g["text"] for g in gold], gpu)
    Pn, *_        = predict(model_dir, [r["text"] for r in neg], gpu)
    pred   = P >= thr
    labels = np.array([[a in g.get("articles", []) for a in names] for g in gold])

    recall = pred.any(1).astype(float)
    fpr    = (Pn >= thr).any(1).astype(float)
    f1     = f1_rows(pred, labels)

    return {
        "dir":          str(model_dir),
        "n_gold":       int(len(gold)),
        "n_holdout":    int(len(neg)),
        "gold_recall":  float(recall.mean()),
        "holdout_fpr":  float(fpr.mean()),
        "gold_f1":      float(f1.mean()),
        "_arrays": {"recall": recall, "fpr": fpr, "f1": f1,
                    "neg_groups": neg_groups},
    }


# ── 집계 및 판정 ─────────────────────────────────────────────────────────────

def _paired_ci(diff: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float, float]:
    """페어드 부트스트랩 CI (mean, lo, hi). 단위: %p."""
    rng = np.random.default_rng(seed)
    point = float(diff.mean() * 100)
    if len(diff) < 2:
        return point, float("nan"), float("nan")
    boots = np.array([
        rng.choice(diff, size=len(diff), replace=True).mean() * 100
        for _ in range(n_boot)
    ])
    boots = np.sort(boots)
    return point, float(boots[int(n_boot * 0.025)]), float(boots[int(n_boot * 0.975)])


def _arm_summary(arm: str, seed_evals: list[dict]) -> dict:
    """3시드 평균 + CI."""
    recalls = np.array([e["gold_recall"] for e in seed_evals])
    fprs    = np.array([e["holdout_fpr"] for e in seed_evals])
    f1s     = np.array([e["gold_f1"]     for e in seed_evals])
    return {
        "arm":           arm,
        "seeds":         SEEDS[:len(seed_evals)],
        "gold_recall_pp": round(float(recalls.mean() * 100), 2),
        "holdout_fpr_pp": round(float(fprs.mean() * 100), 2),
        "gold_f1_pp":     round(float(f1s.mean() * 100), 2),
        "recall_std_pp":  round(float(recalls.std() * 100), 2),
        "fpr_std_pp":     round(float(fprs.std() * 100), 2),
    }


def _compare_arms(arm_c_evals: list[dict], arm_x_evals: list[dict],
                  label: str) -> dict:
    """C와 X를 개별 레코드 단위 페어드 비교."""
    from backend.training.retrain import paired_ci as _paired_ci_retrain
    c_arrs = [e["_arrays"] for e in arm_c_evals]
    x_arrs = [e["_arrays"] for e in arm_x_evals]

    # 각 시드의 diff를 concat해 분포로
    recall_diffs = np.concatenate([c["recall"] - x["recall"]
                                   for c, x in zip(c_arrs, x_arrs)])
    fpr_diffs    = np.concatenate([c["fpr"] - x["fpr"]
                                   for c, x in zip(c_arrs, x_arrs)])
    f1_diffs     = np.concatenate([c["f1"] - x["f1"]
                                   for c, x in zip(c_arrs, x_arrs)])

    def _ci(diff: np.ndarray) -> dict:
        mean_pp, lo, hi = _paired_ci(diff)
        zero_included = lo <= 0 <= hi
        verdict = ("미판정, MDE ±4%p" if zero_included
                   else ("C > " + label if mean_pp > 0 else label + " > C"))
        return {"mean_pp": round(mean_pp, 2), "ci95": [round(lo, 2), round(hi, 2)],
                "verdict": verdict}

    return {
        "comparison": f"C vs {label}",
        "n_per_seed": int(len(recall_diffs) // max(len(c_arrs), 1)),
        "recall": _ci(recall_diffs),
        "fpr":    _ci(fpr_diffs),
        "f1":     _ci(f1_diffs),
        "mde_pp": 4.3,
        "note":   "article_v1→v2 페어드 CI 반폭 ±4.3%p 기준 (paper_plan §E8)",
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms",      nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"])
    parser.add_argument("--seeds",     nargs="+", type=int, default=SEEDS)
    parser.add_argument("--gpu",       type=int,  default=0)
    parser.add_argument("--prep-only", action="store_true", help="데이터 파일만 만들고 멈춤")
    parser.add_argument("--eval-only", action="store_true", help="이미 학습된 모델만 평가")
    parser.add_argument("--analyze",   action="store_true", help="결과 집계만")
    args = parser.parse_args()

    if args.analyze:
        _analyze(args.gpu, args.arms)
        return

    logger.info("=== E8 하류 학습 ablation ===")
    arm_paths = prepare_arms()
    logger.info("  데이터 파일 준비 완료")

    if args.prep_only:
        for arm, p in arm_paths.items():
            records = load_jsonl(p) if p.exists() else []
            logger.info(f"  {arm}: {p} ({len(records)}건)")
        return

    for arm in args.arms:
        labels_path = arm_paths[arm]
        for seed in args.seeds:
            if not args.eval_only:
                train_arm(arm, seed, labels_path, args.gpu)
            out_dir = _model_dir(arm, seed)
            if not (out_dir / "metrics.json").exists():
                logger.warning(f"  {out_dir} 없음 — 건너뜀")
                continue
            logger.info(f"  평가: Arm {arm} seed {seed}")
            ev = eval_model(out_dir, args.gpu)
            logger.info(
                f"    recall={ev['gold_recall']*100:.1f}%  "
                f"fpr={ev['holdout_fpr']*100:.1f}%  "
                f"f1={ev['gold_f1']*100:.1f}%"
            )

    _analyze(args.gpu, args.arms)


def _analyze(gpu: int, arms: list[str]) -> None:
    logger.info("=== E8 분석 ===")
    arm_evals: dict[str, list[dict]] = {}

    for arm in ["A", "B", "C"]:
        seed_evals = []
        for seed in SEEDS:
            d = _model_dir(arm, seed)
            if not (d / "metrics.json").exists():
                logger.warning(f"  {d} 없음 — 건너뜀")
                continue
            ev = eval_model(d, gpu)
            seed_evals.append(ev)
        if seed_evals:
            arm_evals[arm] = seed_evals

    if not arm_evals:
        logger.error("  평가할 모델이 없다 — 먼저 학습을 실행하세요")
        return

    summaries = {arm: _arm_summary(arm, evs) for arm, evs in arm_evals.items()}
    comparisons: dict[str, Any] = {}
    if "C" in arm_evals and "A" in arm_evals:
        comparisons["C_vs_A"] = _compare_arms(arm_evals["C"], arm_evals["A"], "A")
    if "C" in arm_evals and "B" in arm_evals:
        comparisons["C_vs_B"] = _compare_arms(arm_evals["C"], arm_evals["B"], "B")

    report = {
        "experiment":  "E8",
        "description": "하류 학습 ablation (사전 등록, 미판정 예상)",
        "arms":        {"A": "필터 없음(forward, 2305건)", "B": "무작위(forward, 1473건)",
                        "C": "FB-Check(agreed, 1473건)"},
        "seeds":       SEEDS,
        "mde_pp":      4.3,
        "judgment_rule": "CI 하한 > 0 → C > X, CI 상한 < 0 → X > C, 포함 → 미판정 MDE±4%p",
        "summaries":   summaries,
        "comparisons": comparisons,
    }

    save_json(report, OUT_PATH)
    logger.info(f"저장: {OUT_PATH}")

    for arm, s in summaries.items():
        logger.info(f"  Arm {arm}: recall={s['gold_recall_pp']}% fpr={s['holdout_fpr_pp']}% "
                    f"f1={s['gold_f1_pp']}% (±{s['recall_std_pp']}%p std)")
    for name, comp in comparisons.items():
        r = comp["recall"]
        f = comp["fpr"]
        logger.info(f"  {comp['comparison']}: recall Δ={r['mean_pp']:+.1f}%p {r['ci95']} → {r['verdict']} | "
                    f"fpr Δ={f['mean_pp']:+.1f}%p → {f['verdict']}")


if __name__ == "__main__":
    main()
