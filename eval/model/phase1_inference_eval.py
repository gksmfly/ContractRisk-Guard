# backend/eval/phase1_inference_eval.py
"""Phase 1 GPU 공유 추론 — E4·E5·E6·E7

인코더 추론을 gold 255 + intact 908에서 한 번 돌려 네 실험을 산출한다.

E4 조 간 혼동 분석 (gold 255)
    예측 조 vs 정답 조 혼동 행렬. 제6조 흡수, 제8조 방향성 오류를 구체적으로 낸다.

E5 오경보 하한 재계산 (intact 908)
    paper_plan §E5: 분모 908, 순수 오경보 = 모델 플래그 ∧ GPT forward 빈 배열.
    "헤딩아님 74" 같은 재현 불가 분류는 쓰지 않는다.

E6 조 F1 49.1%(지목 200건)의 CI (gold 255 중 지목 건)
    페어드 부트스트랩(모델 vs 같은 200건의 상수). CI가 0을 포함하면
    "49.1%는 논문에서 빼고 38.5%만 쓴다" — paper_plan §E6 사전 등록.

E7 article_v2 확률 보정 (gold 255 + holdout 음성)
    조별 sigmoid 확률 10-bin 신뢰도 곡선, ECE.
    목적: UI에 확률을 내도 되는지 근거. 논문에서는 부록.

실행:
    .venv/bin/python -m eval.phase1_inference_eval --gpu 0
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from eval.channel.gpt_channel_fpr import _doc, intact_stratum, load_negative_pool
from backend.labeling.articles import ARTICLES
from backend.model.electra import ArticleMultiLabelElectra
from backend.training.train_article import (
    ArticleDataset,
    load_ftc_gold,
)
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("phase1_inference_eval.log")

OUT_PATH = PROJECT_ROOT / "data/eval/phase1_inference_eval.json"
BOOT_N   = 5_000
SEED     = 42


# ── 추론 ─────────────────────────────────────────────────────────────────────

def _run_inference(records: list[dict], model_dir: Path, gpu: int) -> np.ndarray:
    """records 각 항목의 'text'에 대해 sigmoid 확률 배열 반환. shape=(n, n_articles)"""
    cfg      = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    names    = cfg["article_names"]
    max_len  = int((cfg.get("train_config") or {}).get("max_len", 512))
    device   = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model    = ArticleMultiLabelElectra.load(model_dir).to(device).eval()
    tok      = AutoTokenizer.from_pretrained(str(model_dir))
    ds       = ArticleDataset([{"text": r["text"], "articles": [], "group": "g"} for r in records],
                              tok, max_len, names)
    batches  = []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=32, num_workers=0):
            batches.append(
                torch.sigmoid(model(b["input_ids"].to(device),
                                    b["attention_mask"].to(device),
                                    b["token_type_ids"].to(device))).cpu().numpy()
            )
    return np.concatenate(batches)          # (n, n_articles)


def _load_names(model_dir: Path) -> list[str]:
    return json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))["article_names"]


def _load_thresholds(model_dir: Path) -> np.ndarray:
    return np.load(model_dir / "thresholds.npy")


# ── E4 혼동 행렬 ──────────────────────────────────────────────────────────────

def run_e4(gold: list[dict], probs: np.ndarray, names: list[str],
           thr: np.ndarray) -> dict:
    """예측 조 vs 정답 조 혼동 행렬 (조별 pair count)."""
    EMPTY = "(없음)"
    all_labels = names + [EMPTY]
    # pair_counts[pred][true] = count
    pair_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for i, g in enumerate(gold):
        preds = {names[j] for j in range(len(names)) if probs[i, j] >= thr[j]}
        trues = set(g["articles"])
        for p in (preds or {EMPTY}):
            for t in (trues or {EMPTY}):
                pair_counts[p][t] += 1

    # 제6조 흡수: 예측=제6조, 실제≠제6조인 건수
    absorb_to_6 = sum(pair_counts.get("제6조", {}).get(t, 0)
                      for t in names if t != "제6조")

    # 제8조 방향성 오류: 예측=제8조인데 실제는 다른 조 (또는 그 역)
    art8_fp = sum(v for t, v in pair_counts.get("제8조", {}).items() if t != "제8조" and t != EMPTY)
    art8_fn = sum(pair_counts.get(p, {}).get("제8조", 0) for p in names if p != "제8조")

    # per-article TP/FP/FN
    per_art = {}
    for a in names:
        tp = pair_counts.get(a, {}).get(a, 0)
        fp = sum(v for t, v in pair_counts.get(a, {}).items() if t != a and t != EMPTY)
        fn = sum(pair_counts.get(p, {}).get(a, 0) for p in names if p != a)
        fn += pair_counts.get(EMPTY, {}).get(a, 0)
        per_art[a] = {"tp": tp, "fp": fp, "fn": fn}

    return {
        "experiment":   "E4",
        "n_gold":       len(gold),
        "pair_counts":  {p: {k: int(v) for k, v in dv.items()} for p, dv in pair_counts.items()},
        "제6조_흡수_건": int(absorb_to_6),
        "제8조_fp":      int(art8_fp),
        "제8조_fn":      int(art8_fn),
        "per_article":  {a: {k: int(v) for k, v in d.items()} for a, d in per_art.items()},
        "note":         "pair_counts[pred][true]. EMPTY=(없음)",
    }


# ── E5 순수 오경보 ─────────────────────────────────────────────────────────────

def run_e5(intact: list[dict], probs_neg: np.ndarray, names: list[str],
           thr: np.ndarray) -> dict:
    """순수 오경보 = 모델 플래그 ∧ GPT forward 빈 배열 / 908.

    paper_plan §E5: 분모 908(온전 지층), 분자는 두 조건 교집합.
    "헤딩아님 74" 같은 재현 불가 분류는 쓰지 않는다.
    """
    rng = np.random.default_rng(SEED)
    n = len(intact)
    assert n == len(probs_neg), f"intact {n} ≠ probs {len(probs_neg)}"

    model_flag  = [(probs_neg[i] >= thr).any() for i in range(n)]
    gpt_empty   = [not r.get("forward_articles") for r in intact]
    pure_fp     = [mf and ge for mf, ge in zip(model_flag, gpt_empty)]

    cnt_flag    = sum(model_flag)
    cnt_gpt_e   = sum(gpt_empty)
    cnt_pure    = sum(pure_fp)

    # 부트스트랩 CI (레코드 단위)
    boot = np.array([
        np.mean([pure_fp[rng.integers(0, n)] for _ in range(n)])
        for _ in range(BOOT_N)
    ])
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    logger.info(f"  E5  n={n}  모델플래그={cnt_flag}({cnt_flag/n*100:.1f}%)  "
                f"GPT빈={cnt_gpt_e}({cnt_gpt_e/n*100:.1f}%)  "
                f"순수오경보={cnt_pure}({cnt_pure/n*100:.1f}%) [{lo*100:.1f},{hi*100:.1f}]")

    return {
        "experiment":          "E5",
        "n_intact":            int(n),
        "n_model_flagged":     int(cnt_flag),
        "n_gpt_empty":         int(cnt_gpt_e),
        "n_pure_fp":           int(cnt_pure),
        "pure_fpr":            float(cnt_pure / n),
        "pure_fpr_pct":        float(cnt_pure / n * 100),
        "ci95_pp":             [float(lo * 100), float(hi * 100)],
        "note":                "순수 오경보 = 모델플래그 ∧ GPT forward 빈 배열 / 908. "
                               "기존 2.3%는 n=613, 헤딩아님 74 제외 기준 — 재현 불가라 폐기",
    }


# ── E6 49.1% CI ───────────────────────────────────────────────────────────────

def _f1(pred: set, ref: set) -> float:
    if not pred and not ref:
        return 1.0
    i = len(pred & ref)
    p = i / len(pred) if pred else 0.0
    r = i / len(ref)  if ref  else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def run_e6(gold: list[dict], probs: np.ndarray, names: list[str],
           thr: np.ndarray) -> dict:
    """49.1%(지목 200건)의 페어드 부트스트랩 CI.

    판정 규칙(paper_plan §E6 사전 등록):
    CI가 0을 포함하면 "49.1%는 논문에서 빼고 38.5%만 쓴다".
    """
    rng = np.random.default_rng(SEED)

    # 지목 = 모델이 ≥1개 조 예측한 건
    targeted = []
    for i, g in enumerate(gold):
        preds = {names[j] for j in range(len(names)) if probs[i, j] >= thr[j]}
        trues = set(g["articles"])
        if preds:
            targeted.append({"pred": preds, "true": trues})

    # 상수 최고 (top-k, 지목 건 안에서)
    from collections import Counter
    freq = Counter(a for r in targeted for a in r["true"])
    best_const = None
    best_const_f1 = -1.0
    for k in range(1, len(names) + 1):
        const_set = {a for a, _ in freq.most_common(k)}
        cf = float(np.mean([_f1(const_set, r["true"]) for r in targeted]))
        if cf > best_const_f1:
            best_const_f1, best_const = cf, const_set

    model_f1s = np.array([_f1(r["pred"], r["true"]) for r in targeted])
    const_f1s = np.array([_f1(best_const, r["true"]) for r in targeted])
    diffs     = model_f1s - const_f1s

    # 페어드 부트스트랩
    n = len(targeted)
    boot_d = np.array([
        diffs[rng.integers(0, n, n)].mean() for _ in range(BOOT_N)
    ])
    lo = float(np.percentile(boot_d, 2.5) * 100)
    hi = float(np.percentile(boot_d, 97.5) * 100)
    obs_d = float(diffs.mean() * 100)

    verdict = ("49.1%는 논문에서 사용" if lo > 0
               else "49.1%는 논문에서 빼고 38.5%만 쓴다(CI 0 포함)")

    logger.info(f"  E6  지목 n={n}  모델={model_f1s.mean()*100:.1f}%  "
                f"상수={const_f1s.mean()*100:.1f}%  "
                f"Δ={obs_d:+.1f}%p [{lo:+.1f},{hi:+.1f}]  {verdict}")

    return {
        "experiment":     "E6",
        "n_targeted":     n,
        "n_total_gold":   len(gold),
        "n_no_pred":      len(gold) - n,
        "model_f1":       float(model_f1s.mean()),
        "const_f1":       float(const_f1s.mean()),
        "best_const_set": sorted(best_const),
        "diff_pp":        obs_d,
        "ci95_pp":        [lo, hi],
        "verdict":        verdict,
        "judgment_rule":  "CI 0 포함 → 49.1% 논문에서 제거. paper_plan §E6 사전 등록",
    }


# ── E7 확률 보정 ──────────────────────────────────────────────────────────────

def _ece(probs_1d: np.ndarray, labels_1d: np.ndarray, n_bins: int = 10) -> dict:
    """Expected Calibration Error + bin-level 신뢰도 곡선."""
    bins      = np.linspace(0, 1, n_bins + 1)
    bin_data  = []
    ece       = 0.0
    n_total   = len(probs_1d)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs_1d >= lo) & (probs_1d < hi)
        if not mask.any():
            continue
        avg_conf = float(probs_1d[mask].mean())
        avg_acc  = float(labels_1d[mask].mean())
        n_bin    = int(mask.sum())
        ece     += abs(avg_conf - avg_acc) * n_bin / n_total
        bin_data.append({"lo": float(lo), "hi": float(hi),
                         "conf": avg_conf, "acc": avg_acc, "n": n_bin})
    return {"ece": float(ece), "bins": bin_data, "n_samples": n_total}


def run_e7(gold: list[dict], probs_gold: np.ndarray,
           holdout_neg: list[dict], probs_neg: np.ndarray,
           names: list[str]) -> dict:
    """조별 10-bin 신뢰도 곡선 + ECE.

    양성: gold 255 (각 조 binary label)
    음성: holdout neg (모두 label=0)
    """
    results = {}
    for j, art in enumerate(names):
        # gold: label = 1 if article in gold['articles']
        p_gold = probs_gold[:, j]
        y_gold = np.array([1.0 if art in g["articles"] else 0.0 for g in gold])

        # holdout neg: label = 0
        p_neg  = probs_neg[:, j]
        y_neg  = np.zeros(len(holdout_neg))

        p_all  = np.concatenate([p_gold, p_neg])
        y_all  = np.concatenate([y_gold, y_neg])
        results[art] = _ece(p_all, y_all)

    # macro ECE
    macro_ece = float(np.mean([results[a]["ece"] for a in names]))
    logger.info(f"  E7  macro ECE={macro_ece:.4f}  "
                + "  ".join(f"{a}={results[a]['ece']:.3f}" for a in names))

    return {
        "experiment":       "E7",
        "n_gold":           len(gold),
        "n_holdout_neg":    len(holdout_neg),
        "macro_ece":        macro_ece,
        "per_article":      results,
        "note":             "목적: UI에 확률 노출 여부 판단 근거. 논문 부록",
    }


# ── holdout 음성 로드 ─────────────────────────────────────────────────────────

def _load_holdout_neg(model_dir: Path) -> list[dict]:
    """backend/training/negative_holdout_docs.json에 명시된 12문서의 청크."""
    ho_path = PROJECT_ROOT / "backend/training/negative_holdout_docs.json"
    ho_docs = set(json.loads(ho_path.read_text(encoding="utf-8")))

    clean_path = PROJECT_ROOT / "data/fb_check/clean.jsonl"
    noise_path = PROJECT_ROOT / "data/fb_check/noise.jsonl"

    out = []
    for path in [clean_path, noise_path]:
        for r in load_jsonl(path):
            if r.get("source") != "standard_contract":
                continue
            doc = ":".join(str(r.get("chunk_id", "")).split(":")[:2])
            if doc in ho_docs:
                out.append(r)
    logger.info(f"  holdout 음성: {len(out)}건 ({len(ho_docs)}문서)")
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/article_v2")
    ap.add_argument("--gpu",   type=int, default=0)
    args = ap.parse_args()

    model_dir = PROJECT_ROOT / args.model
    names = _load_names(model_dir)
    thr   = _load_thresholds(model_dir)

    # ── 데이터 로드 ──
    gold        = load_ftc_gold("clean")          # 255건
    intact      = intact_stratum(load_negative_pool())  # 908건 (standard_contract)
    holdout_neg = _load_holdout_neg(model_dir)    # 12문서 holdout

    logger.info(f"  gold={len(gold)}  intact={len(intact)}  holdout_neg={len(holdout_neg)}")

    # ── 추론 (gold + intact + holdout 합쳐서 한 번) ──
    # gold와 intact는 겹치지 않음. holdout은 intact의 부분집합이라 별도 추론
    logger.info("  [1/2] gold + intact 추론…")
    all_recs  = gold + intact
    all_probs = _run_inference(all_recs, model_dir, args.gpu)
    probs_gold   = all_probs[:len(gold)]
    probs_intact = all_probs[len(gold):]

    logger.info("  [2/2] holdout neg 추론…")
    probs_ho = _run_inference(holdout_neg, model_dir, args.gpu)

    # ── 실험 실행 ──
    logger.info("=== E4: 조 간 혼동 분석 ===")
    e4 = run_e4(gold, probs_gold, names, thr)
    logger.info(f"  제6조 흡수={e4['제6조_흡수_건']}건  제8조 FP={e4['제8조_fp']}건 FN={e4['제8조_fn']}건")

    logger.info("=== E5: 오경보 하한 재계산 ===")
    e5 = run_e5(intact, probs_intact, names, thr)

    logger.info("=== E6: 49.1% CI ===")
    e6 = run_e6(gold, probs_gold, names, thr)

    logger.info("=== E7: 확률 보정 ===")
    e7 = run_e7(gold, probs_gold, holdout_neg, probs_ho, names)

    report = {
        "model":      args.model,
        "n_gold":     len(gold),
        "n_intact":   len(intact),
        "n_holdout":  len(holdout_neg),
        "e4": e4, "e5": e5, "e6": e6, "e7": e7,
    }
    save_json(report, OUT_PATH)
    logger.info(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
