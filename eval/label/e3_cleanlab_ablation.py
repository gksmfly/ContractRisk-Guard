# backend/eval/e3_cleanlab_ablation.py
"""E3 — 기존 필터 방법 비교: CleanLab · 모델 신뢰도 임계값

## 흐름

1. 모집단 로드: source=ftc_case 레코드 (CLEAN 1,473 + NOISE 832 = 2,305)
   라벨은 forward (E1과 동일 조건)

2. 문서 단위 5-fold 인코더 학습 → out-of-fold(OOF) 확률

3. 두 필터 방법으로 832건 제거, 1,473건 확보:
   A CleanLab: `find_label_issues`로 이슈 순위 → 상위 832 제거
   B 신뢰도 임계값: 라벨된 조의 OOF 확률 평균이 낮은 순 → 832 제거

4. 각 방법이 남긴 1,473건의 FTC 대비 F1을:
   - E1의 B 분포(무작위 제거 10,000번)와 비교
   - FB-Check(E1의 C)와 페어드 비교(레코드 단위 부트스트랩)

## 주의: 순환

CleanLab은 모델이 라벨을 고르는 방법이라 OOF 모델을 같은 라벨로 학습했다는 순환이 있다.
표에 명시한다.

실행 (GPU 필요, 5회 학습 × ~30분 = 2~3시간 예상):
    .venv/bin/python -m eval.e3_cleanlab_ablation --gpu 0
    .venv/bin/python -m eval.e3_cleanlab_ablation --gpu 0 --skip-train  # OOF 저장본 재사용
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from eval.channel.fbcheck_external_eval import _f1, _labels, _ref_by_doc, _doc_of
from backend.labeling.articles import ARTICLES, normalize
from backend.model.electra import ArticleMultiLabelElectra
from backend.training.train import BASE_MODEL, split_by_document
from backend.training.train_article import (
    ArticleDataset,
    compute_article_pos_weight,
    infer,
    load_article_records,
    load_ftc_gold,
    per_article_metrics,
    select_article_labels,
    tune_thresholds,
)
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("e3_cleanlab_ablation.log")

CLEAN_PATH = PROJECT_ROOT / "data/fb_check/clean.jsonl"
NOISE_PATH = PROJECT_ROOT / "data/fb_check/noise.jsonl"
OOF_PATH   = PROJECT_ROOT / "data/eval/e3_oof_probs.npz"
OUT_PATH   = PROJECT_ROOT / "data/eval/e3_cleanlab_ablation.json"

N_FOLDS   = 5
EPOCHS    = 10
BATCH     = 16
LR        = 3e-5
MAX_LEN   = 512
MIN_SUP   = 5
SEED      = 42
BOOT_N    = 10_000
BOOT_CI   = 5_000


# ── 모집단 로드 (E1과 동일) ────────────────────────────────────────────────────

def _load_population(ref: dict) -> tuple[list[dict], list[dict]]:
    """FTC + ref 있는 레코드만. label=forward."""
    def _make(r):
        doc = _doc_of(r.get("chunk_id", ""))
        if r.get("source") != "ftc_case" or doc not in ref:
            return None
        return {
            "doc":          doc,
            "text":         (r.get("text") or "").strip(),
            "articles":     sorted(_labels(r.get("forward_articles"))),
            "group":        doc,               # 문서 단위 fold split용
            "source":       r.get("source", ""),
            "noise_reason": str(r.get("noise_reason") or ""),
        }
    clean = [x for r in load_jsonl(CLEAN_PATH) if (x := _make(r))]
    noise = [x for r in load_jsonl(NOISE_PATH) if (x := _make(r))]
    return clean, noise


# ── 5-fold OOF 학습 ───────────────────────────────────────────────────────────

def _train_fold(train_recs: list[dict], dev_recs: list[dict],
                names: list[str], gpu: int, fold: int) -> ArticleMultiLabelElectra:
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    random.seed(SEED + fold); np.random.seed(SEED + fold)
    torch.manual_seed(SEED + fold); torch.cuda.manual_seed_all(SEED + fold)

    tok    = AutoTokenizer.from_pretrained(BASE_MODEL)
    counts = Counter(a for r in train_recs for a in r["articles"])
    pos_w  = compute_article_pos_weight(dict(counts), names, len(train_recs)).to(device)
    crit   = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    model  = ArticleMultiLabelElectra(BASE_MODEL, names).to(device)
    optim  = torch.optim.AdamW(model.parameters(), lr=LR)

    tl = DataLoader(ArticleDataset(train_recs, tok, MAX_LEN, names),
                    batch_size=BATCH, shuffle=True, num_workers=0)
    dl = DataLoader(ArticleDataset(dev_recs, tok, MAX_LEN, names),
                    batch_size=BATCH, shuffle=False, num_workers=0)

    best_f1 = -1.0
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for b in tl:
            optim.zero_grad()
            loss = crit(model(b["input_ids"].to(device), b["attention_mask"].to(device),
                              b["token_type_ids"].to(device)),
                        b["labels"].to(device))
            loss.backward(); optim.step()
        probs, labels, _ = infer(model, dl, device)
        thr = tune_thresholds(probs, labels, names)
        m = per_article_metrics(probs, labels, names, thr)
        if m["macro_f1_excl_majority"] > best_f1:
            best_f1 = m["macro_f1_excl_majority"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        logger.info(f"  fold={fold} epoch={epoch} macro_f1_excl={m['macro_f1_excl_majority']:.4f}")

    model.load_state_dict(best_state)
    return model, tok


def _compute_oof(population: list[dict], names: list[str], gpu: int) -> np.ndarray:
    """문서 단위 5-fold. 반환: (n_records, n_articles) OOF 확률."""
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    docs = sorted(set(r["group"] for r in population))
    random.seed(SEED); random.shuffle(docs)
    doc_to_fold = {d: i % N_FOLDS for i, d in enumerate(docs)}

    n = len(population)
    oof = np.zeros((n, len(names)), dtype=np.float32)
    idx_map = {id(r): i for i, r in enumerate(population)}

    for fold in range(N_FOLDS):
        train_recs = [r for r in population if doc_to_fold[r["group"]] != fold]
        val_recs   = [r for r in population if doc_to_fold[r["group"]] == fold]
        val_idx    = [idx_map[id(r)] for r in val_recs]
        logger.info(f"  fold {fold+1}/{N_FOLDS}: train={len(train_recs)} val={len(val_recs)}")

        model, tok = _train_fold(train_recs, val_recs, names, gpu, fold)
        model.eval()
        dl = DataLoader(
            ArticleDataset([{"text": r["text"], "articles": [], "group": "g"} for r in val_recs],
                           tok, MAX_LEN, names),
            batch_size=BATCH, shuffle=False, num_workers=0)
        probs, _, _ = infer(model, dl, device)
        for i_fold, i_pop in enumerate(val_idx):
            oof[i_pop] = probs[i_fold]
        del model

    return oof


# ── CleanLab 필터 ─────────────────────────────────────────────────────────────

def _cleanlab_remove(population: list[dict], oof: np.ndarray,
                     names: list[str], n_remove: int) -> list[int]:
    """CleanLab으로 이슈 점수 산출 → 상위 n_remove 인덱스 반환.

    단일 클래스(양성 0건 또는 전부 양성)인 조는 건너뛴다 — CleanLab이 at least 2 classes
    를 요구하기 때문이다. 건너뛴 조는 점수에 기여하지 않는다.
    """
    from cleanlab.filter import find_label_issues as _find_binary

    labels = np.array([[1 if a in r["articles"] else 0 for a in names]
                       for r in population], dtype=int)

    # 레코드별 이슈 점수: 조별 binary 이슈 개수 합산
    issue_score = np.zeros(len(population), dtype=float)
    skipped = []
    for j, art in enumerate(names):
        y_j = labels[:, j]
        if len(set(y_j)) < 2:
            skipped.append(art)
            continue
        p_j = np.stack([1 - oof[:, j], oof[:, j]], axis=1)
        try:
            flags = _find_binary(labels=y_j, pred_probs=p_j, return_indices_ranked_by="self_confidence")
            issue_score[flags] += 1
        except Exception as e:
            logger.warning(f"  CleanLab {art} 건너뜀: {e}")
            skipped.append(art)
    if skipped:
        logger.info(f"  CleanLab 단일 클래스 건너뜀: {skipped}")

    ranked = np.argsort(-issue_score).tolist()
    return ranked[:n_remove]


# ── 신뢰도 임계값 필터 ────────────────────────────────────────────────────────

def _confidence_remove(population: list[dict], oof: np.ndarray,
                       names: list[str], n_remove: int) -> list[int]:
    """라벨된 조의 OOF 확률 평균이 낮은 순 → 상위 n_remove 인덱스."""
    name_idx = {a: j for j, a in enumerate(names)}
    scores = []
    for i, r in enumerate(population):
        labeled = [name_idx[a] for a in r["articles"] if a in name_idx]
        if labeled:
            score = float(oof[i, labeled].mean())
        else:
            score = 1.0   # 빈 라벨은 제거 대상 아님(score 높게)
        scores.append(score)
    return sorted(range(len(scores)), key=lambda i: scores[i])[:n_remove]


# ── E1 비교 틀 재사용 ─────────────────────────────────────────────────────────

def _score_pop(population: list[dict], ref: dict,
               keep_mask: np.ndarray, nonempty: bool = True) -> np.ndarray:
    out = []
    for i, r in enumerate(population):
        if not keep_mask[i]:
            continue
        fwd = set(r["articles"])
        if nonempty and not fwd:
            continue
        out.append(_f1(fwd, ref[r["doc"]]))
    return np.array(out)


def _compare_vs_random(scores_method: np.ndarray, scores_all: np.ndarray,
                       n_keep: int, rng: np.random.Generator,
                       label: str) -> dict:
    """scores_method(필터된 1,473건)를 B(무작위 10,000번)와 비교."""
    boot = np.array([
        scores_all[rng.choice(len(scores_all), size=min(n_keep, len(scores_all)), replace=False)].mean()
        for _ in range(BOOT_N)
    ])
    pct975 = float(np.percentile(boot, 97.5))
    m_mean = float(scores_method.mean()) if len(scores_method) else float("nan")
    verdict = ("이 필터가 질을 높인다(> B 97.5%ile)" if m_mean > pct975
               else "미판정")
    logger.info(f"  {label:<25} mean={m_mean*100:.1f}%  B_p975={pct975*100:.1f}%  {verdict}")
    return {"n": int(len(scores_method)), "mean_f1": float(m_mean),
            "B_p975": float(pct975), "diff_pp": float((m_mean - boot.mean()) * 100),
            "perm_p": float((boot >= m_mean).mean()), "verdict": verdict}


def _paired_vs_fbcheck(scores_method: np.ndarray, scores_fbcheck: np.ndarray,
                       rng: np.random.Generator) -> dict:
    """레코드 단위 페어드 부트스트랩(겹치는 레코드가 많다)."""
    n = min(len(scores_method), len(scores_fbcheck))
    diff = scores_method[:n] - scores_fbcheck[:n]
    boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(BOOT_CI)])
    lo, hi = float(np.percentile(boot, 2.5) * 100), float(np.percentile(boot, 97.5) * 100)
    obs = float(diff.mean() * 100)
    verdict = ("이 필터 > FB-Check(유의)" if lo > 0 else
               "FB-Check > 이 필터(유의)" if hi < 0 else "미판정")
    return {"diff_vs_fbcheck_pp": obs, "ci95_pp": [lo, hi], "verdict": verdict}


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu",        type=int, default=0)
    ap.add_argument("--skip-train", action="store_true",
                    help=f"{OOF_PATH.name} 저장본이 있으면 재사용")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    ref, _ = _ref_by_doc()

    clean, noise = _load_population(ref)
    population   = clean + noise        # 2,305건
    n_clean      = len(clean)           # 1,473
    n_remove     = len(noise)           # 832
    logger.info(f"모집단: CLEAN {n_clean} + NOISE {n_remove} = {len(population)}")

    # names는 article_v2와 동일하게 고정 (접힌 조 제거 재현)
    counts = Counter(a for r in population for a in r["articles"])
    names  = select_article_labels(dict(counts), MIN_SUP)
    logger.info(f"names: {names}")

    # ── 5-fold OOF ──
    if args.skip_train and OOF_PATH.exists():
        logger.info(f"  OOF 저장본 재사용: {OOF_PATH}")
        oof = np.load(OOF_PATH)["oof"]
    else:
        logger.info("  === 5-fold OOF 학습 시작 ===")
        oof = _compute_oof(population, names, args.gpu)
        np.savez_compressed(OOF_PATH, oof=oof)
        logger.info(f"  OOF 저장: {OOF_PATH}")

    # ── 각 필터로 제거할 인덱스 ──
    logger.info("  === CleanLab 이슈 찾기 ===")
    cl_remove_idx = set(_cleanlab_remove(population, oof, names, n_remove))
    logger.info(f"  CleanLab 제거: {len(cl_remove_idx)}건")

    logger.info("  === 신뢰도 임계값 ===")
    ct_remove_idx = set(_confidence_remove(population, oof, names, n_remove))
    logger.info(f"  신뢰도 임계값 제거: {len(ct_remove_idx)}건")

    # ── 비교용 마스크 ──
    all_mask  = np.ones(len(population), dtype=bool)
    cl_mask   = np.array([i not in cl_remove_idx for i in range(len(population))])
    ct_mask   = np.array([i not in ct_remove_idx for i in range(len(population))])
    fb_mask   = np.array([True]  * n_clean + [False] * n_remove)  # FB-Check = CLEAN만

    # ── FTC 대비 F1 계산 ──
    s_all  = _score_pop(population, ref, all_mask)
    s_cl   = _score_pop(population, ref, cl_mask)
    s_ct   = _score_pop(population, ref, ct_mask)
    s_fb   = _score_pop(population, ref, fb_mask)

    logger.info("  === E3 비교 ===")
    result_cl = _compare_vs_random(s_cl, s_all, n_clean, rng, "CleanLab")
    result_ct = _compare_vs_random(s_ct, s_all, n_clean, rng, "신뢰도임계값")
    result_fb = _compare_vs_random(s_fb, s_all, n_clean, rng, "FB-Check(참고)")

    # FB-Check와 페어드 비교 (겹치는 레코드: CLEAN 부분만)
    # CleanLab/CT가 남긴 레코드 중 FB-Check(CLEAN)와 겹치는 것
    clean_set = set(range(n_clean))
    cl_clean  = sorted(clean_set - cl_remove_idx)
    ct_clean  = sorted(clean_set - ct_remove_idx)
    fb_clean  = list(range(n_clean))

    paired_cl_fb = _paired_vs_fbcheck(
        np.array([_f1(set(population[i]["articles"]), ref[population[i]["doc"]]) for i in cl_clean]),
        np.array([_f1(set(population[i]["articles"]), ref[population[i]["doc"]]) for i in cl_clean]),
        rng,
    )

    # CleanLab과 FB-Check의 제거 집합 겹침
    cl_noise_removed   = cl_remove_idx & set(range(n_clean, len(population)))
    cl_clean_removed   = cl_remove_idx & clean_set
    ct_noise_removed   = ct_remove_idx & set(range(n_clean, len(population)))
    ct_clean_removed   = ct_remove_idx & clean_set

    report = {
        "experiment":    "E3",
        "population":    f"CLEAN {n_clean} + NOISE {n_remove} = {len(population)}",
        "label":         "forward, non-empty 기준",
        "seed":          SEED,
        "boot_n":        BOOT_N,
        "caveat_circular": (
            "CleanLab은 OOF 모델로 이슈를 찾는데 그 모델을 같은 라벨로 학습함 — "
            "순환 있음. 표에 명시한다"
        ),
        "A_all": {
            "n": int(len(s_all)),
            "mean_f1": float(s_all.mean()),
        },
        "CleanLab": {
            **result_cl,
            "n_removed":        len(cl_remove_idx),
            "n_noise_removed":  len(cl_noise_removed),
            "n_clean_removed":  len(cl_clean_removed),
            "overlap_with_noise_pct": float(len(cl_noise_removed) / n_remove * 100),
        },
        "ConfThreshold": {
            **result_ct,
            "n_removed":        len(ct_remove_idx),
            "n_noise_removed":  len(ct_noise_removed),
            "n_clean_removed":  len(ct_clean_removed),
            "overlap_with_noise_pct": float(len(ct_noise_removed) / n_remove * 100),
        },
        "FBCheck_ref": {
            **result_fb,
            "note": "E1의 C. 비교 기준점으로만 쓴다",
        },
    }

    save_json(report, OUT_PATH)
    logger.info(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
