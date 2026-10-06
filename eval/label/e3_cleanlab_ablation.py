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
    .venv/bin/python -m eval.label.e3_cleanlab_ablation --gpu 1
    .venv/bin/python -m eval.label.e3_cleanlab_ablation --gpu 1 --skip-train  # OOF 저장본 재사용
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from backend.model.electra import ArticleMultiLabelElectra
from backend.training.train import BASE_MODEL
from backend.training.train_article import (
    ArticleDataset,
    compute_article_pos_weight,
    infer,
    per_article_metrics,
    select_article_labels,
    tune_thresholds,
)
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json
from eval.channel.fbcheck_external_eval import _doc_of, _f1, _labels, _ref_by_doc
from eval.label.label_ablation import filter_diff_ci, random_keep_baseline

logger = load_logger("e3_cleanlab_ablation.log")

CLEAN_PATH = PROJECT_ROOT / "data/fb_check/clean.jsonl"
NOISE_PATH = PROJECT_ROOT / "data/fb_check/noise.jsonl"
OOF_PATH   = PROJECT_ROOT / "data/eval/e3_oof_probs.npz"
REMOVED_PATH = PROJECT_ROOT / "data/eval/e3_removed_ids.json"
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
            "chunk_id":     r.get("chunk_id", ""),
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

def _compare_vs_random(scores: np.ndarray, has: np.ndarray, keep: np.ndarray,
                       rng: np.random.Generator, label: str, docs: list[str] | None = None) -> dict:
    """필터가 남긴 집합 vs 레코드 수준 같은 수 무작위 유지 (label_ablation과 같은 기준선).

    2026-09-30 수정: 옛 구현은 빈 라벨을 뺀 배열에서 1,473개를 뽑아 필터 쪽(빈 라벨 제외
    1,142~1,287건)과 표본 크기가 달랐다 — 통과 쪽으로 기운다.
    """
    r = random_keep_baseline(scores, has, keep, rng, BOOT_N)
    rs = random_keep_baseline(scores, has, keep, rng, BOOT_N, groups=docs) if docs else None
    verdict = "이 필터가 질을 높인다(> B 97.5%ile)" if r["pass"] else "미판정"
    logger.info(f"  {label:<25} n={r['n_scored']} mean={r['mean_f1']*100:.1f}%  "
                f"B_p975={r['B_p975']*100:.1f}%  Δ={r['diff_pp']:+.1f}%p  {verdict}")
    return {"n": r["n_scored"], "mean_f1": r["mean_f1"], "B_p975": r["B_p975"],
            "diff_pp": r["diff_pp"], "perm_p": r["perm_p"], "verdict": verdict,
            "docstrat": ({"diff_pp": rs["diff_pp"], "perm_p": rs["perm_p"],
                          "verdict": "질을 높인다(문서 층화)" if rs["pass"] else "미판정"} if rs else None)}


def _paired_vs_fbcheck(scores: np.ndarray, has: np.ndarray, keep_method: np.ndarray,
                       keep_fb: np.ndarray, rng: np.random.Generator,
                       docs: list[str] | None = None) -> dict:
    """이 필터 − FB-Check. 레코드를 한 번에 재표집해 두 집합의 겹침을 보존한다.

    2026-09-30 수정: 옛 구현은 (1) 서로 다른 두 필터 결과 배열을 **위치 순서대로** 짝지어
    뺐고 (2) 호출부가 같은 배열을 두 번 넘겨 차가 항상 0이었으며 (3) 결과를 리포트에
    싣지도 않았다. "FB-Check > CleanLab +0.4%p"는 CI 없는 평균 차였다.
    """
    d = filter_diff_ci(scores, has, keep_method, keep_fb, rng, BOOT_CI, groups=docs)
    verdict = {"A > B(유의)": "이 필터 > FB-Check(유의)",
               "B > A(유의)": "FB-Check > 이 필터(유의)"}.get(d["verdict"], "미판정")
    return {"diff_vs_fbcheck_pp": d["diff_pp"], "ci95_pp": d["ci95_pp"],
            "mde80_pp": d["mde80_pp"], "verdict": verdict}


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
    cl_mask   = np.array([i not in cl_remove_idx for i in range(len(population))])
    ct_mask   = np.array([i not in ct_remove_idx for i in range(len(population))])
    fb_mask   = np.array([True]  * n_clean + [False] * n_remove)  # FB-Check = CLEAN만

    # ── FTC 대비 F1 계산 (레코드 벡터 + 마스크) ──
    has    = np.array([bool(r["articles"]) for r in population])
    scores = np.array([_f1(set(r["articles"]), ref[r["doc"]]) if r["articles"] else 0.0
                       for r in population])
    s_all  = scores[has]
    clean_set = set(range(n_clean))

    logger.info("  === E3 비교 ===")
    docs = [r["doc"] for r in population]
    result_cl = _compare_vs_random(scores, has, cl_mask, rng, "CleanLab", docs)
    result_ct = _compare_vs_random(scores, has, ct_mask, rng, "신뢰도임계값", docs)
    result_fb = _compare_vs_random(scores, has, fb_mask, rng, "FB-Check(참고)", docs)
    paired_cl_fb = _paired_vs_fbcheck(scores, has, cl_mask, fb_mask, rng)
    paired_ct_fb = _paired_vs_fbcheck(scores, has, ct_mask, fb_mask, rng)
    # 문서 군집 재표집 — 같은 문서 조항이 정답을 공유하므로 레코드 단위 CI는 좁게 나온다
    paired_cl_fb["doc_cluster"] = _paired_vs_fbcheck(scores, has, cl_mask, fb_mask, rng, docs)
    paired_ct_fb["doc_cluster"] = _paired_vs_fbcheck(scores, has, ct_mask, fb_mask, rng, docs)
    logger.info(f"  CleanLab − FB-Check  {paired_cl_fb}")
    logger.info(f"  신뢰도임계값 − FB-Check  {paired_ct_fb}")

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
            "vs_fbcheck": paired_cl_fb,
        },
        "ConfThreshold": {
            **result_ct,
            "n_removed":        len(ct_remove_idx),
            "n_noise_removed":  len(ct_noise_removed),
            "n_clean_removed":  len(ct_clean_removed),
            "overlap_with_noise_pct": float(len(ct_noise_removed) / n_remove * 100),
            "vs_fbcheck": paired_ct_fb,
        },
        "FBCheck_ref": {
            **result_fb,
            "note": "E1의 C. 비교 기준점으로만 쓴다",
        },
    }

    save_json(report, OUT_PATH)
    logger.info(f"저장: {OUT_PATH}")
    # 하류 학습 팔(E8 CL)이 같은 제거 집합을 쓰도록 ID를 남긴다
    save_json({"cleanlab_removed": sorted(population[i]["chunk_id"] for i in cl_remove_idx),
               "confthreshold_removed": sorted(population[i]["chunk_id"] for i in ct_remove_idx)},
              REMOVED_PATH)
    logger.info(f"제거 ID 저장: {REMOVED_PATH}")


if __name__ == "__main__":
    main()
