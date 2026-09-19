# backend/eval/threshold_match_sweep.py
"""`article_v2`의 임계값을 낮춰 sLLM과 **같은 재현율**로 맞추고 오경보를 비교한다.

## 이 실험이 답하는 것 하나

EXAONE LoRA는 재현 78.4% → 87.8%(+9.4%p, p=0.0005)를 냈고 오경보는 12.1% → 14.2%
(+2.1%p)였다. 그런데 **인코더도 임계값만 낮추면 더 많이 짚는다.** 그러면 "7.8B가 더
낫다"가 아니라 "sLLM이 그냥 더 공격적인 동작점에 있다"가 된다. 두 설명은 **같은
재현율에서 오경보를 비교해야만** 갈린다.

    인코더를 재현 87.8%에 맞춘다  →  그때 오경보가 14.2%보다 낮으면 인코더가 낫다
                                    14.2%와 비슷하면 sLLM이 준 건 동작점 이동뿐이다

## 임계값을 gold에서 고르는 것 아닌가

맞다. 그래서 **이 값을 배포에 쓰지 않는다.** 여기서 구하는 건 "같은 재현율에서의
오경보"라는 **비교량**이지 배포 설정이 아니다. 배포 임계값은 여전히 dev에서 뽑은
`thresholds.npy`다(`article_gold_eval` 주석과 같은 규율).

실행:

    .venv/bin/python -m backend.eval.threshold_match_sweep --gpu 1
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from backend.eval.gpt_channel_fpr import _doc, intact_stratum, load_negative_pool
from backend.training.train_article import load_ftc_gold
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("threshold_match_sweep.log")
OUT_DIR = PROJECT_ROOT / "data/eval"   # 파일명에 태그를 붙인다 — 고정이면 다음 모델이 덮어쓴다


def _probs(texts: list[str], model_dir: str, gpu: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    import torch
    from transformers import AutoTokenizer

    from backend.model.electra import ArticleMultiLabelElectra
    from backend.training.train_article import ArticleDataset

    d = PROJECT_ROOT / model_dir
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    names = list(m["article_names"])
    max_len = int((m.get("train_config") or {}).get("max_len", 256))
    thr = np.load(d / "thresholds.npy")
    dev = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = ArticleMultiLabelElectra.load(d).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(str(d))
    ds = ArticleDataset([{"text": t, "articles": [], "group": "g"} for t in texts],
                        tok, max_len, names)
    out = []
    with torch.no_grad():
        for b in torch.utils.data.DataLoader(ds, batch_size=32):
            out.append(torch.sigmoid(model(b["input_ids"].to(dev), b["attention_mask"].to(dev),
                                           b["token_type_ids"].to(dev))).cpu().numpy())
    return np.concatenate(out), names, thr


def _boot_ci(vals: list[int], docs: list[str], seed: int = 42) -> tuple[float, float]:
    by: dict[str, list[int]] = {}
    for v, d in zip(vals, docs):
        by.setdefault(d, []).append(v)
    g = list(by.values())
    rng = np.random.default_rng(seed)
    s = [np.mean([x for i in rng.integers(0, len(g), len(g)) for x in g[i]]) for _ in range(3000)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/article_v2")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--tag", default="exaone")
    args = ap.parse_args(argv)

    sllm = json.loads((PROJECT_ROOT / f"data/eval/sllm_lora_{args.tag}.json").read_text(encoding="utf-8"))
    target_recall = sllm["recall_any"]
    sllm_fpr = sllm["fpr_intact"]

    gold = load_ftc_gold("clean")
    neg = intact_stratum(load_negative_pool())
    logger.info(f"★ 목표 재현 {target_recall*100:.1f}% (sLLM {args.tag}) · sLLM 오경보 {sllm_fpr*100:.1f}%")
    logger.info(f"  gold {len(gold)} · 음성 온전 지층 {len(neg)}")

    gp, names, thr = _probs([g["text"] for g in gold], args.model, args.gpu)
    np_, _, _ = _probs([r["text"] for r in neg], args.model, args.gpu)
    g_docs = [g["doc_id"] for g in gold]
    n_docs = [_doc(r["chunk_id"]) for r in neg]

    # 배포 임계값에 **곱하는 배율**을 훑는다. 조별 상대 관계(제9조가 가장 민감)를 유지한다 —
    # 조마다 따로 고르면 8차원을 gold에서 맞추는 것이라 과적합이 심해진다.
    rows = []
    for s in np.arange(1.0, 0.04, -0.02):
        t = thr * s
        rec = float(np.mean([(gp[i] >= t).any() for i in range(len(gp))]))
        fpr = float(np.mean([(np_[i] >= t).any() for i in range(len(np_))]))
        rows.append({"scale": round(float(s), 2), "recall": rec, "fpr": fpr})

    base = rows[0]
    match = min(rows, key=lambda r: abs(r["recall"] - target_recall))
    logger.info("")
    logger.info(f"  {'배율':>6}{'재현':>9}{'오경보':>10}")
    for r in rows:
        mark = ""
        if r is base:
            mark = "  ← 배포 임계값"
        elif r is match:
            mark = f"  ← 재현 {target_recall*100:.1f}% 정합점"
        if r is base or r is match or round(r["scale"] * 100) % 10 == 0:
            logger.info(f"  {r['scale']:>6.2f}{r['recall']*100:>8.1f}%{r['fpr']*100:>9.1f}%{mark}")

    t = thr * match["scale"]
    n_flag = [int(bool((np_[i] >= t).any())) for i in range(len(np_))]
    lo, hi = _boot_ci(n_flag, n_docs)
    diff = match["fpr"] - sllm_fpr
    verdict = ("인코더가 같은 재현에서 오경보가 더 낮다 — sLLM이 진 것"
               if hi < sllm_fpr else
               "겹친다 — sLLM이 준 것은 동작점 이동뿐이라는 설명과 구분되지 않는다"
               if lo <= sllm_fpr <= hi else
               "sLLM이 같은 재현에서 오경보가 더 낮다 — 실제 이득")

    logger.info("")
    logger.info("  " + "=" * 62)
    logger.info(f"  ★ 같은 재현율에서 비교 (재현 ≈ {match['recall']*100:.1f}%)")
    logger.info(f"    인코더(임계값 ×{match['scale']:.2f})  오경보 {match['fpr']*100:.1f}% "
                f"[{lo*100:.1f}, {hi*100:.1f}]")
    logger.info(f"    sLLM {args.tag:20} 오경보 {sllm_fpr*100:.1f}%")
    logger.info(f"    차 {diff*100:+.1f}%p → **{verdict}**")
    logger.info("  " + "=" * 62)
    logger.info("  ⚠ 이 배율은 gold에서 고른 값이라 **배포에 쓰지 않는다.** 비교량일 뿐이다")

    save_json({"model": args.model, "sllm_tag": args.tag,
               "target_recall": target_recall, "sllm_fpr": sllm_fpr,
               "deployed": base, "matched": match, "matched_fpr_ci95": [lo, hi],
               "diff_fpr": diff, "verdict": verdict, "sweep": rows,
               "caveat": "배율은 gold에서 고른 비교량. 배포 임계값은 dev에서 뽑은 thresholds.npy 그대로"},
              OUT_DIR / f"threshold_match_{args.tag}.json")
    logger.info(f"저장: data/eval/threshold_match_{args.tag}.json")


if __name__ == "__main__":
    main()
