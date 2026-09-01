# backend/eval/input_parity_eval.py
"""
**학습·평가는 조항 원문을, 운영은 evidence_span을 넣고 있었다.** 그 격차를 잰다.

## 무엇이 어긋나 있었나

    학습    train_article.py      r["text"]                     조항 원문 (평균 183자)
    평가    article_gold_eval.py  r["text"]                     조항 원문   ← 재현 78.0%가 나온 조건
    운영    judgment_node         evidence_span or clause       GPT가 뽑은 조각 (평균 43자)

`models/v4`(domain+risk)는 evidence_span 증강으로 학습했으므로 운영에서 span을 넣는 것이
맞았다. 조 multi-label로 갈아타면서 학습은 원문 전용(`augment=False`, 규칙 6)으로 바뀌었는데
**운영 입력만 그대로 남았다.** README에 실린 재현 78.0%는 운영이 실제로 받는 입력을
설명하지 않는다.

## 왜 단순한 "짧은 입력" 문제가 아닌가 — 입력이 정답과 상관한다

evidence_span은 GPT가 **위반이라고 본 부분**을 뽑은 것이므로, 위반이 없다고 판단하면
비어 있다. 실측 보유율:

    gold clean (위반)      255건 중 135건 (52.9%)  span 있음
    음성 holdout (비위반)  151건 중   2건 ( 1.3%)  span 있음

즉 운영에서 **양성 쪽은 절반이 조각으로, 음성 쪽은 거의 전부가 원문으로** 들어간다.
입력 길이가 정답과 상관하므로 재현과 오경보가 **서로 다른 방향으로** 어긋난다 —
"입력이 짧아져 전반적으로 나빠진다"가 아니라 **한쪽 축만 나빠진다.**

## 세 조건을 낸다

    text        조항 원문                  ← 학습·평가 조건 (보고돼 있던 값)
    serving     evidence_span or text      ← `judgment_node`의 규칙 그대로. 운영 실제값
    span_135    span 보유분만 원문 vs 조각  ← 페어드. 구성 차이를 빼고 입력 효과만 본다

`serving`과 `text`의 차이에는 **입력 효과와 구성 효과가 섞여 있다**(span을 가진 135건은
GPT가 위반을 찾은 건이라 애초에 다른 집단이다). `span_135`는 같은 조항을 두 입력으로
넣어 비교하므로 그 혼입이 없다. **판단은 span_135로 한다.**

## 한계

evidence_span은 `fb_check`의 forward 패스(**gpt-4o**)가 만든 것이다. 운영 Analysis Agent는
`gpt-4o-mini`라 실제 span은 이것과 다르다 — 두 채널 게이트 판단이 막혀 있는 것과 같은
조건 불일치다. 여기서 재는 것은 "**원문 대신 40자대 조각을 넣으면 얼마나 달라지는가**"이지
운영 span의 정확한 재현이 아니다.

실행:
    .venv/bin/python -m backend.eval.input_parity_eval models/article_v1 --gpu 1
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from backend.model.electra import ArticleMultiLabelElectra
from backend.training.train_article import (
    ArticleDataset,
    exclude_gold_documents,
    load_article_records,
    load_ftc_gold,
    split_negative_holdout,
)
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("input_parity_eval.log")

FB_RESULTS = Path(os.environ.get("FB_RESULTS", str(PROJECT_ROOT / "data/fb_check/fb_check_results.jsonl")))
CLEAN_PATH = Path(os.environ.get("ARTICLE_LABELS", str(PROJECT_ROOT / "data/fb_check/clean.jsonl")))
OUT_PATH   = PROJECT_ROOT / "data/eval/input_parity_report.json"


def _span_by_text() -> dict[str, str]:
    """조항 원문 → GPT가 뽑은 evidence_span.

    CLEAN만이 아니라 `fb_check_results.jsonl` 전체를 읽는다 — 운영에서는 NOISE 여부와
    무관하게 Analysis Agent가 span을 내므로, 커버리지를 게이트로 좁히면 운영과 멀어진다.
    """
    out: dict[str, str] = {}
    with open(FB_RESULTS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = str(r.get("text") or "").strip()
            s = str(r.get("evidence_span") or "").strip()
            if t and s and t not in out:
                out[t] = s
    return out


def _probs(records: list[dict], model: Any, tokenizer: Any, device: Any, names: list[str], bs: int, max_len: int) -> np.ndarray:
    ds = ArticleDataset(records, tokenizer, max_len, names)
    P = []
    with torch.no_grad():
        for b in torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False):
            P.append(torch.sigmoid(model(b["input_ids"].to(device), b["attention_mask"].to(device),
                                         b["token_type_ids"].to(device))).cpu().numpy())
    return np.concatenate(P) if P else np.zeros((0, len(names)))


def _sample_f1(probs: np.ndarray, labels: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """건별 F1 배열 — 평균만 내면 페어드 CI를 못 만든다."""
    f1s = []
    for p_row, y_row in zip(probs >= thr, labels > 0.5):
        tp = int((p_row & y_row).sum())
        pr = tp / p_row.sum() if p_row.sum() else 0.0
        rc = tp / y_row.sum() if y_row.sum() else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return np.array(f1s)


def _paired_ci(d: np.ndarray, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    boot = np.sort([d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(5000)])
    return float(d.mean() * 100), float(boot[125]), float(boot[4875])


def main() -> dict:
    ap = argparse.ArgumentParser(description="운영 입력(evidence_span) vs 학습 입력(원문) 정합 측정")
    ap.add_argument("model_dir")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    names = metrics["article_names"]
    saved = metrics.get("thresholds") or {}
    if not saved:
        raise SystemExit("metrics.json에 thresholds가 없다 — 여기서 고르면 그게 곧 오염이다")
    thr = np.array([saved[a] for a in names], dtype=float)

    spans = _span_by_text()
    gold = load_ftc_gold("clean")
    recs = exclude_gold_documents(load_article_records(CLEAN_PATH), gold)
    _, neg = split_negative_holdout(recs, 12, 42)
    neg = [{"text": r["text"], "articles": [], "group": "g"} for r in neg if not r["articles"]]

    g_span = [spans.get(g["text"].strip(), "") for g in gold]
    n_span = [spans.get(r["text"].strip(), "") for r in neg]
    has_g = [i for i, s in enumerate(g_span) if s]
    has_n = [i for i, s in enumerate(n_span) if s]

    logger.info(f"========== 입력 정합 측정 | {model_dir.name} ==========")
    logger.info("  ★ evidence_span은 정답과 상관한다 — 운영 입력이 라벨에 따라 달라진다")
    logger.info(f"    gold clean(위반)     {len(gold):>4}건 중 {len(has_g):>4}건 "
                f"({len(has_g) / len(gold) * 100:.1f}%) span 있음")
    logger.info(f"    음성 holdout(비위반) {len(neg):>4}건 중 {len(has_n):>4}건 "
                f"({len(has_n) / max(1, len(neg)) * 100:.1f}%) span 있음")
    if has_g:
        rr = [len(g_span[i]) / max(1, len(gold[i]["text"])) for i in has_g]
        logger.info(f"    span 평균 {np.mean([len(g_span[i]) for i in has_g]):.0f}자 / "
                    f"원문 평균 {np.mean([len(gold[i]['text']) for i in has_g]):.0f}자 "
                    f"(원문의 {np.mean(rr) * 100:.0f}%)")

    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = ArticleMultiLabelElectra.load(model_dir).to(device).eval()

    def as_recs(texts: list[str], srcs: list[str]) -> list[dict]:
        return [{"text": t, "articles": s["articles"], "group": "g"} for t, s in zip(texts, srcs)]

    # 세 조건. serving은 `judgment_node`의 규칙 그대로: evidence_span or clause
    g_text = [g["text"] for g in gold]
    g_serv = [g_span[i] or g_text[i] for i in range(len(gold))]
    n_text = [r["text"] for r in neg]
    n_serv = [n_span[i] or n_text[i] for i in range(len(neg))]

    P_text = _probs(as_recs(g_text, gold), model, tokenizer, device, names, args.batch_size, args.max_len)
    P_serv = _probs(as_recs(g_serv, gold), model, tokenizer, device, names, args.batch_size, args.max_len)
    N_text = _probs(as_recs(n_text, neg), model, tokenizer, device, names, args.batch_size, args.max_len)
    N_serv = _probs(as_recs(n_serv, neg), model, tokenizer, device, names, args.batch_size, args.max_len)
    labels = np.array([[1.0 if a in g["articles"] else 0.0 for a in names] for g in gold])

    def row(P: Any, N: Any) -> dict:
        return {
            "clause_recall": float(np.mean((P >= thr).any(1))) if len(P) else 0.0,
            "disagree_with_gpt": float(np.mean((N >= thr).any(1))) if len(N) else 0.0,
            "article_f1": float(_sample_f1(P, labels, thr).mean()) if len(P) else 0.0,
        }

    report = {"text": row(P_text, N_text), "serving": row(P_serv, N_serv)}

    logger.info("  ----- 전체 (구성 효과 섞임) -----")
    logger.info(f"  {'조건':<28}{'조항 재현':>10}{'GPT불일치':>11}{'조 F1':>9}")
    for k, lab in (("text", "text        학습·평가 조건"), ("serving", "serving     운영 실제")):
        v = report[k]
        logger.info(f"  {lab:<28}{v['clause_recall'] * 100:>9.1f}%{v['disagree_with_gpt'] * 100:>10.1f}%"
                    f"{v['article_f1'] * 100:>8.1f}%")

    # ----- 페어드: span 보유분만. 같은 조항을 두 입력으로 넣는다 -----
    if has_g:
        idx = np.array(has_g)
        r_t = (P_text[idx] >= thr).any(1).astype(float)
        r_s = (P_serv[idx] >= thr).any(1).astype(float)
        f_t = _sample_f1(P_text[idx], labels[idx], thr)
        f_s = _sample_f1(P_serv[idx], labels[idx], thr)
        d_r, lo_r, hi_r = _paired_ci(r_s - r_t)
        d_f, lo_f, hi_f = _paired_ci(f_s - f_t)
        report["span_135"] = {
            "n": len(idx),
            "recall_text": float(r_t.mean()), "recall_span": float(r_s.mean()),
            "recall_diff_pp": d_r, "recall_ci95_pp": [lo_r, hi_r],
            "f1_text": float(f_t.mean()), "f1_span": float(f_s.mean()),
            "f1_diff_pp": d_f, "f1_ci95_pp": [lo_f, hi_f],
        }
        logger.info(f"  ----- 페어드: span 보유 {len(idx)}건, 같은 조항 · 두 입력 (판단 기준) -----")
        logger.info(f"    조항 재현   원문 {r_t.mean() * 100:5.1f}%  →  조각 {r_s.mean() * 100:5.1f}%   "
                    f"{d_r:+.1f}%p [{lo_r:+.1f},{hi_r:+.1f}] — "
                    f"{'유의하게 나빠짐' if hi_r < 0 else ('미판정' if lo_r < 0 else '유의하게 좋아짐')}")
        logger.info(f"    조 F1       원문 {f_t.mean() * 100:5.1f}%  →  조각 {f_s.mean() * 100:5.1f}%   "
                    f"{d_f:+.1f}%p [{lo_f:+.1f},{hi_f:+.1f}] — "
                    f"{'유의하게 나빠짐' if hi_f < 0 else ('미판정' if lo_f < 0 else '유의하게 좋아짐')}")

    logger.info("  ⚠ evidence_span은 fb_check forward(**gpt-4o**)가 만든 것이다. 운영 Analysis는 "
                "gpt-4o-mini라 실제 span은 다르다 — 재는 것은 '조각을 넣으면 달라지는가'이지 "
                "운영 span의 재현이 아니다")
    save_json({"model_dir": str(model_dir), "thresholds": {a: float(t) for a, t in zip(names, thr)},
               "span_coverage": {"gold": [len(has_g), len(gold)], "negative": [len(has_n), len(neg)]},
               "conditions": report,
               "note": "판단은 span_135(페어드)로 한다. text vs serving 차이에는 구성 효과가 섞인다",
               "caveat": "span 생성기는 gpt-4o(fb_check), 운영 Analysis는 gpt-4o-mini"},
              OUT_PATH)
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
