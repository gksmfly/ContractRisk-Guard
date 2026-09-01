# backend/eval/maxlen_harm_check.py
"""
`max_len` 변경이 **해를 끼치지 않는지**만 판정한다. 사전 등록된 규칙으로.

## 왜 별도 스크립트인가

`article_gold_eval`은 체크포인트를 **하나만** 받는다 — 여러 개를 받으면 gold로 고르게
되고 그게 곧 누수이기 때문이다. 그 제약은 옳다. 하지만 지금 필요한 것은 선택이 아니라
**페어드 무해 확인**이고, 페어드 CI는 같은 조항에 대한 두 예측이 동시에 있어야 나온다.

그래서 이 파일은 목적을 좁혀 둘을 받되, **판정 규칙을 코드에 박아 사후 변경을 막는다:**

    채택   재현 하락이 유의하지 않다 (페어드 CI 상한 ≥ 0)
    기각   유의하게 나빠졌다 (CI 상한 < 0)

"더 좋은 쪽을 고른다"는 **하지 않는다.** 512가 더 좋게 나와도 그건 부수 결과이고,
근거는 여전히 "입력을 버리지 않는다"이다. 사후에 근거를 바꾸면 gold 오염이다.

## 판정에 쓰지 않는 것

    조 F1                참고. 자유도를 늘리지 않는다
    disagree_with_gpt    **순환**(음성 풀의 정답이 GPT 라벨 그 자체)이라 기록만

실행:
    .venv/bin/python -m backend.eval.maxlen_harm_check models/article_v1 models/_article_len512 --gpu 1
"""

import argparse
import json
from pathlib import Path

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

logger = load_logger("maxlen_harm_check.log")
OUT_PATH = PROJECT_ROOT / "data/eval/maxlen_harm_check.json"


def _score(model_dir: Path, texts: list[str], gpu: int, bs: int = 16):
    """체크포인트를 **자기 학습 시 max_len으로** 채점한다 — 배포하면 그렇게 도니까."""
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    names = metrics["article_names"]
    thr = np.array([(metrics.get("thresholds") or {})[a] for a in names], dtype=float)
    max_len = int((metrics.get("train_config") or {}).get("max_len", 256))
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = ArticleMultiLabelElectra.load(model_dir).to(device).eval()
    ds = ArticleDataset([{"text": t, "articles": [], "group": "g"} for t in texts], tok, max_len, names)
    P = []
    with torch.no_grad():
        for b in torch.utils.data.DataLoader(ds, batch_size=bs):
            P.append(torch.sigmoid(model(b["input_ids"].to(device), b["attention_mask"].to(device),
                                         b["token_type_ids"].to(device))).cpu().numpy())
    trunc = float(np.mean([len(tok(t)["input_ids"]) > max_len for t in texts]))
    del model
    torch.cuda.empty_cache()
    return np.concatenate(P), thr, names, max_len, trunc


def _f1_rows(P, thr, labels):
    f = []
    for p, y in zip(P >= thr, labels > 0.5):
        tp = int((p & y).sum())
        a = tp / p.sum() if p.sum() else 0.0
        b = tp / y.sum() if y.sum() else 0.0
        f.append(2 * a * b / (a + b) if a + b else 0.0)
    return np.array(f)


def _paired(d: np.ndarray, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    boot = np.sort([d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(5000)])
    return float(d.mean() * 100), float(boot[125]), float(boot[4875])


def main() -> None:
    ap = argparse.ArgumentParser(description="max_len 변경 무해 확인 (사전 등록된 규칙)")
    ap.add_argument("base", help="현행 배포 체크포인트")
    ap.add_argument("cand", help="후보 체크포인트")
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()

    gold = load_ftc_gold("clean")
    recs = exclude_gold_documents(load_article_records(PROJECT_ROOT / "data/fb_check/clean.jsonl"), gold)
    _, neg = split_negative_holdout(recs, 12, 42)
    negs = [r["text"] for r in neg if not r["articles"]]
    gt = [g["text"] for g in gold]

    logger.info("========== max_len 무해 확인 ==========")
    logger.info("  ★ 사전 등록: **주 판정은 gold 조항 단위 재현 하나뿐.** "
                "유의하게 나빠지지 않으면 채택한다. 더 좋은 쪽을 고르는 게 아니다")

    out = {}
    for tag, d in (("base", args.base), ("cand", args.cand)):
        P, thr, names, ml, tr = _score(Path(d), gt, args.gpu)
        Pn, *_ = _score(Path(d), negs, args.gpu)
        labels = np.array([[1.0 if a in g["articles"] else 0.0 for a in names] for g in gold])
        out[tag] = {"dir": d, "max_len": ml, "recall_rows": (P >= thr).any(1).astype(float),
                    "f1_rows": _f1_rows(P, thr, labels),
                    "disagree": float((Pn >= thr).any(1).mean()),
                    "gold_trunc": tr,
                    "thresholds": {a: float(t) for a, t in zip(names, thr)}}

    b, c = out["base"], out["cand"]
    logger.info(f"  {'':<10}{'max_len':>8}{'gold 재현':>11}{'조 F1':>9}{'GPT불일치':>11}{'gold 절단':>10}")
    for tag, r in (("현행", b), ("후보", c)):
        logger.info(f"  {tag:<10}{r['max_len']:>8}{r['recall_rows'].mean() * 100:>10.1f}%"
                    f"{r['f1_rows'].mean() * 100:>8.1f}%{r['disagree'] * 100:>10.1f}%"
                    f"{r['gold_trunc'] * 100:>9.1f}%")

    d_r, lo_r, hi_r = _paired(c["recall_rows"] - b["recall_rows"])
    d_f, lo_f, hi_f = _paired(c["f1_rows"] - b["f1_rows"])
    verdict = "기각 — 유의하게 나빠짐" if hi_r < 0 else "채택 — 해가 없다"
    logger.info(f"  ★ 주 판정  조항 재현 {d_r:+.1f}%p [{lo_r:+.1f},{hi_r:+.1f}]  →  **{verdict}**")
    logger.info(f"    보조     조 F1     {d_f:+.1f}%p [{lo_f:+.1f},{hi_f:+.1f}]  (참고. 판정에 안 씀)")
    logger.info(f"    기록     GPT불일치 {b['disagree'] * 100:.1f}% → {c['disagree'] * 100:.1f}%  "
                f"**순환 지표라 판정 금지**")

    save_json({"rule": "사전 등록: gold 조항 단위 재현이 유의하게 나빠지지 않으면 채택. 선택이 아니라 무해 확인",
               "base": {k: v for k, v in b.items() if not k.endswith("_rows")},
               "cand": {k: v for k, v in c.items() if not k.endswith("_rows")},
               "primary_recall_diff_pp": d_r, "primary_ci95_pp": [lo_r, hi_r],
               "secondary_f1_diff_pp": d_f, "secondary_f1_ci95_pp": [lo_f, hi_f],
               "verdict": verdict}, OUT_PATH)
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
