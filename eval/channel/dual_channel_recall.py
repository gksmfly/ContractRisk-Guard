# backend/eval/dual_channel_recall.py
"""두 채널 결합 재현율 — **운영 모델(gpt-4o-mini)** 기준 재측정. B5.

## 왜 재는가

기존 +13.4%p(두 채널 ③ 재현율 이득)는 `gpt-4o`로 측정한 forward_articles 기반이다.
운영 Analysis는 `.env`의 `FORWARD_MODEL=gpt-4o-mini`이므로, 실제 제품에서 사용자가
계약서를 올렸을 때 두 채널이 얼마나 잡는지는 여기서 나온다.

## 측정 방법

1. Gold clean 255건을 `run_forward(model="gpt-4o-mini")`로 실행 — mini의 조 판단을 얻는다
2. `predict_articles(text)`로 인코더 article_v2의 조항 지목 여부를 얻는다
3. 세 구역 계산:
    모델만 지목    recall_model_only = 모델 O, mini ✗
    mini만 지목    recall_mini_only   = 모델 ✗, mini O
    둘 다 지목     recall_both        = 모델 O, mini O
    결합 재현율    recall_union       = 모델 O OR mini O

## 기준선

    article_v2 단독      78.4%  (gold clean 255, 배포 임계값, article_gold_eval)
    GPT(gpt-4o) 단독    88.2%  (prompt_block_ablation.md §"두 채널 분리")
    ③ 결합(gpt-4o)      91.3%  ─ 지금 재는 것은 이 값의 mini 판본이다

## 임계값

모델: thresholds.npy(배포 임계값) 그대로. 여기서 다시 고르지 않는다.
GPT: articles가 하나라도 있으면 "지목". 조 이름이 ARTICLE_IDS 안에 있어야 유효.

실행:
    .venv/bin/python -m eval.dual_channel_recall
    .venv/bin/python -m eval.dual_channel_recall --limit 20   # 연기 테스트
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from backend.labeling.articles import ARTICLE_IDS
from backend.training.train_article import load_ftc_gold
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("dual_channel_recall.log")

OUT_PATH   = PROJECT_ROOT / "data/eval/dual_channel_recall_mini.json"
CACHE_PATH = OUT_PATH.with_suffix(".cache.json")


def _gpt_flag(result: dict | None) -> bool:
    """articles에 유효한 조가 하나라도 있으면 True."""
    if not result:
        return False
    arts = result.get("articles") or []
    return bool([a for a in arts if a in set(ARTICLE_IDS)])


def _recall_at_clause(pred_flag: bool, gold_arts: list[str], names: set[str]) -> float:
    """조항 단위 재현: gold에 조가 있고 모델이 지목했으면 1.0, 지목 안 했으면 0.0."""
    has_gold = bool([a for a in gold_arts if a in names])
    if not has_gold:
        return float("nan")  # gold 없으면 제외
    return 1.0 if pred_flag else 0.0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini-model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-gpt", action="store_true", help="캐시 재사용")
    args = ap.parse_args(argv)

    import openai
    from backend.fb_check.forward_labeling import run_forward
    from backend.agents.judgment_agent import predict_articles

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # ── gold 로드 ─────────────────────────────────────────────────────────────
    gold = load_ftc_gold("clean")
    if args.limit:
        gold = gold[: args.limit]
    logger.info(f"★ gold clean {len(gold)}건 로드")

    # 조 이름 집합 (article_v2 기준)
    import json as _json
    metrics = _json.loads((PROJECT_ROOT / "models/article_v2/metrics.json").read_text())
    names = set(metrics["article_names"])

    # ── GPT-mini forward 실행 ────────────────────────────────────────────────
    if args.skip_gpt and CACHE_PATH.exists():
        mini_results = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        logger.info(f"  GPT 캐시 재사용: {CACHE_PATH} ({len(mini_results)}건)")
    else:
        logger.info(f"  GPT({args.mini_model}) forward 실행 중…")
        mini_results = []
        for i, g in enumerate(gold):
            if i % 20 == 0:
                logger.info(f"  {i}/{len(gold)}건…")
            r = run_forward(client, g["text"], model=args.mini_model)
            mini_results.append(r)
            time.sleep(0.03)
        CACHE_PATH.write_text(json.dumps(mini_results, ensure_ascii=False, indent=1),
                              encoding="utf-8")
        logger.info(f"  GPT 완료 → 캐시: {CACHE_PATH}")

    # ── 인코더 추론 ───────────────────────────────────────────────────────────
    logger.info("  인코더(article_v2) 추론…")
    model_flags = []
    for i, g in enumerate(gold):
        if i % 50 == 0:
            logger.info(f"  {i}/{len(gold)}건…")
        model_flags.append(bool(predict_articles(g["text"])))

    # ── 채점 ─────────────────────────────────────────────────────────────────
    mini_flags = [_gpt_flag(r) for r in mini_results]

    rows = []
    for g, mf, gf in zip(gold, model_flags, mini_flags):
        gold_arts = [a for a in g["articles"] if a in names]
        has_gold = bool(gold_arts)
        rows.append({
            "text": g["text"][:100],
            "gold_arts": gold_arts,
            "model_flag": mf,
            "mini_flag": gf,
            "mini_arts": [a for a in ((mini_results[gold.index(g)] or {}).get("articles") or []) if a in names]
                          if not args.skip_gpt else [],
        })

    # 조항 단위 재현: gold_arts 있는 건만
    has_gold_idx = [i for i, r in enumerate(rows) if r["gold_arts"]]
    n_has = len(has_gold_idx)
    logger.info(f"  gold에 조 있는 조항: {n_has}/{len(gold)}")

    def _rc(mask):
        return float(np.mean([1.0 if mask[i] else 0.0 for i in has_gold_idx]))

    rc_model   = _rc(model_flags)
    rc_mini    = _rc(mini_flags)
    both       = [model_flags[i] and mini_flags[i] for i in range(len(gold))]
    union_flag = [model_flags[i] or mini_flags[i] for i in range(len(gold))]
    rc_both    = _rc(both)
    rc_union   = _rc(union_flag)

    model_only = sum(1 for i in has_gold_idx if model_flags[i] and not mini_flags[i])
    mini_only  = sum(1 for i in has_gold_idx if not model_flags[i] and mini_flags[i])
    both_cnt   = sum(1 for i in has_gold_idx if model_flags[i] and mini_flags[i])
    neither    = sum(1 for i in has_gold_idx if not model_flags[i] and not mini_flags[i])

    delta_pp = rc_union - rc_model

    logger.info("=" * 66)
    logger.info(f"★ 인코더(article_v2) 단독    {rc_model*100:.1f}%  ({sum(model_flags[i] for i in has_gold_idx)}/{n_has})")
    logger.info(f"★ GPT-mini 단독              {rc_mini*100:.1f}%  ({sum(mini_flags[i] for i in has_gold_idx)}/{n_has})")
    logger.info(f"★ 결합(union) 재현율         {rc_union*100:.1f}%  → 인코더 대비 {delta_pp*100:+.1f}%p")
    logger.info(f"  둘 다 지목: {both_cnt}건 · 모델만: {model_only}건 · mini만: {mini_only}건 · 둘 다 못 잡음: {neither}건")
    logger.info(f"  참고 — gpt-4o 기준 결합 91.3%(+13.4%p). 지금 값은 mini 기준.")
    logger.info("=" * 66)

    result = {
        "n_gold": len(gold),
        "n_has_gold_art": n_has,
        "mini_model": args.mini_model,
        "recall_model": rc_model,
        "recall_mini": rc_mini,
        "recall_both": rc_both,
        "recall_union": rc_union,
        "delta_vs_model": delta_pp,
        "breakdown": {
            "model_only": model_only,
            "mini_only": mini_only,
            "both": both_cnt,
            "neither": neither,
        },
        "reference": {
            "article_v2_recall_reported": 0.784,
            "gpt4o_recall": 0.882,
            "gpt4o_union_recall": 0.913,
            "gpt4o_delta": 0.134,
        },
        "note": (
            "조항 단위 재현: gold에 조가 있고(n_has_gold_art) 모델/GPT가 지목하면 1. "
            "결합은 OR — 어느 채널이든 지목하면 잡힌다. "
            "이 값이 제품 화면에서 실제로 보여줄 수 있는 최대 재현율이다."
        ),
    }
    save_json({**result, "rows": rows}, OUT_PATH)
    logger.info(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
