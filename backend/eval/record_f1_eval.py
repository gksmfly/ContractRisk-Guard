# backend/eval/record_f1_eval.py
"""**건별 집합 F1**로 모델을 상수와 비교한다 — 학습 로그의 macro F1과 단위가 다르다.

## 왜 별도 스크립트인가

`train_article`은 **조별 F1의 macro 평균**을 찍는다(조마다 이진 분류로 보고 평균).
교락 측정(`confound_articles`)의 상수 기준선 58.9%는 **건별 집합 F1**이다
(조항 하나의 예측 조 집합 vs 정답 조 집합, 그 F1을 건마다 내고 평균).

    macro F1 0.7256   조 8개 각각의 F1을 평균한 값
    건별 F1           예: 정답 {제6,제8} 예측 {제6} → F1 0.67, 이걸 건마다 평균

**둘은 비교 불가다.** 특히 "위반 없음"(빈 집합) 건이 macro에는 아예 안 들어가는데,
건별에서는 정답으로 쳐서 F1 1.0을 받는다 — 상수 "항상 빈 배열"이 58.9%를 받는 이유가
정확히 그것이다. macro 0.7256을 58.9%와 나란히 놓으면 오늘 하루 경계한 그 오독이 된다.

## 상수는 학습셋에서 도출한다

dev에서 최적 상수를 뽑아 dev에 적용하면 상수가 과대평가되고 모델이 부당하게 불리해진다.
`fixedsplit_verify.py`가 지름길 규칙을 학습셋에서만 뽑은 것과 같은 이유다.
비교용으로 dev-최적 상수(도달 불가능한 상한)도 함께 찍는다.

실행:
    .venv/bin/python -m backend.eval.record_f1_eval --gpu 1
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from backend.eval.confound_articles import _f1, best_constant
from backend.model.electra import ArticleMultiLabelElectra
from backend.training.train_article import (
    LABELED_PATH,
    ArticleDataset,
    apply_negative_ratio,
    exclude_gold_documents,
    load_article_records,
    load_ftc_gold,
    split_by_document,
    split_negative_holdout,
)
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("record_f1_eval.log")
OUT_PATH = PROJECT_ROOT / "data/eval/record_f1_eval.json"


def predict_sets(model_dir: Path, records: list[dict], names: list[str],
                 device: torch.device, max_len: int = 256) -> list[frozenset]:
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = ArticleMultiLabelElectra.load(model_dir).to(device).eval()   # 형식은 모델이 안다
    assert model.article_names == names, f"조 목록 불일치: {model.article_names} vs {names}"
    th = np.load(model_dir / "thresholds.npy")

    dl = DataLoader(ArticleDataset(records, tok, max_len, names), batch_size=32, shuffle=False)
    out = []
    with torch.no_grad():
        for batch in dl:
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device),
                           token_type_ids=batch["token_type_ids"].to(device))
            probs = torch.sigmoid(logits if not isinstance(logits, tuple) else logits[0]).cpu().numpy()
            for row in probs:
                out.append(frozenset(n for n, p, t in zip(names, row, th) if p >= t))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="건별 집합 F1로 모델 vs 상수")
    ap.add_argument("--model-dir", default=str(PROJECT_ROOT / "models/_article"))
    ap.add_argument("--labels", default=str(LABELED_PATH))
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--negative-holdout", type=int, default=12)
    ap.add_argument("--negative-ratio", type=float, default=None)
    ap.add_argument("--label-source", choices=["agreed", "forward"], default="agreed")
    a = ap.parse_args()

    meta = json.loads((Path(a.model_dir) / "metrics.json").read_text(encoding="utf-8"))
    names = meta["article_names"]
    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")

    # 학습과 **똑같은 순서**로 분할을 재현한다 — 하나라도 어긋나면 dev가 학습을 본다
    recs = load_article_records(Path(a.labels), label_source=a.label_source)
    recs = exclude_gold_documents(recs, load_ftc_gold())
    recs, _ = split_negative_holdout(recs, a.negative_holdout, 42)
    recs = apply_negative_ratio(recs, a.negative_ratio, a.seed)
    for r in recs:
        r["has_article"] = bool([x for x in r["articles"] if x in names])
    train_recs, dev_recs = split_by_document(recs, a.test_ratio, a.seed, stratify_key="has_article")
    assert len(dev_recs) == meta["dev_samples"], \
        f"dev 분할이 학습과 다르다 ({len(dev_recs)} vs {meta['dev_samples']}) — 설정이 어긋났다"

    gold = [frozenset(x for x in r["articles"] if x in names) for r in dev_recs]
    logger.info(f"========== 건별 집합 F1 | dev {len(dev_recs)}건 ==========")
    logger.info(f"  빈 라벨 {sum(1 for g in gold if not g) / len(gold) * 100:.1f}%")

    # 상수 — 학습셋에서 도출해 dev에 적용
    train_pairs = [("", frozenset(x for x in r["articles"] if x in names)) for r in train_recs]
    _, const = best_constant(train_pairs)
    const_f1 = sum(_f1(const, g) for g in gold) / len(gold)
    dev_opt_f1, dev_opt = best_constant([("", g) for g in gold])

    pred = predict_sets(Path(a.model_dir), dev_recs, names, device)
    model_f1 = sum(_f1(p, g) for p, g in zip(pred, gold)) / len(gold)

    logger.info(f"  {'상수(학습셋에서 도출)':<28}{const_f1 * 100:>7.1f}%   {sorted(const) or ['(위반 없음)']}")
    logger.info(f"  {'모델':<28}{model_f1 * 100:>7.1f}%")
    logger.info(f"  {'상수(dev 최적 — 도달 불가 상한)':<26}{dev_opt_f1 * 100:>7.1f}%   {sorted(dev_opt) or ['(위반 없음)']}")

    diff = (model_f1 - const_f1) * 100
    rng = random.Random(42)
    pairs = list(zip(pred, gold))
    boot = sorted(
        (sum(_f1(p, g) for p, g in s) / len(s)) - (sum(_f1(const, g) for _, g in s) / len(s))
        for s in (rng.choices(pairs, k=len(pairs)) for _ in range(5000)))
    lo, hi = boot[125] * 100, boot[4875] * 100
    logger.info(f"  → 모델 − 상수 {diff:+.1f}%p  (95% CI {lo:+.1f} ~ {hi:+.1f})")
    if lo > 0:
        logger.info("     **상수를 유의하게 이긴다** — 모델이 내용을 읽는다고 말할 수 있다")
    elif hi < 0:
        logger.error("     **상수보다 못하다.** 학습 데이터·라벨 정의를 먼저 볼 것 "
                     "(프롬프트·하이퍼파라미터가 아니다)")
    else:
        logger.warning("     **미판정** — CI가 0을 포함한다. 표본을 늘리거나 효과가 작다")

    save_json({"dev_n": len(dev_recs), "constant_from_train": {"f1": const_f1, "articles": sorted(const)},
               "model_f1": model_f1, "dev_optimal_constant_f1": dev_opt_f1,
               "diff_pp": diff, "ci95_pp": [lo, hi],
               "note": "건별 집합 F1. 학습 로그의 macro F1과 단위가 다르다 — 섞어 쓰지 말 것"}, OUT_PATH)
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
