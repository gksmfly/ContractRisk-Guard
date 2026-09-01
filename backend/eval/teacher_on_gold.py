# backend/eval/teacher_on_gold.py
"""teacher(gpt-4o)를 gold 327건에 돌려 **천장이 어디인지** 잰다.

## 무엇을 묻나

학생은 이미 gold에서 상수를 유의하게 이긴다(교차적합 49.4% vs 40.3%, +9.1%p).
그러니 물음은 "전이가 됐나"가 아니라 **"천장이 어디인가"** 다:

    teacher ≈ 49%   →  학생이 선생을 따라잡았다. 라벨 파이프라인이 천장이다
    teacher ≈ 65%   →  여유가 있다. 용량·증류가 지렛대다

## 비교의 비대칭

    teacher   임계값 없음. 집합을 그대로 출력
    학생      교차적합 임계값 8개를 맞춤 (편향 없지만 자유 파라미터를 씀)

학생 쪽에 유리하다. **teacher가 비슷하게 나오면 "동률"이 아니라 "teacher가 조금 앞선다".**

## 왜 forward만 부르나

채점에 쓰는 것은 `forward_articles`다. verify는 이 측정에 안 쓰이므로 순수 낭비다
(호출이 절반, $3.6 → $1.8).

실행:
    .venv/bin/python -m backend.eval.teacher_on_gold --dry-scope   # 건수·비용만
    .venv/bin/python -m backend.eval.teacher_on_gold
"""

import argparse
import json
import os
import time

import numpy as np
from openai import OpenAI

from backend.eval.confound_articles import _f1, best_constant
from backend.fb_check.forward_labeling import run_forward
from backend.training.train_article import load_ftc_gold
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("teacher_on_gold.log")
OUT_PATH = PROJECT_ROOT / "data/eval/teacher_on_gold.json"
CACHE = PROJECT_ROOT / "data/eval/teacher_on_gold_raw.jsonl"
_USD_PER_RECORD = 0.0055          # forward만, 프롬프트 캐싱 반영


def _cached() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    out = {}
    with open(CACHE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    out[r["key"]] = r
                except json.JSONDecodeError:
                    pass                      # 중단 시점의 잘린 줄
    return out


def _paired_ci(a: list[float], b: list[float], seed: int = 42) -> tuple[float, float, float]:
    d = np.array(a) - np.array(b)
    rng = np.random.default_rng(seed)
    boot = np.sort([d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(5000)])
    return float(d.mean() * 100), float(boot[125]), float(boot[4875])


def main() -> None:
    ap = argparse.ArgumentParser(description="teacher(gpt-4o)를 gold에 돌려 천장을 잰다")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--student-f1", type=float, default=0.494,
                    help="비교할 학생의 gold 교차적합 F1 (article_gold_eval의 threshold_regimes.crossfit)")
    ap.add_argument("--dry-scope", action="store_true", help="건수·비용만 찍고 종료")
    args = ap.parse_args()

    gold = load_ftc_gold()
    cache = _cached()
    pending = [g for g in gold if g["doc_id"] + "|" + g["text"][:40] not in cache]
    logger.info(f"========== teacher-on-gold | {args.model} ==========")
    logger.info(f"  gold {len(gold)}건 | 캐시 {len(cache)}건 | 처리 대상 {len(pending)}건")
    logger.info(f"  예상 비용 ${len(pending) * _USD_PER_RECORD:.2f} (forward만, 호출 {len(pending)}회)")
    if args.dry_scope:
        logger.info("  --dry-scope: API 호출 없이 종료")
        return

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    with open(CACHE, "a", encoding="utf-8") as f:
        for i, g in enumerate(pending, 1):
            key = g["doc_id"] + "|" + g["text"][:40]
            fwd = run_forward(client, g["text"], model=args.model)
            row = {"key": key, "doc_id": g["doc_id"],
                   "articles": (fwd or {}).get("articles", []), "ok": fwd is not None}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            cache[key] = row
            if i % 25 == 0:
                logger.info(f"  [{i}/{len(pending)}]")
            time.sleep(0.2)

    rows = [(frozenset(g["articles"]), frozenset(cache[g["doc_id"] + "|" + g["text"][:40]]["articles"]))
            for g in gold if g["doc_id"] + "|" + g["text"][:40] in cache]
    G = [x[0] for x in rows]
    cf, cset = best_constant([("x", g) for g in G])
    const = [_f1(cset, g) for g in G]
    teach = [_f1(p, g) for g, p in rows]
    # 학생은 **페어드 비교를 하지 않는다.** 집계값(`--student-f1`) 하나만 알고 건별 점수가
    # 없어서, 같은 값을 len(G)만큼 복제해 봐야 분산 0인 가짜 벡터가 된다 — CI가 실제보다
    # 좁게 나온다. 아래 ①(teacher vs 상수)만 페어드로 재고, 학생과의 격차는 평균 차이로만
    # 보고한다. 건별 점수를 확보하면 그때 페어드로 바꿀 것.

    logger.info(f"  채점 {len(rows)}건 | 상수 {sorted(cset)} = {cf * 100:.1f}%")
    logger.info("  ----- 결과 -----")
    logger.info(f"    teacher (gpt-4o)   {np.mean(teach) * 100:5.1f}%   임계값 없음")
    logger.info(f"    학생 (교차적합)      {args.student_f1 * 100:5.1f}%   임계값 8개를 맞춤 ← 학생에 유리")
    logger.info(f"    상수                {cf * 100:5.1f}%")
    d, lo, hi = _paired_ci(teach, const)
    v = "신호 있음" if lo > 0 else ("미판정" if hi > 0 else "상수보다 못함")
    logger.info(f"  ① teacher vs 상수   {d:+.1f}%p [{lo:+.1f},{hi:+.1f}] — {v}")
    gap = (np.mean(teach) - args.student_f1) * 100
    logger.info(f"  ② teacher vs 학생   {gap:+.1f}%p (평균 비교)")
    if abs(gap) < 5:
        logger.info("     **천장에 도달했다고 본다** — 라벨 파이프라인이 한계다. 용량을 키워도 "
                    "안 오른다. 다음 수는 라벨 소스 교체(의결서 대조표·다른 준거)")
    elif gap > 0:
        logger.info("     여유가 있다 — 용량·증류가 지렛대다. 다만 학생은 임계값 8개를 "
                    "맞춘 값이므로 실제 여유는 이보다 크다")
    save_json({"n": len(rows), "model": args.model, "teacher_f1": float(np.mean(teach)),
               "student_crossfit_f1": args.student_f1, "constant": {"f1": cf, "articles": sorted(cset)},
               "teacher_vs_constant_pp": d, "teacher_vs_constant_ci95": [lo, hi],
               "teacher_vs_student_pp": gap,
               "note": "teacher는 임계값 없음, 학생은 교차적합 임계값 8개 사용 — 학생에 유리한 비교"},
              OUT_PATH)
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
