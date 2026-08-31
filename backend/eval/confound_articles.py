# backend/eval/confound_articles.py
"""
출처 교락 측정 — **조 multi-label 판본.** 학습을 시작하기 전에 반드시 돌린다.

## 왜 새로 쓰는가

`confound_analysis.py`는 전부 `risk_level`(3-class single-label) 기준이라
(`RISK_MAP[r["risk_level"]]`, 출처별 최빈 `risk_level`) 새 라벨 형식에서는 조회부터
막힌다. 다만 그 파일의 **`source_separability`(TF-IDF로 출처 맞히기)는 라벨과 무관해서
그대로 재사용한다** — 사실 그게 가장 중요한 지표다.

## 무엇을 재나

    ① 출처 판별 가능성    TF-IDF+로지스틱으로 출처를 몇 %나 맞히는가   ← 라벨 무관, 재사용
    ② 출처 조건부 상수    출처별 최빈 조 집합을 찍는 예측기의 건별 F1  ← 새로 씀
    ③ 무조건 상수         항상 같은 조 집합(top1/top3)                 ← 비교용

**②가 이 파일의 핵심이다.** 교락이 완전하면 `standard_contract → {}`(위반 없음) 하나로
표본의 절반이 맞아버린다. 그게 지금 가장 걱정되는 시나리오고, 이 지표가 정확히 그걸 잰다.

그리고 ②가 ③보다 크게 높으면 **모델이 넘어야 할 선이 올라간다** — 무조건 상수를 이기는
걸로는 부족하고, **출처 조건부 상수를 이겨야** 내용을 읽었다고 말할 수 있다.

## 판단 기준

    출처 판별 정확도 90%대  →  문체만으로 코퍼스가 갈린다. 학습해도 그걸 배울 유인이 크다
    60%대로 내려오면       →  진짜 신호가 있다

안 재고 학습하면 "85.3%가 나왔는데 그중 얼마가 문체인지 모른다"가 또 반복된다
(`measurement_findings_2026-08-16` 문제 3).

실행:
    .venv/bin/python -m backend.eval.confound_articles
    .venv/bin/python -m backend.eval.confound_articles --label-source forward
"""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from backend.eval.confound_analysis import source_separability   # 라벨 무관 — 그대로 재사용
from backend.labeling.articles import ARTICLE_IDS
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("confound_articles.log")

CLEAN_PATH = PROJECT_ROOT / "data/fb_check/clean.jsonl"
OUT_PATH   = PROJECT_ROOT / "data/eval/confound_articles_report.json"


def _articles(r: dict, label_source: str) -> frozenset:
    if label_source == "agreed":
        a = r.get("agreed_articles")
        if a is None:
            a = r.get("forward_articles") or []
    else:
        a = r.get("forward_articles") or []
    return frozenset(x for x in a if x in ARTICLE_IDS)


def _f1(pred: frozenset, gold: frozenset) -> float:
    """건별 F1. 둘 다 비면 1.0 — "위반 없음"을 맞힌 것도 맞힌 것이다."""
    if not pred and not gold:
        return 1.0
    inter = len(pred & gold)
    p = inter / len(pred) if pred else 0.0
    r = inter / len(gold) if gold else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def best_constant(rows: list[tuple[str, frozenset]]) -> tuple[float, frozenset]:
    """이 표본에서 건별 F1을 최대화하는 상수 조집합을 **탐색해서** 찾는다.

    ## 왜 탐색인가 — 이 파일이 처음에 틀렸던 지점

    처음 판본은 무조건 상수를 "빈도 상위 k개 조"로만 만들었다(top1/top3). 그래서
    **빈 집합을 한 번도 후보에 넣지 않았다.** 그런데 조 multi-label에서 최적 상수는
    빈 집합("위반 없음")이다 — 표준계약서의 91.3%가 빈 배열이라 그렇다.

        보고했던 값   무조건 상수 12.9%, 출처 조건부 61.6%  →  "+48.8%p 교락, 심각"
        실제 값       무조건 상수 61.6%, 출처 조건부 62.6%  →  "+0.9%p, 사실상 없음"

    상수 기준선을 무르게 잡아 모델(또는 이 경우 교락)을 실제보다 좋아 보이게 만든
    전형적인 사례다 — 이 프로젝트가 반복해서 밟은 함정이고
    ([[feedback_measure_constant_baseline]]), **이번엔 상수 쪽에서 밟았다.**

    후보는 빈 집합 + 표본에 실제로 나타난 모든 조합 + 빈도 상위 6개 조의 1~4개 조합.
    """
    freq = Counter(a for _, arts in rows for a in arts)
    top = [a for a, _ in freq.most_common(6)]
    candidates: set[frozenset] = {frozenset()} | {arts for _, arts in rows}
    for k in range(1, 5):
        candidates |= {frozenset(c) for c in combinations(top, k)}
    return max((sum(_f1(c, arts) for _, arts in rows) / len(rows), c) for c in candidates)


def source_conditional_constant(rows: list[tuple[str, frozenset]]) -> tuple[float, dict]:
    """출처별로 **최적** 상수 조집합을 찍는 예측기의 건별 F1.

    `risk_level` 판본의 "출처별 최빈 라벨"을 조 집합으로 옮긴 것인데, 최빈이 아니라
    최적을 쓴다 — 최빈은 교락을 **과소평가**한다. 실제로 ftc_case의 최빈 조합은
    빈 배열(24.3%)이지만 F1 최적은 `{제6조, 제8조, 제10조}`(26.4%)다.

    이 값이 무조건 상수보다 크게 높으면 **출처만 알면 라벨을 거의 맞힐 수 있다**는 뜻이고,
    모델이 내용 대신 문체를 배울 유인이 그만큼 크다.
    """
    by_src: dict[str, list] = defaultdict(list)
    for src, arts in rows:
        by_src[src].append((src, arts))
    table, total = {}, 0.0
    for s, sub in by_src.items():
        f1, best = best_constant(sub)
        total += f1 * len(sub)
        table[s] = {"최적_조집합": sorted(best) or ["(위반 없음)"], "f1": f1, "n": len(sub),
                    "빈배열_비율": sum(1 for _, a in sub if not a) / len(sub)}
    return total / len(rows), table


def main() -> None:
    ap = argparse.ArgumentParser(description="조 multi-label 출처 교락 측정")
    ap.add_argument("--labels", default=str(CLEAN_PATH))
    ap.add_argument("--label-source", choices=["agreed", "forward"], default="agreed",
                    help="agreed는 표준계약서를 빈 배열로 밀어 교락을 키울 수 있다 — 둘 다 재볼 것")
    args = ap.parse_args()

    raw = load_jsonl(Path(args.labels))
    rows = [(r.get("source", "?"), _articles(r, args.label_source)) for r in raw if (r.get("text") or "").strip()]
    if not rows:
        raise SystemExit(f"{args.labels} 에 레코드가 없다 — 라벨링이 끝났는지 확인할 것")

    logger.info(f"========== 출처 교락 (조 multi-label) | label_source={args.label_source} ==========")
    logger.info(f"  레코드 {len(rows)}건 | 출처 {dict(Counter(s for s, _ in rows))}")

    # ① 출처 판별 가능성 — 라벨과 무관. 이게 높으면 문체만으로 코퍼스가 갈린다.
    texts = [(r.get("text") or "").strip() for r in raw if (r.get("text") or "").strip()]
    sep = source_separability(texts, [s for s, _ in rows])
    logger.info(f"  ① TF-IDF로 출처 맞히기        {sep * 100:5.1f}%   "
                f"(90%대면 문체만으로 갈린다 / 60%대면 진짜 신호)")

    # ③ 무조건 상수 — **빈 집합을 반드시 후보에 넣는다**(`best_constant` docstring 참고)
    uncond_f1, uncond_set = best_constant(rows)
    logger.info(f"  ③ 무조건 상수 (최적 탐색)      건별 F1 {uncond_f1 * 100:5.1f}%   "
                f"항상 {sorted(uncond_set) or ['(위반 없음)']}")
    logger.info(f"       ★ 학습된 모델이 최소한 넘어야 할 선이 이것이다")

    # ② 출처 조건부 상수
    cond_f1, table = source_conditional_constant(rows)
    logger.info(f"  ② 출처 조건부 상수 (최적 탐색)  건별 F1 {cond_f1 * 100:5.1f}%")
    for s, t in table.items():
        logger.info(f"       {s:<20} F1 {t['f1'] * 100:5.1f}%  n={t['n']:<5} → {t['최적_조집합']}"
                    f"  (빈배열 {t['빈배열_비율'] * 100:.1f}%)")

    gap = (cond_f1 - uncond_f1) * 100
    logger.info(f"  → 출처를 아는 것의 순수 이득 {gap:+.1f}%p   ← 라벨 쪽 교락의 크기")
    if gap > 10:
        logger.warning("     출처를 아는 것만으로 크게 유리하다 = 교락이 크다. "
                       "학습된 모델은 **출처 조건부 상수를 이겨야** 내용을 읽었다고 말할 수 있다")
    else:
        logger.info("     라벨 쪽 교락은 작다 — 출처를 알아도 라벨 예측이 거의 안 나아진다. "
                    "다만 ①이 높으면 모델이 출처를 **식별할 수는** 있다는 뜻이므로, "
                    "진짜 판단은 '지름길이 틀리는 구간'에서 한다")

    save_json({
        "n": len(rows), "label_source": args.label_source,
        "source_separability": sep,
        "source_conditional_constant": {"f1": cond_f1, "table": table},
        "unconditional_constant": {"f1": uncond_f1, "articles": sorted(uncond_set)},
        "gap_vs_unconditional_pp": gap,
        "note": "모델 평가 시 unconditional_constant.f1(넘어야 할 최소선)과 "
                "source_conditional_constant.f1 둘 다와 비교할 것",
    }, OUT_PATH)
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
