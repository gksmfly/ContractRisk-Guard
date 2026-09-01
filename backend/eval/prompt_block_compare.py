# backend/eval/prompt_block_compare.py
"""
조문 블록 절제 실험 채점 — A / A' / B / C 를 페어드로 비교한다.

판정 기준은 **실행 전에** `backend/eval/prompt_block_ablation.md`에 확정해 뒀다.
이 스크립트는 그 기준에 필요한 수치만 낸다.

## 왜 페어드 검정인가

같은 100건을 같은 순서로 돌렸으므로 건별 F1이 짝지어져 있다. "δ보다 크냐"를 눈으로
재는 것보다 Wilcoxon signed-rank가 방어하기 쉽다 — 이 프로젝트가 이진 결과에 McNemar를
쓴 것과 같은 계열이다.

**A vs A'는 음성 대조군이다.** 같은 프롬프트·temperature 0인데도 gpt-4o는 완전
결정적이지 않다. 여기서 유의하게 나오면 검정이 과민한 것이므로, 그 경우 B·C의 p값도
믿으면 안 된다.

## 조별 분포도 본다 (F1만 보면 놓친다)

파일럿에서 제6조를 과잉 예측했다(47 vs 정답 36). 제6조는 일반원칙이라 조문 전문이
"고객에게 부당하게 불리한 조항" 같은 포괄적 문구로 채워져 있다 — **그 전문을 매번
보여주는 것이 과잉 예측의 원인일 수 있다.** C(제목만)에서 제6조 예측이 정답 분포에
가까워지면 토큰 절감보다 큰 발견이다.

실행:
    .venv/bin/python -m backend.eval.prompt_block_compare
"""

import json

from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("prompt_block_compare.log")

EVAL_DIR = PROJECT_ROOT / "data/eval"
OUT_PATH = EVAL_DIR / "prompt_block_ablation_report.json"

# (표시 이름, 파일). A는 v4 프롬프트 실행 당시 저장본이라 별도 이름이다.
_RUNS = [
    ("A  (전문, 저장본)", "label_pilot_report_A_full.json"),
    ("A' (전문, 재실행)", "label_pilot_report_full.json"),
    ("B  (제목+요지)",   "label_pilot_report_summary.json"),
    ("C  (제목만)",      "label_pilot_report_title.json"),
]


def _f1(s: dict) -> float:
    """저장본에는 `f1` 필드가 없을 수 있다(추가 전 실행) — 정밀/재현에서 다시 만든다."""
    if "f1" in s:
        return float(s["f1"])
    p, r = s.get("precision", 0.0), s.get("recall", 0.0)
    return 2 * p * r / (p + r) if p + r else 0.0


def load_run(fname: str, model: str = "gpt-4o") -> dict | None:
    path = EVAL_DIR / fname
    if not path.exists():
        return None
    rep = json.loads(path.read_text(encoding="utf-8"))
    m = rep["models"].get(model)
    if not m:
        return None
    rows = m["rows"]
    return {
        "path": str(path),
        "block": rep.get("block", "full"),
        "case_names": [r["case_name"] for r in rows],
        "f1": [_f1(r["score_forward"]) for r in rows],
        "empty": [not r["forward_articles"] for r in rows],
        "pred_dist": m.get("pred_article_dist", {}),
        "gold_dist": m.get("gold_article_dist", {}),
    }


def total_deviation(pred: dict[str, int], gold: dict[str, int]) -> int:
    """총편차 Σ|pred − gold| — **1차 판정 지표.**

    건별 F1은 이 표본에서 노이즈 바닥이 3.2%p인데 기대 효과 크기도 그 언저리라
    판정에 쓸 수 없다(A vs A' 실측). 반면 조별 예측 분포는 재실행 간 최대 Δ2로
    훨씬 안정적이다 — 총편차로 보면 46 → 44, 스케일 45 대비 4%다.

    한계: 분포 수준 지표라 **건별 정확도를 보장하지 않는다.** 제6조를 정확히 36건
    예측해도 엉뚱한 조항에 붙였을 수 있다. 그래서 F1을 버리지 않고 2차로 병기한다.
    """
    return sum(abs(pred.get(a, 0) - gold.get(a, 0)) for a in set(gold) | set(pred))


def wilcoxon(a: list[float], b: list[float]) -> dict:
    """b − a 의 Wilcoxon signed-rank. 동점 쌍은 제외(표준 처리)."""
    from scipy.stats import wilcoxon as _w
    diffs = [y - x for x, y in zip(a, b) if y != x]
    if not diffs:
        return {"n_nonzero": 0, "p_value": 1.0, "note": "차이 있는 쌍이 없다"}
    stat, p = _w([y - x for x, y in zip(a, b)], zero_method="wilcox", alternative="two-sided")
    better = sum(1 for d in diffs if d > 0)
    return {"n_nonzero": len(diffs), "better": better, "worse": len(diffs) - better,
            "statistic": float(stat), "p_value": float(p)}


def main() -> None:
    runs = {name: load_run(f) for name, f in _RUNS}
    have = {k: v for k, v in runs.items() if v}
    missing = [k for k, v in runs.items() if not v]
    if missing:
        logger.warning(f"  아직 없는 실행: {missing}")
    if len(have) < 2:
        raise SystemExit("비교할 실행이 2개 미만이다 — label_pilot을 --block 별로 먼저 돌릴 것")

    base_key = next(iter(have))          # A가 기준
    base = have[base_key]

    # 표본이 같아야 페어드가 성립한다. 다르면 비교 자체를 중단한다.
    for k, v in have.items():
        if v["case_names"] != base["case_names"]:
            raise SystemExit(f"{k}의 표본이 {base_key}와 다르다 — 페어드 비교 불가")

    logger.info(f"========== 조문 블록 절제 ({len(base['f1'])}건, 표본 동일 확인) ==========")
    # F1은 **건별 F1의 평균**으로 고정한다. F1(평균정밀, 평균재현)은 다른 값이 나오고
    # (A 기준 50.83 vs 53.92), 두 숫자가 같이 돌아다니면 나중에 어느 쪽을 인용했는지
    # 알 수 없게 된다. 상수 기준선도 같은 정의로 계산되므로 비교가 성립한다.
    logger.info(f"  {'구성':<20}{'총편차':>8}{'건별F1평균':>12}{'빈 배열':>10}")
    summary = {}
    for k, v in have.items():
        f1 = sum(v["f1"]) / len(v["f1"])
        empty = sum(v["empty"]) / len(v["empty"])
        dev = total_deviation(v["pred_dist"], base["gold_dist"])
        summary[k] = {"f1": f1, "empty_rate": empty, "total_deviation": dev}
        logger.info(f"  {k:<20}{dev:>8}{f1 * 100:>11.1f}%{empty * 100:>9.1f}%")

    base_dev = summary[base_key]["total_deviation"]
    logger.info(f"  ----- 1차 판정: 총편차 Σ|pred−gold| (기준선 {base_dev}, 노이즈 바닥 2) -----")
    for k, s in summary.items():
        if k == base_key:
            continue
        d = s["total_deviation"] - base_dev
        if d <= -6:
            verdict = "블록 전문이 과잉 예측의 원인 — 채택"
        elif abs(d) <= 4:
            verdict = "차이 없음 — 더 짧은 쪽 채택(토큰 절감)"
        elif d >= 6:
            verdict = "블록이 값을 한다 — A 유지"
        else:
            verdict = "판정 구간 사이(5) — 보류"
        logger.info(f"  {k:<20} 총편차 {s['total_deviation']:>3} (Δ{d:+d})  →  {verdict}")

    logger.info("  ----- 2차(참고): 건별 F1 페어드 검정. 노이즈 바닥 δ=3.2%p라 단독 판정 불가 -----")
    tests = {}
    for k, v in have.items():
        if k == base_key:
            continue
        t = wilcoxon(base["f1"], v["f1"])
        tests[k] = t
        tag = "  ← 음성 대조군" if k.startswith("A'") else ""
        logger.info(f"  {k:<20} Δ{ (summary[k]['f1'] - summary[base_key]['f1']) * 100:+6.1f}%p  "
                    f"개선 {t.get('better', 0)} / 악화 {t.get('worse', 0)}  p={t['p_value']:.4f}{tag}")

    if "A' (전문, 재실행)" in summary:
        delta = abs(summary["A' (전문, 재실행)"]["f1"] - summary[base_key]["f1"]) * 100
        logger.info(f"  → 노이즈 바닥 δ = {delta:.1f}%p (같은 프롬프트 재실행 차이)")

    logger.info("  ----- 조별 예측 분포 (제6조 과잉 예측이 조문 전문 때문인가) -----")
    arts = sorted({a for v in have.values() for a in v["pred_dist"]},
                  key=lambda a: -base["gold_dist"].get(a, 0))
    header = "".join(f"{k.split()[0]:>9}" for k in have)
    logger.info(f"  {'조':<7}{'정답':>6}{header}")
    for a in arts:
        row = "".join(f"{v['pred_dist'].get(a, 0):>9}" for v in have.values())
        logger.info(f"  {a:<7}{base['gold_dist'].get(a, 0):>6}{row}")

    save_json({"summary": summary, "tests": tests,
               "pred_dist": {k: v["pred_dist"] for k, v in have.items()},
               "gold_dist": base["gold_dist"]}, OUT_PATH)
    logger.info(f"  저장: {OUT_PATH}")
    logger.info("  ※ 판정 기준은 backend/eval/prompt_block_ablation.md (실행 전 확정)")


if __name__ == "__main__":
    main()
