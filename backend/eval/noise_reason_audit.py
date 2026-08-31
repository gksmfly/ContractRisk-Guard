# backend/eval/noise_reason_audit.py
"""
NOISE 사유별 감사 — **버려진 레코드가 정말 버릴 값이었는가.**

## 왜 필요한가

FTC NOISE의 61%가 `label_mismatch`(forward risk ≠ verify risk)다. 그런데 **우리는 이제
risk_level을 학습하지 않는다** — P1에서 risk 헤드를 일부러 뺐고(gold가 미정이라),
학습 타깃은 조 multi-label이다.

즉 **쓰지도 않을 라벨의 불일치로 FTC의 26%를 버리고 있다.**

정당할 수도 있다 — risk 불일치가 "두 번의 판단이 조항을 다르게 이해했다"는 일반적
신호일 수 있으니까. 하지만 측정된 적이 없다. 그리고 공짜로 잰다:

    label_mismatch로 버려진 건의 forward_articles를 FTC 근거_법령과 대조
      CLEAN 건과 조 정확도가 비슷하면  →  risk 게이트가 멀쩡한 조 라벨을 버리고 있다
      뚜렷이 낮으면                    →  게이트가 값을 한다

결과에 따라 FTC 학습 데이터가 26% 늘어날 수 있다.

## 논문 정의와의 관계

논문(KAICTS 2025)의 판정은 `L == L'`이고 L은 리스크 라벨이므로, **현재 구현이 논문에
충실한 것은 맞다.** 다만 taxonomy를 조 multi-label로 바꾸면서 "L이 무엇인가"가 모호해졌고,
구현은 `risk 일치 AND (Low가 아니면 조 교집합 비어있지 않음)`으로 해석했다.
그 해석이 옳은지를 이 측정이 가린다.

## 이 스크립트는 판정하지 않는다

수치만 낸다. 게이트를 바꿀지는 사람이 정한다 — 결과를 보고 게이트를 푸는 것은
"결과 보고 기준 바꾸기"이므로, 바꾼다면 그 사실과 근거를 문서에 남겨야 한다.

실행:
    .venv/bin/python -m backend.eval.noise_reason_audit
"""

import argparse
import random
import json
from collections import defaultdict
from pathlib import Path

from backend.labeling.articles import normalize
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("noise_reason_audit.log")

RESULTS_PATH = PROJECT_ROOT / "data/fb_check/fb_check_results.jsonl"
FTC_PATH     = PROJECT_ROOT / "data/raw/ftc_cases/ftc_cases_parsed.json"
OUT_PATH     = PROJECT_ROOT / "data/eval/noise_reason_audit.json"


def _read(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return list({r.get("chunk_id"): r for r in rows}.values())


def _gold_by_doc() -> dict[str, set[str]]:
    cases = json.loads(FTC_PATH.read_text(encoding="utf-8"))["사례"]
    out = {}
    for c in cases:
        doc = str((c.get("셀_데이터") or {}).get("사건번호") or c.get("사건명", ""))
        arts = {a for g in (c.get("근거_법령") or []) if (a := normalize(str(g)))}
        if doc and arts:
            out[f"ftc_case:{doc}"] = arts
    return out


def _doc_of(chunk_id: str) -> str:
    parts = str(chunk_id).split(":")
    return ":".join(parts[:2]) if len(parts) >= 3 else str(chunk_id)


def _f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    inter = len(pred & gold)
    p = inter / len(pred) if pred else 0.0
    r = inter / len(gold) if gold else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0




def score_group(rows: list[dict], gold: dict[str, set[str]]) -> dict:
    """forward_articles를 사건 gold와 대조.

    **두 벌을 낸다.** 전체(all)와 빈 배열 제외(nonempty). 둘은 틀린 것과 맞는 것이 아니라
    **서로 다른 질문에 답한다** — 어느 쪽도 버리지 말고 둘 다 읽을 것:

        전체(all)        "이 레코드를 학습셋에 넣으면 평균이 나빠지나?"
                         CLEAN 37.7% vs label_mismatch 38.4% → 아니오 (중립)

        nonempty         "이 레코드가 하는 말이 CLEAN만큼 믿을 만한가?"
                         CLEAN 49.4% vs label_mismatch 39.2% → 아니오 (-10.2%p)

    두 수치가 다른 이유는 구성이다 — CLEAN에는 "위반 없음"으로 빈 배열을 낸 건이 24%
    섞여 있고 FTC 사건 gold는 비어있지 않으므로 그 건들은 전부 F1 0점인 반면,
    `label_mismatch`는 risk 불일치가 곧 Low가 아니라는 뜻이라 빈 배열이 2.5%뿐이다.

    **게이트 유지 여부는 nonempty로 판단한다** — 라벨의 신뢰도를 묻는 게이트이므로
    "믿을 만한가"가 맞는 질문이다. 다만 전체 수치가 중립이라는 사실은 **학습 데이터가
    부족해지면 이 게이트를 다시 저울질할 여지**를 뜻한다. 지우지 말 것.
    """
    pairs = [(set(r.get("forward_articles") or []), gold[_doc_of(r.get("chunk_id", ""))])
             for r in rows if _doc_of(r.get("chunk_id", "")) in gold]
    if not pairs:
        return {"n": len(rows), "n_scored": 0}

    def block(ps: list[tuple[set, set]]) -> dict:
        if not ps:
            return {"n_scored": 0}
        return {
            "n_scored": len(ps),
            "f1": sum(_f1(p, g) for p, g in ps) / len(ps),
            "hit_any": sum(1 for p, g in ps if p & g) / len(ps),
            "precision": sum(len(p & g) / len(p) for p, g in ps if p) / max(1, sum(1 for p, _ in ps if p)),
            "recall": sum(len(p & g) / len(g) for p, g in ps if g) / max(1, sum(1 for _, g in ps if g)),
            "scores": [_f1(p, g) for p, g in ps],
        }

    return {
        "n": len(rows), "n_scored": len(pairs),
        "empty_rate": sum(1 for p, _ in pairs if not p) / len(pairs),
        "all": block(pairs),
        "nonempty": block([(p, g) for p, g in pairs if p]),
    }


def _significance(a: list[float], b: list[float]) -> dict:
    """CLEAN(a)이 버려진 집단(b)보다 나은가. Mann-Whitney + 차이 부트스트랩 CI."""
    if len(a) < 10 or len(b) < 10:
        return {"note": f"표본 부족 (n={len(a)}, {len(b)})"}
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(a, b, alternative="greater")
    rng = random.Random(42)
    d = sorted((sum(rng.choices(a, k=len(a))) / len(a)) - (sum(rng.choices(b, k=len(b))) / len(b))
               for _ in range(5000))
    return {"p_value": float(p), "diff_pp": (sum(a) / len(a) - sum(b) / len(b)) * 100,
            "ci95_pp": [d[125] * 100, d[4875] * 100]}


def main() -> None:
    ap = argparse.ArgumentParser(description="NOISE 사유별 감사")
    ap.add_argument("--results", default=str(RESULTS_PATH))
    ap.add_argument("--source", default="ftc_case", help="gold가 있는 출처만 의미가 있다")
    a = ap.parse_args()

    rows = [r for r in _read(Path(a.results)) if r.get("source") == a.source]
    gold = _gold_by_doc()
    logger.info(f"========== NOISE 사유 감사 ({a.source}, {len(rows)}건) ==========")

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("status") == "CLEAN":
            groups["CLEAN"].append(r)
        elif r.get("status") == "NOISE":
            groups[r.get("noise_reason", "?").split(":")[0]].append(r)

    report = {}
    logger.info("  ----- 빈 배열 제외 (판단 기준) -----")
    logger.info(f"  {'집단':<20}{'건수':>6}{'채점':>6}{'조 F1':>9}{'hit@any':>10}{'정밀도':>9}{'재현율':>9}{'빈배열':>9}")
    for name in ["CLEAN"] + sorted(k for k in groups if k != "CLEAN"):
        s = score_group(groups[name], gold)
        report[name] = s
        if not s.get("n_scored") or not s["nonempty"].get("n_scored"):
            logger.info(f"  {name:<20}{s['n']:>6}{s.get('n_scored', 0):>6}   (채점 불가)")
            continue
        ne = s["nonempty"]
        logger.info(f"  {name:<20}{s['n']:>6}{ne['n_scored']:>6}{ne['f1'] * 100:>8.1f}%"
                    f"{ne['hit_any'] * 100:>9.1f}%{ne['precision'] * 100:>8.1f}%"
                    f"{ne['recall'] * 100:>8.1f}%{s['empty_rate'] * 100:>8.1f}%")

    logger.info("  ----- 참고: 전체 (빈 배열 포함 — 편향됨) -----")
    for name, s in report.items():
        if s.get("n_scored"):
            logger.info(f"  {name:<20}{s['all']['n_scored']:>6}{s['all']['f1'] * 100:>8.1f}%"
                        f"{s['all']['hit_any'] * 100:>9.1f}%")
    logger.info("     CLEAN에만 빈 배열이 몰려 있어 버려진 쪽이 유리하게 나온다. 판단에 쓰지 말 것")

    cl = report.get("CLEAN", {})
    if cl.get("n_scored"):
        for name in sorted(k for k in report if k != "CLEAN"):
            s = report[name]
            if not s.get("n_scored") or not s["nonempty"].get("n_scored"):
                continue
            sig = _significance(cl["nonempty"]["scores"], s["nonempty"]["scores"])
            report[name]["vs_clean"] = sig
            if "note" in sig:
                logger.info(f"  → CLEAN vs {name}: {sig['note']}")
                continue
            ci = sig["ci95_pp"]
            logger.info(f"  → CLEAN vs {name:<18} 조 F1 {sig['diff_pp']:+5.1f}%p "
                        f"(95% CI {ci[0]:+.1f} ~ {ci[1]:+.1f}, p={sig['p_value']:.4f})")
            if name == "label_mismatch":
                if sig["p_value"] < 0.05 and ci[0] > 0:
                    logger.info("     **게이트가 값을 한다** — risk 불일치 건은 조 라벨도 실제로 나쁘다. "
                                "risk_level을 학습하지 않더라도 버리는 게 맞다")
                else:
                    logger.warning("     **risk 게이트가 멀쩡한 조 라벨을 버리고 있을 수 있다.** "
                                   "risk_level은 학습 타깃이 아니다(P1에서 헤드를 뺐다) — "
                                   f"{s['n']}건을 살리면 {a.source} 학습 데이터가 늘어난다")

    for s in report.values():                       # scores 배열은 저장하지 않는다 (부피만 큼)
        for k in ("all", "nonempty"):
            if isinstance(s.get(k), dict):
                s[k].pop("scores", None)
    save_json({"source": a.source, "groups": report,
               "판단_기준": "nonempty (빈 배열 제외). 전체 수치는 CLEAN 쪽에 빈 배열이 몰려 편향됨",
               "note": "판정하지 않는다. 게이트 변경은 사람이 정하고, 바꾼다면 근거를 문서에 남길 것"},
              OUT_PATH)
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
