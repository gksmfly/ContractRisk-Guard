# eval/requirement_post_analysis.py
"""본 실행 이후의 **사후·서술 분석** (사전 등록 §10). API 0. 판정에 쓰지 않는다.

  1) B1과 같은 개수를 무작위로 제거한 B2 — "FB-Check는 무작위 제거와 구분되지 않는다" 주장용
     (조항 단위: B1이 버린 조항 수만큼 / 라벨 단위: B1이 버린 라벨 수만큼, 제6조·제7~14조 별도)
  2) V1이 유지했지만 틀린 라벨의 오류 분석 — 어떤 조 혼동이 남는가 (프롬프트·규칙은 고치지 않는다)

    .venv/bin/python -m eval.requirement_post_analysis
"""

import json
from collections import Counter

import numpy as np

from backend.utils import save_json
from eval.requirement_main_run import (ART6, OUT_DIR, _b2, _boot_diff, _cache, _counts, _decisions, _f1, _key,
                                       load_cards, load_clauses)

DRAWS = 10_000


def _clause_level_b2(clauses, n_remove: int, drop6: bool) -> tuple[np.ndarray, np.ndarray]:
    """조항 단위 무작위 제거(B1이 버린 조항 수만큼). 조항별 기대 (TP,FP,FN)와 F1 분포."""
    n = len(clauses)
    exp, f1s = np.zeros((n, 3)), []
    full = [set(c["fwd"]) for c in clauses]
    for seed in range(DRAWS):
        gone = set(np.random.default_rng(seed).choice(n, n_remove, replace=False).tolist())
        cnt = _counts(clauses, [set() if i in gone else full[i] for i in range(n)], drop6)
        exp += cnt / DRAWS
        f1s.append(_f1(cnt))
    return exp, np.array(f1s)


def b1_matched() -> dict:
    clauses = load_clauses()
    b1 = [set(c["fwd"]) if c["b1_pass"] else set() for c in clauses]
    n_gone = sum(not c["b1_pass"] for c in clauses)
    out = {"B1이_버린_조항": n_gone, "조항수": len(clauses)}
    label = _b2(clauses, b1)                      # 라벨 단위(사전 등록 B2와 같은 정의)
    for d6, scope in ((True, "제6조제외"), (False, "전체")):
        c1 = _counts(clauses, b1, d6)
        f1 = _f1(c1) * 100
        exp_c, f1s_c = _clause_level_b2(clauses, n_gone, d6)
        exp_l, f1s_l = label[d6]
        out[scope] = {
            "B1_f1": f1,
            "조항단위_B2": {"f1_mean": float(f1s_c.mean() * 100), "f1_95": [float(x * 100) for x in np.percentile(f1s_c, [2.5, 97.5])],
                            "B1_백분위": float((f1s_c * 100 < f1).mean() * 100), "B1-B2": _boot_diff(c1, exp_c)},
            "라벨단위_B2": {"f1_mean": float(f1s_l.mean() * 100), "f1_95": [float(x * 100) for x in np.percentile(f1s_l, [2.5, 97.5])],
                            "B1_백분위": float((f1s_l * 100 < f1).mean() * 100), "B1-B2": _boot_diff(c1, exp_l)}}
    return out


def v1_error_analysis() -> dict:
    clauses, cards = load_clauses(), load_cards()
    kept_sets, _ = _decisions("V1", clauses, cards)
    cache = _cache("V1")
    rows = []
    for c, ks in zip(clauses, kept_sets):
        for a in c["fwd"]:
            rec = cache[_key("V1", c["doc"], a)]
            from eval.requirement_verifier import parse
            p = parse("V1", rec["raw"], a, cards)
            verdicts = [i["verdict"] for i in p["items"]]
            reason = "충족" if "충족" in verdicts else ("판단불가만" if "판단불가" in verdicts else "전부불충족")
            rows.append({"doc": c["doc"], "art": a, "kept": a in ks, "correct": a in c["ref"], "ref": sorted(c["ref"]),
                         "why": reason, "n_ref": len(c["ref"])})
    scope = [r for r in rows if r["art"] != ART6]
    out = {"제7~14조_라벨": len(scope)}
    cell = Counter((r["kept"], r["correct"]) for r in scope)
    out["혼동(유지/정답)"] = {"유지·정답(TP)": cell[(True, True)], "유지·오답(FP)": cell[(True, False)],
                           "제거·정답(손실된 TP)": cell[(False, True)], "제거·오답(걸러낸 FP)": cell[(False, False)]}
    kept_fp = [r for r in scope if r["kept"] and not r["correct"]]
    out["유지된_오답의_유지_사유"] = dict(Counter(r["why"] for r in kept_fp))
    out["유지된_정답의_유지_사유"] = dict(Counter(r["why"] for r in scope if r["kept"] and r["correct"]))
    out["유지된_오답_예측조별"] = dict(Counter(r["art"] for r in kept_fp).most_common())
    conf = Counter()
    for r in kept_fp:
        for g in r["ref"]:
            conf[(r["art"], g)] += 1 / len(r["ref"])           # 정답이 여럿이면 균등 분배
    out["유지된_오답의_혼동쌍(예측→정답)"] = [[f"{a}→{g}", round(v, 1)] for (a, g), v in conf.most_common(10)]
    lost = [r for r in scope if (not r["kept"]) and r["correct"]]
    out["손실된_정답_조별"] = dict(Counter(r["art"] for r in lost).most_common())
    per = {}
    for a in sorted({r["art"] for r in scope}, key=lambda x: int(x[1:-1])):
        sub = [r for r in scope if r["art"] == a]
        b0_p = sum(r["correct"] for r in sub) / len(sub)
        kept = [r for r in sub if r["kept"]]
        per[a] = {"B0_정밀도": round(b0_p * 100, 1), "V1_정밀도": round(sum(r["correct"] for r in kept) / len(kept) * 100, 1) if kept else None,
                  "forward_라벨": len(sub), "V1_유지": len(kept)}
    out["조별_정밀도"] = per
    return out


if __name__ == "__main__":
    res = {"설명": "사후·서술 분석 — 판정에 쓰지 않는다", "B1_기준_B2": b1_matched(), "V1_오류분석": v1_error_analysis()}
    save_json(res, OUT_DIR / "post_analysis.json")
    print(json.dumps(res, ensure_ascii=False, indent=2))
