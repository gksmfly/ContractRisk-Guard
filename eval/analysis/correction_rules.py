# eval/analysis/correction_rules.py
"""교정형 FB-Check 오프라인 규칙 탐색, 0단계 (사전 등록: eval/paper_plan.md §7-29, 2026-10-17, GPU 0 · API 0 · 학습 0)

기존 forward·verify 결과(AI Hub 5,150건, Qwen)만으로 NOISE 레코드를 고치는 규칙 6개를 정답(전문가 ref)과 비교한다.
개발 절반에서 통과 기준(P1~P3)을 보고 net이 가장 큰 규칙 1개를 고른 뒤 평가 절반에서 같은 기준을 다시 본다.

    .venv/bin/python -m eval.analysis.correction_rules
"""

import json

import numpy as np

from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("correction_rules.log")
OUT = PROJECT_ROOT / "data/eval/analysis/correction_rules.json"
BOOT_N = 2000
RULES = ["R0", "RD", "R1", "R2", "R3", "R4", "R5", "R6"]
P1_RATIO, P2_DROP, P3_HARM = 0.5, 0.02, 0.20          # §7-29 통과 기준(고정)


def apply(rule: str, r: dict) -> set | None:
    """최종 라벨 집합. None = 레코드 제거."""
    F, V, A = set(r["forward"]), set(r.get("verify") or []), set(r.get("agreed") or [])
    if r["status"] == "CLEAN" or rule == "R0":
        return F
    rho = r.get("reason")
    if rule == "RD":
        return None
    if rule == "R1":
        return set()
    if rule == "R2":
        return V
    if rule == "R3":
        return A
    if rule == "R4":
        if rho == "snippet_not_found":
            return set()
        if rho == "label_mismatch":
            return V if V else set()
        return V                                      # article_mismatch
    if rule == "R5":
        return A if A else F
    if rule == "R6":
        return set() if rho == "snippet_not_found" else F
    raise ValueError(rule)


def counts(rows: list[dict], rule: str) -> dict:
    c = dict.fromkeys(("n", "kept", "fair", "fair_fp", "viol_inter", "viol_pred", "viol_ref", "exact",
                       "orig_ok", "harm", "orig_wrong", "fixed", "broken_total"), 0)
    for r in rows:
        ref, F = set(r["ref"]), set(r["forward"])
        L = apply(rule, r)
        c["n"] += 1
        orig_ok = bool(F) and F == ref and r["label"] == 1
        orig_wrong = F != ref
        c["orig_ok"] += orig_ok
        c["orig_wrong"] += orig_wrong
        if L is None:
            c["harm"] += orig_ok
            continue
        c["kept"] += 1
        c["exact"] += L == ref
        if r["label"] == 0:
            c["fair"] += 1
            c["fair_fp"] += bool(L)
        else:
            c["viol_inter"] += len(L & ref)
            c["viol_pred"] += len(L)
            c["viol_ref"] += len(ref)
        c["harm"] += orig_ok and L != ref
        c["fixed"] += orig_wrong and L == ref
        c["broken_total"] += orig_ok and L != ref
    return c


def metrics(c: dict) -> dict:
    d = c["viol_pred"] + c["viol_ref"]
    return {"fp_rate_pct": round(c["fair_fp"] / c["fair"] * 100, 2) if c["fair"] else float("nan"),
            "viol_micro_f1_pp": round(2 * c["viol_inter"] / d * 100, 2) if d else float("nan"),
            "exact_pct": round(c["exact"] / c["kept"] * 100, 2) if c["kept"] else float("nan"),
            "coverage_pct": round(c["kept"] / c["n"] * 100, 1),
            "harm_pct": round(c["harm"] / c["orig_ok"] * 100, 1) if c["orig_ok"] else float("nan"),
            "fix_pct": round(c["fixed"] / c["orig_wrong"] * 100, 1) if c["orig_wrong"] else float("nan"),
            "net": int(c["fixed"] - c["broken_total"])}


def boot_fp_diff(rows: list[dict], rule: str, rng) -> list[float]:
    """(규칙 − R0) FP율 차이의 파일 군집 부트스트랩 95% CI(%p)."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["key"].rsplit("#", 1)[0], []).append(r)
    keys = list(groups)
    per = {k: (counts(groups[k], rule), counts(groups[k], "R0")) for k in keys}
    diffs = []
    for _ in range(BOOT_N):
        pick = rng.integers(0, len(keys), len(keys))
        fp = fair = fp0 = fair0 = 0
        for i in pick:
            a, b = per[keys[i]]
            fp, fair, fp0, fair0 = fp + a["fair_fp"], fair + a["fair"], fp0 + b["fair_fp"], fair0 + b["fair"]
        if fair and fair0:
            diffs.append(fp / fair - fp0 / fair0)
    return [round(float(x) * 100, 2) for x in np.percentile(diffs, [2.5, 97.5])]


def evaluate(rows: list[dict], rng) -> dict:
    base = metrics(counts(rows, "R0"))
    out = {}
    for rule in RULES:
        m = metrics(counts(rows, rule))
        if rule != "R0":
            m["fp_diff_ci95_pp"] = boot_fp_diff(rows, rule, rng)
        p1 = (rule not in ("R0",) and m["fp_rate_pct"] <= base["fp_rate_pct"] * P1_RATIO and m["fp_diff_ci95_pp"][1] < 0)
        p2 = m["viol_micro_f1_pp"] >= base["viol_micro_f1_pp"] - P2_DROP * 100
        p3 = m["harm_pct"] <= P3_HARM * 100
        m["P1_fp_reduced"], m["P2_f1_kept"], m["P3_harm_low"] = bool(p1), bool(p2), bool(p3)
        m["pass"] = bool(p1 and p2 and p3)
        out[rule] = m
    return out


def main() -> None:
    rows = [r for r in load_jsonl(PROJECT_ROOT / "data/eval/label/fbcheck_aihub_raw.jsonl") if r["status"] != "ERROR"]
    train_files = set(json.loads((PROJECT_ROOT / "data/eval/model/aihub_split.json").read_text(encoding="utf-8"))["train_files"])
    dev = [r for r in rows if r["key"].rsplit("#", 1)[0] in train_files]
    ev = [r for r in rows if r["key"].rsplit("#", 1)[0] not in train_files]
    rng = np.random.default_rng(42)
    res = {"dev": evaluate(dev, rng), "eval": evaluate(ev, rng)}
    passed = [k for k, v in res["dev"].items() if v["pass"]]
    chosen = max(passed, key=lambda k: res["dev"][k]["net"]) if passed else None
    confirmed = bool(chosen and res["eval"][chosen]["pass"])
    rep = {"rule": "eval/paper_plan.md §7-29", "n": {"dev": len(dev), "eval": len(ev)},
           "criteria": {"P1": f"FP율 ≤ R0의 {P1_RATIO:.0%} 이고 차이 CI 상한 < 0", "P2": f"위반 조 micro-F1 ≥ R0 − {P2_DROP * 100:.0f}%p",
                        "P3": f"harm ≤ {P3_HARM:.0%}"},
           "dev": res["dev"], "eval": res["eval"], "dev_passed": passed, "chosen_on_dev": chosen, "confirmed_on_eval": confirmed,
           "verdict": (f"1단계 후보: {chosen} (개발 선택·평가 확인 모두 통과)" if confirmed
                       else "통과 규칙 없음 — 현재 신호(재판정)로는 안전한 교정 불가, 교정형은 향후 과제" if not chosen
                       else f"{chosen}은 개발에서 통과했으나 평가에서 확인되지 않음 — 1단계 보류")}
    save_json(rep, OUT)
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
