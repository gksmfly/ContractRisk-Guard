# eval/analysis/typed_checks.py
"""오류 유형별 검증(A)·신뢰도 단계(B)·누락 라벨(C)의 타당성을 기존 행으로 확인 (보고용, 사전 등록 아님, 2026-10-13, 학습·API 0)

AI Hub 5,150건(Qwen 검증, 조항 단위 전문가 정답)의 라벨 하나하나(조 단위)를 정답과 맞춰 본다.
 A. 유형을 나눠 검증하면 기존 FB-Check가 버린 정답을 얼마나 살리고 정밀도는 얼마나 유지하나
 B. FB-Check 결과를 단계(신뢰도)로 나누면 단계가 올라갈수록 라벨 정확도가 실제로 올라가나
 C. 누락 라벨: 이미 §7-19·§7-20에서 두 번 잼 — 여기서는 규모만 다시 센다

    .venv/bin/python -m eval.analysis.typed_checks
"""

import json
from collections import Counter

import numpy as np

from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json
from eval.label.label_ablation import _doc_of, _labels, _ref_by_doc

logger = load_logger("typed_checks.log")
OUT = PROJECT_ROOT / "data/eval/analysis/typed_checks.json"
rng = np.random.default_rng(42)


def tier(r) -> str:
    F, V = set(r["forward"]), set(r.get("verify") or [])
    if r.get("reason") == "snippet_not_found":
        return "근거 문구가 원문에 없음"
    if r["status"] == "ERROR" or r.get("verify") is None:
        return "검증 실패"
    art = "조 전부 일치" if F <= V else ("조 일부 일치" if F & V else "조 불일치")
    grade = "등급 같음" if r["f_label"] == r.get("v_label") else "등급 다름"
    return f"{art} · {grade}"


def ci(groups, fn, n=2000):
    idx = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    cl = [np.array(v) for v in idx.values()]
    b = [fn(np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])) for _ in range(n)]
    return [round(float(x) * 100, 1) for x in np.percentile(b, [2.5, 97.5])]


def aihub() -> dict:
    rows = [r for r in load_jsonl(PROJECT_ROOT / "data/eval/label/fbcheck_aihub_raw.jsonl") if r["forward"]]
    # 라벨 단위 표: 한 레코드의 조 라벨 하나가 한 행
    L = []
    for i, r in enumerate(rows):
        ref, V = set(r["ref"]), set(r.get("verify") or [])
        for a in r["forward"]:
            L.append({"rec": i, "file": r["key"].rsplit("#", 1)[0], "tier": tier(r), "ok": a in ref, "a": a,
                      "inV": a in V, "grade_same": r["f_label"] == r.get("v_label"), "snippet": r.get("reason") != "snippet_not_found",
                      "clean": r["status"] == "CLEAN", "viol": r["label"] == 1})
    ok = np.array([x["ok"] for x in L])
    out = {"n_records_labeled": len(rows), "n_labels": len(L), "label_precision_all_pct": round(float(ok.mean()) * 100, 1)}

    # B. 단계별 라벨 정확도
    tiers = {}
    for t in sorted({x["tier"] for x in L}):
        m = np.array([x["tier"] == t for x in L])
        tiers[t] = {"n_labels": int(m.sum()), "precision_pct": round(float(ok[m].mean()) * 100, 1),
                    "ci95": ci([x["file"] for x, k in zip(L, m) if k], lambda i, mk=m: ok[mk][i].mean())}
    out["B_tier_label_precision"] = tiers

    # A. 정책 비교 (라벨 단위): 남긴 수·정밀도·남긴 정답 수
    def pol(keep: np.ndarray) -> dict:
        return {"kept": int(keep.sum()), "precision_pct": round(float(ok[keep].mean()) * 100, 1),
                "correct_kept": int((ok & keep).sum()), "correct_kept_pct_of_all_correct": round(float((ok & keep).sum() / ok.sum()) * 100, 1)}
    clean = np.array([x["clean"] for x in L])
    inV = np.array([x["inV"] for x in L])
    gsame = np.array([x["grade_same"] for x in L])
    sn = np.array([x["snippet"] for x in L])
    out["A_policies"] = {
        "전부 유지(필터 없음)": pol(np.ones(len(L), bool)),
        "FB-Check CLEAN": pol(clean),
        "유형 분리: 이 조를 검증도 짚었으면 유지(등급·근거 무시)": pol(inV),
        "유형 분리: 이 조를 검증도 짚음 + 근거 문구 있음(등급만 무시)": pol(inV & sn),
        "무작위로 같은 수(FB-Check CLEAN과 동일)": pol(rng.permutation(np.arange(len(L)) < clean.sum())),
    }
    # FB-Check가 버린 라벨 중 사유별로 정답 비율 (살릴 여지)
    rem = {}
    for reason in ("label_mismatch", "article_mismatch", "snippet_not_found"):
        m = np.array([(not x["clean"]) and (rows[x["rec"]].get("reason") == reason) for x in L])
        if m.sum():
            rem[reason] = {"n_labels": int(m.sum()), "correct_pct": round(float(ok[m].mean()) * 100, 1),
                           "and_verify_agrees_on_article_pct": round(float((ok & inV)[m].sum() / max(m.sum(), 1)) * 100, 1)}
    out["A_removed_by_reason"] = rem
    # 등급만 다르고 조가 맞는 라벨 (A의 대표 예: 제14조)
    only_grade = (~gsame) & inV
    out["A_grade_only_mismatch"] = {"n_labels": int(only_grade.sum()), "precision_pct": round(float(ok[only_grade].mean()) * 100, 1),
                                    "removed_by_fbcheck_pct": round(float((~clean[only_grade]).mean()) * 100, 1)}
    # C. 누락 라벨 규모
    allrows = load_jsonl(PROJECT_ROOT / "data/eval/label/fbcheck_aihub_raw.jsonl")
    viol = [r for r in allrows if r["label"] == 1]
    miss = [r for r in viol if not r["forward"]]
    out["C_missed_labels"] = {"violations": len(viol), "empty_label": len(miss), "empty_pct": round(len(miss) / len(viol) * 100, 1),
                              "passed_as_clean": sum(r["status"] == "CLEAN" for r in miss),
                              "verify_flagged_again": sum(bool(r.get("verify")) for r in miss),
                              "empty_by_gold_article": dict(Counter(a for r in miss for a in set(r["ref"])).most_common())}
    return out


def gpt() -> dict:
    """GPT-4o 운영 데이터: 사건 단위 근거 법령을 약한 정답으로 쓴 같은 단계표(해상도가 낮아 보조)."""
    ref = _ref_by_doc()
    rows = [r for r in load_jsonl(PROJECT_ROOT / "data/fb_check/clean.jsonl") + load_jsonl(PROJECT_ROOT / "data/fb_check/noise.jsonl")
            if r.get("source") == "ftc_case" and _doc_of(r.get("chunk_id", "")) in ref and r.get("forward_articles")]
    L = []
    for r in rows:
        F, V = _labels(r.get("forward_articles")), _labels(r.get("verify_articles"))
        rr = ref[_doc_of(r["chunk_id"])]
        if not r.get("snippet_exists"):
            t = "근거 문구가 원문에 없음"
        else:
            art = "조 전부 일치" if F <= V else ("조 일부 일치" if F & V else "조 불일치")
            t = f"{art} · {'등급 같음' if r.get('forward_label') == r.get('verify_label') else '등급 다름'}"
        for a in F:
            L.append((t, a in rr, _doc_of(r["chunk_id"])))
    out = {}
    ok = np.array([x[1] for x in L])
    for t in sorted({x[0] for x in L}):
        m = np.array([x[0] == t for x in L])
        out[t] = {"n_labels": int(m.sum()), "precision_pct": round(float(ok[m].mean()) * 100, 1)}
    return {"n_labels": len(L), "precision_all_pct": round(float(ok.mean()) * 100, 1), "tiers": out,
            "주의": "사건 단위 정답(조항마다 정확하지 않음)이라 절대값은 해석하지 않고 단계 순서만 본다"}


def main() -> None:
    rep = {"rule": "보고용 타당성 확인 (사전 등록 아님)", "AI Hub(Qwen 검증, 조항 단위 전문가 정답)": aihub(), "GPT-4o 운영 데이터(사건 단위 약한 정답)": gpt()}
    save_json(rep, OUT)
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
