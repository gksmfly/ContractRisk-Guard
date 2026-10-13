# eval/analysis/miss_and_filter.py
"""① 놓친 위반 조항 분류 ② 조항별 FB-Check 통과율, 특히 제6조 (보고용, 사전 등록 아님, 2026-10-13, 학습·API 0)

① AI Hub 평가 절반 위반 조항(단일 조 정답)에서 모델이 조를 하나도 지목하지 못한 것을 특성별로 나눈다.
   - 근접 실패(임계값 근처)인지 신호 없음인지: r = max(p/thr)
   - 조각형(조·호 번호로 시작, 문장이 끝나지 않음)인지, 길이
② 조항별 필터 통과율: AI Hub 5,150건(Qwen 검증)의 정답 조별 · GPT-4o 운영 데이터(forward 조별) · 학습 데이터에서의 조별 비중 변화

    .venv/bin/python -m eval.analysis.miss_and_filter
"""

import json
import re
from collections import Counter

import numpy as np

from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json
from eval.label.fbcheck_aihub import population
from eval.label.label_ablation import _doc_of
from eval.model.aihub_eval import CACHE, COMMON, testset

logger = load_logger("miss_and_filter.log")
OUT = PROJECT_ROOT / "data/eval/analysis/miss_and_filter.json"
rng = np.random.default_rng(42)
START = re.compile(r"^\s*(제\s*\d+\s*[호항목]|\d+\s*[\.\)]|[①-⑩]|\(\d+\)|[가-하]\s*[\.\)])")
END = re.compile(r"(다|요|음|함|됨|임|것|니다)\s*[\.\)]?\s*$")


def ci_diff(groups, fn, n=3000):
    idx = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    cl = [np.array(v) for v in idx.values()]
    b = [fn(np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])) for _ in range(n)]
    return [round(float(x) * 100, 1) for x in np.percentile(b, [2.5, 97.5])]


def part1() -> dict:
    items = testset()
    train_files = set(json.loads((PROJECT_ROOT / "data/eval/model/aihub_split.json").read_text(encoding="utf-8"))["train_files"])
    sel = [i for i, it in enumerate(items) if it["label"] == 1 and len(it["articles"]) == 1 and it["articles"][0] in COMMON
           and it["group"] not in train_files]
    out = {"n": len(sel), "models": {}}
    for name in ("C", "CNA"):
        zs = [np.load(CACHE / f"{name}_s{s}.npz") for s in (42, 7, 123)]
        ptd = np.mean([(z["P"][sel] >= z["thr"]).any(1) for z in zs], axis=0) > 0.5
        r = np.median([(z["P"][sel] / z["thr"]).max(1) for z in zs], axis=0)   # 평균은 한 시드의 큰 값에 끌려 근접 실패가 부풀려진다
        miss = ~ptd
        texts = [items[i]["text"] for i in sel]
        arts = np.array([items[i]["articles"][0] for i in sel])
        L = np.array([len(t) for t in texts])
        frag = np.array([bool(START.match(t)) for t in texts])
        incomplete = np.array([not END.search(t.strip()) for t in texts])
        groups = [items[i]["group"] for i in sel]
        m = {"missed": int(miss.sum()), "missed_pct": round(float(miss.mean()) * 100, 1),
             "near_miss(r 0.5~1)_pct_of_missed": round(float(((r >= 0.5) & miss).sum() / max(miss.sum(), 1)) * 100, 1),
             "far_miss(r<0.2)_pct_of_missed": round(float(((r < 0.2) & miss).sum() / max(miss.sum(), 1)) * 100, 1),
             "median_len_missed": int(np.median(L[miss])), "median_len_pointed": int(np.median(L[~miss])),
             "by_len_quartile_missed_pct": {}, "by_feature": {}, "by_article": {}}
        qs = np.quantile(L, [0.25, 0.5, 0.75])
        for lab, mask in (("<=Q1", L <= qs[0]), ("Q1~Q2", (L > qs[0]) & (L <= qs[1])), ("Q2~Q3", (L > qs[1]) & (L <= qs[2])), (">Q3", L > qs[2])):
            m["by_len_quartile_missed_pct"][f"{lab}(~{int(L[mask].mean())}자)"] = round(float(miss[mask].mean()) * 100, 1)
        for lab, mask in (("번호·호로 시작(조각형)", frag), ("문장이 끝나지 않음", incomplete), ("완결 문장", ~incomplete & ~frag)):
            m["by_feature"][lab] = {"n": int(mask.sum()), "missed_pct": round(float(miss[mask].mean()) * 100, 1)}
        for a in COMMON:
            k = arts == a
            m["by_article"][a] = {"n": int(k.sum()), "missed_pct": round(float(miss[k].mean()) * 100, 1),
                                  "near_miss_pct_of_missed": round(float(((r >= 0.5) & miss & k).sum() / max((miss & k).sum(), 1)) * 100, 1)}
        d = lambda i: miss[i][frag[i]].mean() - miss[i][~frag[i]].mean()  # noqa: E731
        m["fragment_minus_other_missed_pp_ci95"] = ci_diff(groups, d)
        out["models"][name] = m
    return out


def part2() -> dict:
    out = {}
    # (a) AI Hub, Qwen 검증: 정답 조별 통과율 (라벨이 붙은 위반 조항)
    pop = {r["key"]: r for r in population()}
    rows = [r for r in load_jsonl(PROJECT_ROOT / "data/eval/label/fbcheck_aihub_raw.jsonl") if r["label"] == 1 and r["forward"]]
    byart = {}
    for r in rows:
        for a in set(pop[r["key"]]["articles"]):
            byart.setdefault(a, []).append(r)
    ai = {}
    for a in COMMON:
        rs = byart.get(a, [])
        if not rs:
            continue
        reasons = Counter((x.get("reason") or "CLEAN") for x in rs)
        ai[a] = {"n_labeled": len(rs), "pass_pct": round(float(np.mean([x["status"] == "CLEAN" for x in rs])) * 100, 1),
                 "label_mismatch_pct": round(reasons["label_mismatch"] / len(rs) * 100, 1),
                 "article_mismatch_pct": round(reasons["article_mismatch"] / len(rs) * 100, 1),
                 "snippet_not_found_pct": round(reasons["snippet_not_found"] / len(rs) * 100, 1)}
    # 제6조 대 나머지 (한 레코드의 정답 조에 제6조가 있나) — 파일 군집 부트스트랩
    is6 = np.array(["제6조" in pop[r["key"]]["articles"] for r in rows])
    ok = np.array([r["status"] == "CLEAN" for r in rows])
    grp = [r["key"].rsplit("#", 1)[0] for r in rows]
    out["aihub_by_gold_article"] = ai
    out["aihub_6_vs_other_pass_pp"] = {"제6조": round(float(ok[is6].mean()) * 100, 1), "그 외": round(float(ok[~is6].mean()) * 100, 1),
                                        "diff_ci95": ci_diff(grp, lambda i: ok[i][is6[i]].mean() - ok[i][~is6[i]].mean())}
    # (b) GPT-4o 운영 데이터: forward가 붙인 조별 통과율 (한 레코드가 여러 조에 속할 수 있다)
    recs = [r for r in load_jsonl(PROJECT_ROOT / "data/fb_check/clean.jsonl") + load_jsonl(PROJECT_ROOT / "data/fb_check/noise.jsonl")
            if r.get("forward_articles")]
    gp = {}
    for r in recs:
        for a in set(r["forward_articles"]):
            gp.setdefault(a, []).append(r)
    gpt = {}
    for a in COMMON:
        rs = gp.get(a, [])
        if rs:
            reasons = Counter((str(x.get("noise_reason") or "CLEAN")).split(":")[0] for x in rs)
            gpt[a] = {"n_labeled": len(rs), "pass_pct": round(float(np.mean([x.get("status") == "CLEAN" for x in rs])) * 100, 1),
                      "label_mismatch_pct": round(reasons["label_mismatch"] / len(rs) * 100, 1),
                      "article_mismatch_pct": round(reasons["article_mismatch"] / len(rs) * 100, 1)}
    out["gpt4o_by_forward_article"] = gpt
    f6 = np.array(["제6조" in r["forward_articles"] for r in recs])
    okg = np.array([r.get("status") == "CLEAN" for r in recs])
    gd = [_doc_of(r["chunk_id"]) for r in recs]
    out["gpt4o_6_vs_other_pass_pp"] = {"제6조 포함": round(float(okg[f6].mean()) * 100, 1), "그 외": round(float(okg[~f6].mean()) * 100, 1),
                                        "diff_ci95": ci_diff(gd, lambda i: okg[i][f6[i]].mean() - okg[i][~f6[i]].mean())}
    mid = np.array([bool({"제7조", "제8조", "제9조", "제10조"} & set(r["forward_articles"])) and "제6조" not in r["forward_articles"] for r in recs])
    out["gpt4o_6_vs_7to10_pass_pp"] = {"제6조 포함": round(float(okg[f6].mean()) * 100, 1), "제7~10조(제6조 없음)": round(float(okg[mid].mean()) * 100, 1),
                                        "diff_ci95": ci_diff([g for g, a_, b_ in zip(gd, f6, mid) if a_ or b_],
                                                             lambda i, a_=f6[f6 | mid], b_=okg[f6 | mid]: b_[i][a_[i]].mean() - b_[i][~a_[i]].mean())}
    # 제6조가 '단독'인지 '덧붙은' 것인지
    only6 = np.array([r["forward_articles"] == ["제6조"] for r in recs])
    plus6 = f6 & ~only6
    out["gpt4o_6_only_vs_6_plus_pass_pct"] = {"제6조 단독": round(float(okg[only6].mean()) * 100, 1), "n_단독": int(only6.sum()),
                                              "제6조 + 다른 조": round(float(okg[plus6].mean()) * 100, 1), "n_덧붙음": int(plus6.sum())}
    # (c) 학습 데이터 조별 비중: 필터 전(A) 대 필터 후(CLEAN)
    A = load_jsonl(PROJECT_ROOT / "data/eval/model/e8_arms/arm_A_forward_all.jsonl")
    cA = Counter(a for r in A for a in set(r.get("forward_articles") or []))
    cC = Counter(a for r in A if r.get("status") == "CLEAN" for a in set(r.get("forward_articles") or []))
    tA, tC = sum(cA.values()), sum(cC.values())
    out["train_article_share"] = {a: {"before": cA[a], "after": cC[a], "kept_pct": round(cC[a] / max(cA[a], 1) * 100, 1),
                                      "share_before_pct": round(cA[a] / tA * 100, 1), "share_after_pct": round(cC[a] / tC * 100, 1)}
                                  for a in COMMON}
    return out


def part3() -> dict:
    """필터 전(A)·무작위(B)·FB-Check(D) 모델이 조별로 놓치는 비율 — 기본 임계값, AI Hub 평가 절반 위반 조항."""
    items = testset()
    train_files = set(json.loads((PROJECT_ROOT / "data/eval/model/aihub_split.json").read_text(encoding="utf-8"))["train_files"])
    sel = [i for i, it in enumerate(items) if it["label"] == 1 and len(it["articles"]) == 1 and it["articles"][0] in COMMON
           and it["group"] not in train_files]
    arts = np.array([items[i]["articles"][0] for i in sel])
    groups = [items[i]["group"] for i in sel]
    seeds10 = [42, 7, 123, 11, 22, 33, 44, 55, 66, 77]
    mats = {}
    for arm in ("A", "B", "D"):
        rows = []
        for sd in seeds10:
            f = CACHE / f"{arm}_s{sd}.npz"
            if f.exists():
                z = np.load(f)
                rows.append(~(z["P"][sel] >= z["thr"]).any(1))     # 놓침
        mats[arm] = np.array(rows, dtype=float)                    # (시드, 조항)
    out = {"n": len(sel), "seeds": {k: int(v.shape[0]) for k, v in mats.items()}, "missed_pct": {}, "diff": {}}
    for arm, M in mats.items():
        out["missed_pct"][arm] = {"전체": round(float(M.mean()) * 100, 1)}
        for a in COMMON:
            out["missed_pct"][arm][a] = round(float(M[:, arts == a].mean()) * 100, 1)

    def diff(x, y, mask):
        X, Y = mats[x][:, mask], mats[y][:, mask]
        pt = float(X.mean() - Y.mean())
        d = X.mean(0) - Y.mean(0)
        g = [gr for gr, m in zip(groups, mask) if m]
        idx = {}
        for i, gr in enumerate(g):
            idx.setdefault(gr, []).append(i)
        cl = [np.array(v) for v in idx.values()]
        b = [d[np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])].mean() for _ in range(2000)]
        se = float(np.sqrt(np.var(b) + X.mean(1).var(ddof=1) / X.shape[0] + Y.mean(1).var(ddof=1) / Y.shape[0]))
        return {"diff_pp": round(pt * 100, 2), "ci95": [round((pt - 1.96 * se) * 100, 2), round((pt + 1.96 * se) * 100, 2)]}
    for a in ("전체",) + tuple(COMMON):
        mask = np.ones(len(sel), bool) if a == "전체" else arts == a
        out["diff"][a] = {"D−B(같은 수 무작위)": diff("D", "B", mask), "D−A(필터 없음)": diff("D", "A", mask)}
    return out


def main() -> None:
    rep = {"rule": "보고용 분석 (사전 등록 아님)", "1_놓친_위반_분류": part1(), "2_조항별_필터_통과율": part2(), "3_필터_전후_모델의_조별_놓침률": part3()}
    save_json(rep, OUT)
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
