# backend/eval/label_ablation.py
"""E1 · E2 — 라벨 수준 ablation: 필터 없음 / 무작위 제거 / FB-Check / 구성요소별

## 판정 규칙 (paper_plan.md §4 E1, 사전 등록)

C의 F1이 B 분포의 97.5 백분위를 초과하면 "양이 아니라 질의 효과" 판정.
순열 p값(C > 무작위 표본) 도 같이 낸다.
빈 예측 처리: fbcheck_external_eval과 동일하게 non-empty 기준 주, 빈 포함 병기.

## 모집단

source=ftc_case이고 사건에 근거_법령이 있는 레코드. fbcheck_external_eval과 동일 필터.
CLEAN 1,473 + NOISE 832 = 2,305.

## 라벨

forward (양 조건이 같은 라벨러 출력). agreed 병기.

실행:
    python -m eval.label_ablation              # E1 + E2
    python -m eval.label_ablation --e1-only
    python -m eval.label_ablation --e2-only
"""

import argparse
import json
from pathlib import Path

import numpy as np

from backend.labeling.articles import ARTICLES, normalize
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("label_ablation.log")

CLEAN_PATH = PROJECT_ROOT / "data/fb_check/clean.jsonl"
NOISE_PATH = PROJECT_ROOT / "data/fb_check/noise.jsonl"
FTC_PATH   = PROJECT_ROOT / "data/raw/ftc_cases/ftc_cases_parsed.json"
OUT_PATH   = PROJECT_ROOT / "data/eval/label/label_ablation.json"

BOOT_N  = 10_000   # E1 무작위 제거 반복 수 (paper_plan §E1)
BOOT_CI = 5_000    # CI용 부트스트랩 반복 수
SEED    = 42
Z_ALPHA, Z_POWER = 1.96, 0.84


# ── helpers ──────────────────────────────────────────────────────────────────

def _f1(pred: set, ref: set) -> float:
    if not pred and not ref:
        return 1.0
    i = len(pred & ref)
    p = i / len(pred) if pred else 0.0
    r = i / len(ref)  if ref  else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def _labels(v) -> set:
    return {x for x in (v or []) if x in ARTICLES}


def _ref_by_doc() -> dict:
    cases = json.loads(FTC_PATH.read_text(encoding="utf-8"))["사례"]
    ref = {}
    for c in cases:
        doc = str((c.get("셀_데이터") or {}).get("사건번호") or c.get("사건명", ""))
        arts = {a for g in (c.get("근거_법령") or []) if (a := normalize(str(g)))}
        if doc and arts:
            ref[f"ftc_case:{doc}"] = arts
    return ref


def _doc_of(chunk_id: str) -> str:
    parts = str(chunk_id).split(":")
    return ":".join(parts[:2]) if len(parts) >= 3 else str(chunk_id)


def _load_filtered(path: Path, ref: dict) -> list[dict]:
    out = []
    for r in load_jsonl(path):
        if r.get("source") != "ftc_case":
            continue
        doc = _doc_of(r.get("chunk_id", ""))
        if doc not in ref:
            continue
        out.append({
            "doc":          doc,
            "forward":      _labels(r.get("forward_articles")),
            "agreed":       _labels(r.get("agreed_articles")),
            "noise_reason": str(r.get("noise_reason") or ""),
        })
    return out


def _score_rows(rows: list[dict], key: str, nonempty: bool = True) -> np.ndarray:
    ref = _score_rows._ref
    vals = [_f1(r[key], ref[r["doc"]]) for r in rows
            if (r[key] or not nonempty)]
    return np.array(vals, dtype=float)


def _ci(a: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(a) < 2:
        return float("nan"), float("nan")
    boots = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(BOOT_CI)]
    boots = np.sort(boots)
    return float(boots[int(BOOT_CI * 0.025)] * 100), float(boots[int(BOOT_CI * 0.975)] * 100)


def _summary(scores: np.ndarray, rng: np.random.Generator) -> dict:
    lo, hi = _ci(scores, rng)
    return {"n": int(len(scores)), "mean_f1": float(scores.mean()) if len(scores) else float("nan"),
            "ci95_pp": [lo, hi]}


# ── 무작위 제거 기준선 (2026-09-30 수정) ─────────────────────────────────────
#
# 옛 구현은 **빈 라벨을 이미 뺀 점수 배열**에서 "C의 레코드 수(빈 포함 1,473)"만큼 뽑았다.
# C 쪽은 빈 라벨을 뺀 1,158건 평균이라, 두 평균의 표본 크기가 달랐다(1,473 vs 1,158).
# 표본이 클수록 무작위 평균의 분산이 작아져 97.5%ile이 낮아진다 → **통과 쪽으로 기운다.**
# E2에서는 남긴 수(2,288)가 빈 라벨 제외 풀(1,974)보다 커서 표본이 풀 전체가 되고
# 분산이 0이 됐다(snippet_not_found가 +0.1%p로 "질을 높인다" 판정을 받은 원인).
#
# 고친 정의: 필터와 **똑같이** 레코드(빈 포함) 수준에서 n_keep개를 무작위로 남기고,
# 그 안에서 빈 라벨을 빼고 채점한다. "같은 수를 무작위로 제거했다면"의 문자 그대로다.

def random_keep_baseline(scores: np.ndarray, has: np.ndarray, keep: np.ndarray,
                         rng: np.random.Generator, boot_n: int = BOOT_N,
                         groups: list[str] | None = None) -> dict:
    """keep 마스크(필터가 남긴 레코드)의 F1(non-empty) vs 레코드 수준 같은 수 무작위 유지.

    `groups`(문서 id)를 주면 **문서마다 필터가 남긴 수만큼만** 무작위로 남긴다(층화 순열).
    준거가 사건 단위라 같은 문서의 조항은 같은 정답을 공유한다 — 문서마다 상수 F1이 크게
    다르므로(이 모집단의 최적 상수가 54.4%로 모든 라벨러를 이긴다), 필터가 "쉬운 문서"의
    조항을 더 남기기만 해도 비층화 기준선을 넘는다. 층화하면 남는 것은 **같은 문서 안에서
    어느 조항을 골랐는가**뿐이다.
    """
    n, n_keep = len(scores), int(keep.sum())
    kept = scores[keep & has]
    boots = np.empty(boot_n)
    if groups is not None:
        idx: dict[str, list[int]] = {}
        for i, g in enumerate(groups):
            idx.setdefault(g, []).append(i)
        blocks = [(np.array(v), int(keep[v].sum())) for v in idx.values()]
    for b in range(boot_n):
        m = np.zeros(n, bool)
        if groups is None:
            m[rng.choice(n, n_keep, replace=False)] = True
        else:
            for ids, k in blocks:
                if k:
                    m[rng.choice(ids, k, replace=False)] = True
        boots[b] = scores[m & has].mean()
    f1 = float(kept.mean())
    p975 = float(np.percentile(boots, 97.5))
    return {"n_keep": n_keep, "n_scored": int(len(kept)), "mean_f1": f1,
            "B_mean": float(boots.mean()), "B_p025": float(np.percentile(boots, 2.5)), "B_p975": p975,
            "diff_pp": (f1 - float(boots.mean())) * 100, "perm_p": float((boots >= f1).mean()),
            "pass": bool(f1 > p975)}


def filter_diff_ci(scores: np.ndarray, has: np.ndarray, keep_a: np.ndarray, keep_b: np.ndarray,
                   rng: np.random.Generator, n_boot: int = BOOT_CI,
                   groups: list[str] | None = None) -> dict:
    """두 필터가 남긴 집합의 F1 차(A−B). **레코드를 한 번에 재표집**해 두 집합의 겹침을 보존한다.

    `groups`(문서 id)를 주면 문서 단위로 재표집한다 — 같은 문서 조항은 정답을 공유하므로
    레코드 단위 CI는 실제보다 좁다.
    """
    n = len(scores)
    if groups is not None:
        idx: dict[str, list[int]] = {}
        for k, g in enumerate(groups):
            idx.setdefault(g, []).append(k)
        clusters = [np.array(v) for v in idx.values()]
    d = np.empty(n_boot)
    for b in range(n_boot):
        if groups is None:
            i = rng.integers(0, n, n)
        else:
            i = np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
        s, h = scores[i], has[i]
        d[b] = s[keep_a[i] & h].mean() - s[keep_b[i] & h].mean()
    point = float(scores[keep_a & has].mean() - scores[keep_b & has].mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff_pp": point * 100, "ci95_pp": [float(lo) * 100, float(hi) * 100],
            "mde80_pp": float(d.std()) * 2.8 * 100,
            "verdict": "A > B(유의)" if lo > 0 else ("B > A(유의)" if hi < 0 else "미판정")}


def score_vector(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    """레코드별 F1(빈 라벨은 0으로 채우고 has=False) — 마스크 연산용."""
    ref = _score_rows._ref
    has = np.array([bool(r[key]) for r in rows])
    s = np.array([_f1(r[key], ref[r["doc"]]) if r[key] else 0.0 for r in rows])
    return s, has


# ── E1 ───────────────────────────────────────────────────────────────────────

def run_e1(clean: list[dict], noise: list[dict], rng: np.random.Generator) -> dict:
    """E1 판정 규칙:
    C(forward)의 F1이 B 분포의 97.5 백분위를 초과 → '양이 아니라 질의 효과'.
    """
    n_clean = len(clean)
    all_rows = clean + noise

    docs = [r["doc"] for r in all_rows]

    def _run_variant(key: str, nonempty: bool, stratify: bool = False) -> dict:
        scores, has = score_vector(all_rows, key)
        if not nonempty:
            has = np.ones(len(all_rows), bool)
        keep = np.array([True] * n_clean + [False] * (len(all_rows) - n_clean))
        r = random_keep_baseline(scores, has, keep, rng, groups=docs if stratify else None)
        return {
            "A_all":       _summary(scores[has], rng),
            "B_random":    {"n_keep": r["n_keep"], "boot_n": BOOT_N,
                            "mean_f1_dist": {"mean": r["B_mean"], "p025": r["B_p025"], "p975": r["B_p975"]},
                            "note": "레코드 수준에서 C와 같은 수(빈 포함)를 무작위 유지 후 C와 같은 규칙으로 채점"},
            "C_fbcheck":   _summary(scores[keep & has], rng),
            "diff_C_minus_B_mean_pp": r["diff_pp"],
            "perm_p_C_gt_B_random":  r["perm_p"],
            "verdict":     "양이 아니라 질의 효과" if r["pass"] else "미판정(C가 B 97.5%ile 이하)",
        }

    return {
        "experiment":  "E1",
        "description": "필터 없음(A) / 무작위 제거(B) / FB-Check(C) 라벨 수준 ablation",
        "population":  f"source=ftc_case & 근거_법령 있음: CLEAN {len(clean)} + NOISE {len(noise)} = {len(all_rows)}",
        "judgment_rule": "C(forward).mean > B(랜덤) 97.5%ile → '양이 아니라 질의 효과'",
        "boot_n_random": BOOT_N,
        "seed": SEED,
        "forward_nonempty": _run_variant("forward", nonempty=True),
        "forward_all_incl_empty": _run_variant("forward", nonempty=False),
        "agreed_nonempty":  _run_variant("agreed",  nonempty=True),
        "agreed_all_incl_empty":  _run_variant("agreed",  nonempty=False),
        # 2026-09-30 추가: 문서 층화 기준선. 사건 단위 준거의 문서 구성 교락을 뗀다
        "forward_nonempty_docstrat": _run_variant("forward", nonempty=True, stratify=True),
        "agreed_nonempty_docstrat":  _run_variant("agreed",  nonempty=True, stratify=True),
    }


# ── E2 ───────────────────────────────────────────────────────────────────────

_RULES = ["snippet_not_found", "article_mismatch", "label_mismatch", "all"]


def run_e2(clean: list[dict], noise: list[dict], rng: np.random.Generator) -> dict:
    """E2: FB-Check 구성요소 ablation.
    규칙 하나씩 켰을 때의 CLEAN F1을 '같은 수 무작위 제거'와 비교.
    탐색 표기 — 확증 주장 금지.
    """
    all_rows = clean + noise

    scores, has = score_vector(all_rows, "forward")
    is_noise = np.array([False] * len(clean) + [True] * len(noise))
    reasons = [r["noise_reason"] for r in all_rows]

    def _run_rule(rule: str) -> dict:
        # '이 규칙으로 걸러낸 NOISE만 제거' = CLEAN + 해당 규칙 미해당 NOISE
        removed = np.array([n and (rule == "all" or rs.startswith(rule)) for n, rs in zip(is_noise, reasons)])
        keep = ~removed
        r = random_keep_baseline(scores, has, keep, rng)
        rs = random_keep_baseline(scores, has, keep, rng, groups=[x["doc"] for x in all_rows])
        return {
            "rule":           rule,
            "n_removed":      int(removed.sum()),
            "n_remaining":    int(keep.sum()),
            "filtered":       _summary(scores[keep & has], rng),
            "random_baseline": {"mean": r["B_mean"], "p975": r["B_p975"]},
            "diff_pp":        r["diff_pp"],
            "perm_p":         r["perm_p"],
            "verdict":        "이 규칙이 질을 높인다" if r["pass"] else "미판정(탐색)",
            "docstrat":       {"p975": rs["B_p975"], "diff_pp": rs["diff_pp"], "perm_p": rs["perm_p"],
                               "verdict": "이 규칙이 질을 높인다" if rs["pass"] else "미판정(탐색)"},
            "note":           "탐색 — 같은 데이터에서 찾은 규칙, 확증 아님",
        }

    return {
        "experiment":  "E2",
        "description": "FB-Check 구성요소 ablation (탐색 표기)",
        "population":  f"CLEAN {len(clean)} + NOISE {len(noise)} = {len(all_rows)}",
        "label":       "forward, non-empty 기준 (무작위 기준선은 레코드 수준 동수 유지)",
        "boot_n":      BOOT_N,
        "seed":        SEED,
        "rules":       {rule: _run_rule(rule) for rule in _RULES},
    }


# ── main ─────────────────────────────────────────────────────────────────────

def _log_e1(r: dict) -> None:
    for label_key in ("forward_nonempty", "agreed_nonempty",
                      "forward_nonempty_docstrat", "agreed_nonempty_docstrat"):
        v = r[label_key]
        a = v["A_all"]
        c = v["C_fbcheck"]
        b = v["B_random"]["mean_f1_dist"]
        logger.info(
            f"  [{label_key}]  A {a['n']:>5} {a['mean_f1']*100:.1f}%  "
            f"B_rand {b['mean']*100:.1f}% p975={b['p975']*100:.1f}%  "
            f"C {c['n']:>5} {c['mean_f1']*100:.1f}%  "
            f"Δ(C-B)={v['diff_C_minus_B_mean_pp']:+.1f}%p  "
            f"perm_p={v['perm_p_C_gt_B_random']:.3f}  {v['verdict']}"
        )


def _log_e2(r: dict) -> None:
    for rule, v in r["rules"].items():
        logger.info(
            f"  [{rule:<22}]  removed={v['n_removed']:>3}  "
            f"filtered_mean={v['filtered']['mean_f1']*100:.1f}%  "
            f"rand_p975={v['random_baseline']['p975']*100:.1f}%  "
            f"Δ={v['diff_pp']:+.1f}%p  {v['verdict']}  | 문서층화 Δ={v['docstrat']['diff_pp']:+.1f}%p "
            f"p={v['docstrat']['perm_p']:.3f} {v['docstrat']['verdict']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-only", action="store_true")
    parser.add_argument("--e2-only", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    ref = _ref_by_doc()
    _score_rows._ref = ref  # 전역 공유

    clean = _load_filtered(CLEAN_PATH, ref)
    noise = _load_filtered(NOISE_PATH, ref)
    logger.info(f"모집단: CLEAN {len(clean)} · NOISE {len(noise)} · TOTAL {len(clean)+len(noise)}")

    report: dict = {
        "provenance": {
            "script":     "backend/eval/label_ablation.py",
            "population": f"CLEAN {len(clean)} NOISE {len(noise)}",
            "seed":       SEED,
            "boot_n_random": BOOT_N,
            "judgment_rule_e1": (
                "C(forward).mean_f1 > B_random 97.5%ile → '양이 아니라 질의 효과'. "
                "paper_plan.md §4 E1 사전 등록"
            ),
        }
    }

    if not args.e2_only:
        logger.info("=== E1: 라벨 수준 ablation ===")
        report["e1"] = run_e1(clean, noise, rng)
        _log_e1(report["e1"])

    if not args.e1_only:
        logger.info("=== E2: FB-Check 구성요소 ablation ===")
        report["e2"] = run_e2(clean, noise, rng)
        _log_e2(report["e2"])

    save_json(report, OUT_PATH)
    logger.info(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
