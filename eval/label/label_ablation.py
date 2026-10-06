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
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from backend.labeling.articles import ARTICLES, normalize
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("label_ablation.log")

CLEAN_PATH = PROJECT_ROOT / "data/fb_check/clean.jsonl"
NOISE_PATH = PROJECT_ROOT / "data/fb_check/noise.jsonl"
FTC_PATH   = PROJECT_ROOT / "data/raw/ftc_cases/ftc_cases_parsed.json"
OUT_PATH   = PROJECT_ROOT / "data/eval/label_ablation.json"

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


# ── E1 ───────────────────────────────────────────────────────────────────────

def run_e1(clean: list[dict], noise: list[dict], rng: np.random.Generator) -> dict:
    """E1 판정 규칙:
    C(forward)의 F1이 B 분포의 97.5 백분위를 초과 → '양이 아니라 질의 효과'.
    """
    n_clean = len(clean)
    all_rows = clean + noise

    def _run_variant(key: str, nonempty: bool) -> dict:
        # A: 필터 없음 (전체 2,305건)
        scores_all  = _score_rows(all_rows, key, nonempty)
        # C: FB-Check (CLEAN만, 1,473건)
        scores_c    = _score_rows(clean, key, nonempty)
        # B: 무작위 제거 — all_rows에서 n_clean개 BOOT_N번 (빈 포함 여부는 scores_all 기준)
        # 인덱스 기반 subsample 후 mean 분포
        idx_all  = np.array([i for i, r in enumerate(all_rows)
                             if r[key] or not nonempty])
        boot_means = np.array([
            scores_all[rng.choice(len(scores_all), size=min(n_clean, len(scores_all)), replace=False)].mean()
            for _ in range(BOOT_N)
        ])
        pct975 = float(np.percentile(boot_means, 97.5))
        c_mean = float(scores_c.mean()) if len(scores_c) else float("nan")
        verdict = ("양이 아니라 질의 효과" if c_mean > pct975
                   else "미판정(C가 B 97.5%ile 이하)")

        # 순열 p: C mean > 무작위 mean (C가 B 분포에서 얼마나 극단인가)
        perm_p = float((boot_means >= c_mean).mean())

        return {
            "A_all":       _summary(scores_all, rng),
            "B_random":    {"n": n_clean, "boot_n": BOOT_N,
                            "mean_f1_dist": {"mean": float(boot_means.mean()),
                                             "p025": float(np.percentile(boot_means, 2.5)),
                                             "p975": float(pct975)},
                            "note": "A 전체에서 C와 같은 수 무작위 추출"},
            "C_fbcheck":   _summary(scores_c, rng),
            "diff_C_minus_B_mean_pp": float((c_mean - boot_means.mean()) * 100),
            "perm_p_C_gt_B_random":  perm_p,
            "verdict":     verdict,
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
    }


# ── E2 ───────────────────────────────────────────────────────────────────────

_RULES = ["snippet_not_found", "article_mismatch", "label_mismatch", "all"]


def _noise_matching_rule(noise: list[dict], rule: str) -> list[dict]:
    """rule에 해당하는 NOISE 레코드 반환. 'all'은 전체."""
    if rule == "all":
        return noise
    return [r for r in noise if r["noise_reason"].startswith(rule)]


def run_e2(clean: list[dict], noise: list[dict], rng: np.random.Generator) -> dict:
    """E2: FB-Check 구성요소 ablation.
    규칙 하나씩 켰을 때의 CLEAN F1을 '같은 수 무작위 제거'와 비교.
    탐색 표기 — 확증 주장 금지.
    """
    all_rows = clean + noise

    def _run_rule(rule: str) -> dict:
        # '이 규칙으로 걸러낸 NOISE를 제거했을 때' = CLEAN + 해당 규칙 미해당 NOISE
        if rule == "all":
            filtered_rows = clean
        else:
            filtered_rows = clean + [r for r in noise if not r["noise_reason"].startswith(rule)]
        removed = _noise_matching_rule(noise, rule)

        scores_filtered = _score_rows(filtered_rows, "forward", nonempty=True)
        scores_all      = _score_rows(all_rows,       "forward", nonempty=True)

        n_removed = len(removed)
        n_filtered = len(filtered_rows)

        # 무작위로 n_filtered개 선택 BOOT_N번
        boot_means = np.array([
            scores_all[rng.choice(len(scores_all), size=min(n_filtered, len(scores_all)), replace=False)].mean()
            for _ in range(BOOT_N)
        ])
        pct975 = float(np.percentile(boot_means, 97.5))
        f_mean = float(scores_filtered.mean()) if len(scores_filtered) else float("nan")
        verdict = ("이 규칙이 질을 높인다" if f_mean > pct975
                   else "미판정(탐색)")

        return {
            "rule":           rule,
            "n_removed":      n_removed,
            "n_remaining":    n_filtered,
            "filtered":       _summary(scores_filtered, rng),
            "random_baseline": {"mean": float(boot_means.mean()), "p975": float(pct975)},
            "diff_pp":        float((f_mean - boot_means.mean()) * 100),
            "verdict":        verdict,
            "note":           "탐색 — 같은 데이터에서 찾은 규칙, 확증 아님",
        }

    return {
        "experiment":  "E2",
        "description": "FB-Check 구성요소 ablation (탐색 표기)",
        "population":  f"CLEAN {len(clean)} + NOISE {len(noise)} = {len(all_rows)}",
        "label":       "forward, non-empty 기준",
        "boot_n":      BOOT_N,
        "seed":        SEED,
        "rules":       {rule: _run_rule(rule) for rule in _RULES},
    }


# ── main ─────────────────────────────────────────────────────────────────────

def _log_e1(r: dict) -> None:
    for label_key in ("forward_nonempty", "agreed_nonempty"):
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
            f"Δ={v['diff_pp']:+.1f}%p  {v['verdict']}"
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
