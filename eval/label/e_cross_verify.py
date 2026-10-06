# eval/label/e_cross_verify.py
"""E_xv — 검증자 독립성: GPT forward × 로컬 verify (P2 ①)

## 질문

FB-Check는 forward와 verify에 **같은 gpt-4o**를 쓴다. 같은 모델은 오류가 상관돼 있을 수
있다 — 둘이 같은 이유로 틀리면 "합의"가 오류를 걸러내지 못한다. verify만 **다른 모델**로
바꾸면 필터가 더 나은 레코드를 고르는가?

## 설계 (사전 등록, 실행 전 paper_plan.md §P2-①에 적음)

- 모집단: E1과 같다. source=ftc_case & 근거_법령 있음, CLEAN 1,473 + NOISE 832 = 2,305
- forward: 기존 gpt-4o 출력 그대로(`forward_label`, `forward_articles`, `evidence_span`,
  `snippet_exists`). **API 호출 0.**
- verify: 로컬 모델이 `consistency_verification`의 **같은 프롬프트**(span/clause, few-shot 동일)로
  재판정. 판정 규칙도 `fb_check/__main__.py`와 같다
  (snippet 없음 → NOISE / 등급 불일치 → NOISE / 위험인데 조 교집합 없음 → NOISE / 나머지 CLEAN)
- 비교 대상: 같은 forward 라벨에 대해
  - F_gpt  : 현행 FB-Check(verify=gpt-4o)가 CLEAN으로 남긴 레코드
  - F_xv   : verify=로컬 모델이 CLEAN으로 남긴 레코드
  - 각각 **같은 수 무작위 유지**(레코드 단위, 10,000회)와 비교
- 지표: 남긴 레코드의 forward 라벨 FTC F1(non-empty). E1과 같은 채점
- 판정:
  1. 각 필터: 남긴 집합 F1 > 같은 수 무작위 97.5%ile → "선별력 있음"
  2. **F_xv − F_gpt** (무작위 대비 이득의 차, 레코드 부트스트랩 95% CI).
     CI가 0을 넘으면 "독립 검증자가 낫다", 0 미만이면 "같은 모델이 낫다", 포함하면 미판정(MDE 병기)
- 부가: GPT verify와 로컬 verify의 CLEAN/NOISE 일치율·Cohen κ, NOISE 집합 겹침

## 실행

    python -m eval.label.e_cross_verify --model qwen2.5-14b --gpu 1
    python -m eval.label.e_cross_verify --model llama-3.1-8b --gpu 1
    python -m eval.label.e_cross_verify --analyze
"""

import argparse
import json
import time

import numpy as np

from backend.fb_check.consistency_verification import (
    _FEW_SHOT_EXAMPLES as VERIFY_FEWSHOT,
)
from backend.fb_check.consistency_verification import (
    _SYSTEM_CLAUSE,
    _SYSTEM_SPAN,
    MAX_VERIFY_CHARS,
)
from backend.labeling.articles import ARTICLES
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json
from eval.label.e9_local_labeling import _generate, load_model
from eval.label.label_ablation import _doc_of, _f1, _labels, _ref_by_doc

logger = load_logger("e_cross_verify.log")

CLEAN_PATH = PROJECT_ROOT / "data/fb_check/clean.jsonl"
NOISE_PATH = PROJECT_ROOT / "data/fb_check/noise.jsonl"
OUT_DIR    = PROJECT_ROOT / "data/eval/e_cross_verify_raw"
OUT_PATH   = PROJECT_ROOT / "data/eval/e_cross_verify.json"

BOOT_N = 10_000
SEED   = 42


# ── 모집단 ───────────────────────────────────────────────────────────────────

def load_population() -> tuple[list[dict], dict]:
    ref = _ref_by_doc()
    rows = []
    for path in (CLEAN_PATH, NOISE_PATH):
        for r in load_jsonl(path):
            if r.get("source") != "ftc_case" or _doc_of(r.get("chunk_id", "")) not in ref:
                continue
            rows.append(r)
    return rows, ref


# ── 로컬 verify (fb_check/__main__.py 판정 규칙 그대로) ─────────────────────

def _verify_local(r: dict, model, tokenizer, device_str: str) -> dict:
    fwd_arts  = [a for a in (r.get("forward_articles") or []) if a in ARTICLES]
    fwd_label = r.get("forward_label", "")
    out = {"chunk_id": r["chunk_id"], "gpt_status": r.get("status")}

    if fwd_arts:
        if not r.get("snippet_exists"):
            return {**out, "status": "NOISE", "noise_reason": "snippet_not_found",
                    "verify_label": None, "verify_articles": []}
        v = _generate(model, tokenizer, device_str, _SYSTEM_SPAN, VERIFY_FEWSHOT,
                      f"근거 문구:\n{r.get('evidence_span', '')}", max_new_tokens=200)
    else:
        v = _generate(model, tokenizer, device_str, _SYSTEM_CLAUSE, [],
                      f"계약 조항:\n{r.get('text', '')[:MAX_VERIFY_CHARS]}", max_new_tokens=200)

    if v is None:
        return {**out, "status": "ERROR", "noise_reason": "parse_failed",
                "verify_label": None, "verify_articles": []}

    # _normalize와 같은 규칙: 알 수 없는 조 제거, 빈 배열이면 Low 강제
    # 로컬 모델은 형식을 어길 때가 있다(Llama가 articles에 객체를 넣음) — 문자열만 조로 인정한다
    arts = v.get("articles") if isinstance(v.get("articles"), list) else []
    v_arts  = list(dict.fromkeys(a for a in arts if isinstance(a, str) and a in ARTICLES))
    v_label = v.get("risk_level") if isinstance(v.get("risk_level"), str) else "Low"
    v_label = (v_label or "Low") if v_arts else "Low"
    agreed  = [a for a in fwd_arts if a in set(v_arts)]
    out.update({"verify_label": v_label, "verify_articles": v_arts, "agreed_articles": agreed})

    if not fwd_label or fwd_label != v_label:
        return {**out, "status": "NOISE", "noise_reason": f"label_mismatch: {fwd_label} != {v_label}"}
    if fwd_label != "Low" and not agreed:
        return {**out, "status": "NOISE", "noise_reason": "article_mismatch"}
    return {**out, "status": "CLEAN"}


def run(model_key: str, gpu: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{model_key}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["chunk_id"] for line in out_path.read_text("utf-8").splitlines() if line.strip()}
    rows, _ = load_population()
    pending = [r for r in rows if r["chunk_id"] not in done]
    logger.info(f"=== E_xv [{model_key}] 전체 {len(rows)} · 완료 {len(done)} · 남음 {len(pending)} ===")
    if not pending:
        return

    model, tokenizer, device_str = load_model(model_key, gpu)
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, r in enumerate(pending, 1):
            f.write(json.dumps(_verify_local(r, model, tokenizer, device_str), ensure_ascii=False) + "\n")
            if i % 100 == 0 or i == len(pending):
                f.flush()
                logger.info(f"  [{i}/{len(pending)}] {(time.time() - t0) / i:.2f}s/건")


# ── 분석 ─────────────────────────────────────────────────────────────────────

def _selection(scores: np.ndarray, has: np.ndarray, keep: np.ndarray, rng) -> dict:
    """keep 마스크로 남긴 집합의 F1(non-empty) vs 같은 수 레코드 무작위 유지."""
    n_keep = int(keep.sum())
    kept   = scores[keep & has]
    boots  = np.empty(BOOT_N)
    for b in range(BOOT_N):
        m = np.zeros(len(scores), bool)
        m[rng.choice(len(scores), n_keep, replace=False)] = True
        boots[b] = scores[m & has].mean()
    f1 = float(kept.mean())
    return {"n_keep": n_keep, "n_scored": int(len(kept)), "f1_pp": round(f1 * 100, 2),
            "rand_mean_pp": round(float(boots.mean()) * 100, 2),
            "rand_p975_pp": round(float(np.percentile(boots, 97.5)) * 100, 2),
            "gain_pp": round((f1 - float(boots.mean())) * 100, 2),
            "perm_p": float((boots >= f1).mean()),
            "verdict": "선별력 있음" if f1 > np.percentile(boots, 97.5) else "미판정"}


def _gain_diff_ci(scores, has, keep_a, keep_b, rng, n_boot: int = 5000) -> dict:
    """(A가 남긴 F1 − 전체 F1) − (B가 남긴 F1 − 전체 F1)의 레코드 부트스트랩 CI.
    전체 평균은 양쪽에 같으므로 차는 kept-F1 차와 같다."""
    n = len(scores)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        s, h, a, c = scores[i], has[i], keep_a[i], keep_b[i]
        diffs[b] = s[a & h].mean() - s[c & h].mean()
    point = scores[keep_a & has].mean() - scores[keep_b & has].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    verdict = "A가 낫다" if lo > 0 else ("B가 낫다" if hi < 0 else "미판정")
    return {"diff_pp": round(float(point) * 100, 2), "ci95_pp": [round(lo * 100, 2), round(hi * 100, 2)],
            "se_pp": round(float(diffs.std()) * 100, 2),
            "mde80_pp": round(float(diffs.std()) * 2.8 * 100, 2), "verdict": verdict}


def _docstrat(scores, has, pop, keep, docs, rng) -> dict:
    """pop 안에서 keep이 고른 집합 vs 문서별 같은 수 무작위(층화 순열)."""
    from eval.label.label_ablation import random_keep_baseline
    idx = np.where(pop)[0]
    r = random_keep_baseline(scores[idx], has[idx], keep[idx], rng, BOOT_N, groups=[docs[i] for i in idx])
    return {"n_pop": int(len(idx)), "n_keep": r["n_keep"], "f1_pp": round(r["mean_f1"] * 100, 2),
            "strat_rand_mean_pp": round(r["B_mean"] * 100, 2), "gain_pp": round(r["diff_pp"], 2),
            "perm_p": r["perm_p"], "verdict": "선별력 있음(문서 층화)" if r["pass"] else "미판정"}


def _kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def analyze() -> None:
    rows, ref = load_population()
    ids    = [r["chunk_id"] for r in rows]
    fwd    = [_labels(r.get("forward_articles")) for r in rows]
    has    = np.array([bool(x) for x in fwd])
    scores = np.array([_f1(x, ref[_doc_of(r["chunk_id"])]) if x else np.nan for x, r in zip(fwd, rows)])
    scores_filled = np.nan_to_num(scores)
    keep_gpt = np.array([r.get("status") == "CLEAN" for r in rows])
    docs = [_doc_of(r["chunk_id"]) for r in rows]

    rng = np.random.default_rng(SEED)
    report = {
        "experiment": "E_xv",
        "population": f"ftc_case & 근거_법령 있음 n={len(rows)} (GPT CLEAN {int(keep_gpt.sum())})",
        "label_scored": "gpt-4o forward_articles (non-empty) — 모든 팔이 같은 라벨, 필터만 다르다",
        "rule": "필터별: 남긴 F1 > 같은 수 무작위 유지 97.5%ile / 필터 간: 남긴 F1 차 레코드 부트스트랩 CI",
        "all_f1_pp": round(float(np.nanmean(scores)) * 100, 2),
        "filters": {"gpt4o_verify": _selection(scores_filled, has, keep_gpt, rng)},
        "vs_gpt": {},
    }

    for raw in sorted(OUT_DIR.glob("*.jsonl")):
        res = {json.loads(line)["chunk_id"]: json.loads(line) for line in raw.read_text("utf-8").splitlines() if line.strip()}
        if len(res) < len(rows):
            logger.warning(f"  {raw.stem}: {len(res)}/{len(rows)} — 미완료, 건너뜀")
            continue
        st = [res[i]["status"] for i in ids]
        keep_xv = np.array([s == "CLEAN" for s in st])
        err     = np.array([s == "ERROR" for s in st])
        n_err = sum(s == "ERROR" for s in st)
        report["filters"][f"{raw.stem}_verify"] = {**_selection(scores_filled, has, keep_xv, rng), "n_parse_error": n_err}
        noise_g, noise_x = ~keep_gpt, ~keep_xv
        report["vs_gpt"][raw.stem] = {
            "cleanrate_gpt": round(float(keep_gpt.mean()), 4), "cleanrate_xv": round(float(keep_xv.mean()), 4),
            "status_agreement": round(float((keep_gpt == keep_xv).mean()), 4),
            "status_kappa": round(_kappa(keep_gpt.astype(int), keep_xv.astype(int)), 4),
            "noise_jaccard": round(float((noise_g & noise_x).sum() / max((noise_g | noise_x).sum(), 1)), 4),
            "xv_minus_gpt": _gain_diff_ci(scores_filled, has, keep_xv, keep_gpt, rng),
            # 둘 다 통과시킨 레코드만(교집합 = 두 검증자 합의) — 앙상블 필터
            "intersection": _selection(scores_filled, has, keep_xv & keep_gpt, rng),
            # ★ 남긴 수가 달라 위 비교는 공정하지 않다(적게 남긴 쪽이 유리할 수 있다).
            # 수와 무관한 질문: GPT가 통과시킨 것 **안에서** 독립 검증자의 판정이 품질을 가르는가,
            # GPT가 버린 것 안에서 독립 검증자가 살린 것이 더 나은가
            "within_gpt_clean": _gain_diff_ci(scores_filled, has, keep_gpt & keep_xv,
                                              keep_gpt & ~keep_xv & ~err, rng),
            "within_gpt_noise": _gain_diff_ci(scores_filled, has, ~keep_gpt & keep_xv,
                                              ~keep_gpt & ~keep_xv & ~err, rng),
            "n_cells": {"gpt_clean&xv_clean": int((keep_gpt & keep_xv).sum()),
                        "gpt_clean&xv_noise": int((keep_gpt & ~keep_xv & ~err).sum()),
                        "gpt_noise&xv_clean": int((~keep_gpt & keep_xv).sum()),
                        "gpt_noise&xv_noise": int((~keep_gpt & ~keep_xv & ~err).sum()),
                        "xv_parse_error": int(err.sum())},
            # 문서 층화: GPT CLEAN 안에서 독립 검증자가 남긴 것 vs 같은 문서에서 같은 수를 무작위로
            # 남긴 것. 사건 단위 준거라 "쉬운 문서를 더 남겼다"만으로도 차가 생기므로 그걸 뗀다
            "within_gpt_clean_docstrat": _docstrat(scores_filled, has, keep_gpt & ~err, keep_xv, docs, rng),
            # 파싱 실패를 CLEAN으로 봤을 때(NOISE로 본 기본값의 반대 극단)
            "parse_error_as_clean": _selection(scores_filled, has, keep_xv | err, rng),
        }

    save_json(report, OUT_PATH)
    logger.info(json.dumps(report, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        analyze()
    elif args.model:
        run(args.model, args.gpu)
    else:
        ap.error("--model 또는 --analyze")


if __name__ == "__main__":
    main()
