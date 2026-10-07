# backend/eval/e9_local_labeling.py
"""E9 — 로컬 오픈모델 라벨러: FB-Check 일반화 (C1'')

## 판정 규칙 (paper_plan.md §E9, 사전 등록)

질문 1: 각 라벨러의 forward F1 (FTC gold 대비).
질문 2: 각 라벨러로 FB-Check를 돌렸을 때 CLEAN > 무작위 제거가 재현되는가 (E1 틀).
질문 3: 라벨러 간 조별 Fleiss kappa.

## 모집단

E1과 동일: source=ftc_case & 근거_법령 있음, CLEAN 1,473 + NOISE 832 = 2,305.

## 실행

    # 100건 파일럿 (속도 + 파싱 실패율 측정):
    python -m eval.e9_local_labeling --model exaone-3.5-7.8b --pilot
    python -m eval.e9_local_labeling --model qwen2.5-14b --pilot
    python -m eval.e9_local_labeling --model llama-3.1-8b --pilot

    # 전체 실행:
    python -m eval.e9_local_labeling --model exaone-3.5-7.8b --gpu 0
    python -m eval.e9_local_labeling --model qwen2.5-14b --gpu 0
    python -m eval.e9_local_labeling --model llama-3.1-8b --gpu 0

    # 세 모델 결과 집계 (모든 raw/*.jsonl 완료 후):
    python -m eval.e9_local_labeling --analyze
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from backend.fb_check.consistency_verification import (
    _FEW_SHOT_EXAMPLES as VERIFY_FEWSHOT,
)
from backend.fb_check.consistency_verification import (
    _SYSTEM_SPAN as VERIFY_SYSTEM,
)
from backend.fb_check.forward_labeling import (
    _FEW_SHOT_EXAMPLES as FWD_FEWSHOT,
)
from backend.fb_check.forward_labeling import (
    _SYSTEM as FWD_SYSTEM,
)
from backend.labeling.articles import ARTICLES, normalize
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json
from eval.label.label_ablation import random_keep_baseline

logger = load_logger("e9_local_labeling.log")

# ── 설정 ────────────────────────────────────────────────────────────────────

MODELS: dict[str, dict] = {
    "exaone-3.5-7.8b": {
        "hf_id":    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        "load_4bit": False,
    },
    "qwen2.5-14b": {
        "hf_id":    "Qwen/Qwen2.5-14B-Instruct",
        "load_4bit": True,
    },
    "llama-3.1-8b": {
        "hf_id":    "meta-llama/Llama-3.1-8B-Instruct",
        "load_4bit": False,
    },
}

CLEAN_PATH = PROJECT_ROOT / "data/fb_check/clean.jsonl"
NOISE_PATH = PROJECT_ROOT / "data/fb_check/noise.jsonl"
FTC_PATH   = PROJECT_ROOT / "data/raw/ftc_cases/ftc_cases_parsed.json"
OUT_DIR    = PROJECT_ROOT / "data/eval/label/e9_raw"
OUT_PATH   = PROJECT_ROOT / "data/eval/label/e9_local_labeling.json"

PILOT_N  = 100
BOOT_N   = 10_000
BOOT_CI  = 5_000
SEED     = 42

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


# ── 데이터 로드 ──────────────────────────────────────────────────────────────

def _ref_by_doc() -> dict:
    cases = json.loads(FTC_PATH.read_text(encoding="utf-8"))["사례"]
    ref: dict[str, set] = {}
    for c in cases:
        doc = str((c.get("셀_데이터") or {}).get("사건번호") or c.get("사건명", ""))
        arts = {a for g in (c.get("근거_법령") or []) if (a := normalize(str(g)))}
        if doc and arts:
            ref[f"ftc_case:{doc}"] = arts
    return ref


def _doc_of(chunk_id: str) -> str:
    parts = str(chunk_id).split(":")
    return ":".join(parts[:2]) if len(parts) >= 3 else str(chunk_id)


def load_population() -> tuple[list[dict], dict]:
    """CLEAN + NOISE에서 source=ftc_case & FTC ref에 있는 레코드 반환."""
    ref = _ref_by_doc()
    rows: list[dict] = []
    for path in (CLEAN_PATH, NOISE_PATH):
        for r in load_jsonl(path):
            if r.get("source") != "ftc_case":
                continue
            doc = _doc_of(r.get("chunk_id", ""))
            if doc not in ref:
                continue
            rows.append({
                "chunk_id":  r["chunk_id"],
                "doc":       doc,
                "text":      r.get("text", ""),
                "gpt_clean": r.get("status") == "CLEAN",
            })
    return rows, ref


# ── 모델 로드 ────────────────────────────────────────────────────────────────

def load_model(model_key: str, gpu: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODELS[model_key]
    hf_id = cfg["hf_id"]
    device_str = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

    if cfg["load_4bit"]:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=bnb_cfg,
            device_map={"": device_str}, trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.bfloat16,
            device_map=device_str, trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer, device_str


# ── 추론 헬퍼 ────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _generate(model, tokenizer, device_str: str, system: str,
              fewshot: list, user_content: str, max_new_tokens: int) -> dict | None:
    import torch
    messages = [{"role": "system", "content": system}]
    messages.extend(fewshot)
    messages.append({"role": "user", "content": user_content})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return _extract_json(text)


def _run_forward(model, tokenizer, device_str: str, clause_text: str) -> dict | None:
    user_content = f"계약 조항:\n{clause_text[:3000]}"
    return _generate(model, tokenizer, device_str,
                     FWD_SYSTEM, FWD_FEWSHOT, user_content, max_new_tokens=300)


def _run_verify(model, tokenizer, device_str: str, evidence_span: str) -> dict | None:
    user_content = f"근거 문구:\n{evidence_span}"
    return _generate(model, tokenizer, device_str,
                     VERIFY_SYSTEM, VERIFY_FEWSHOT, user_content, max_new_tokens=200)


def _label_set(v) -> set:
    return {x for x in (v or []) if x in ARTICLES}


def infer_one(record: dict, model, tokenizer, device_str: str) -> dict:
    """레코드 하나에 대해 forward + verify → {forward_articles, agreed_articles, ...}."""
    clause_text = record["text"]
    t0 = time.time()

    fwd = _run_forward(model, tokenizer, device_str, clause_text)
    if fwd is None:
        return {"chunk_id": record["chunk_id"], "parse_error": "forward_failed",
                "forward_articles": [], "agreed_articles": [],
                "elapsed_s": round(time.time() - t0, 2)}

    forward_articles = list(_label_set(fwd.get("articles", [])))
    evidence_span    = fwd.get("evidence_span", "")

    verify_articles: list = []
    if evidence_span:
        ver = _run_verify(model, tokenizer, device_str, evidence_span)
        if ver is not None:
            verify_articles = list(_label_set(ver.get("articles", [])))

    agreed_articles = list(set(forward_articles) & set(verify_articles))

    return {
        "chunk_id":        record["chunk_id"],
        "doc":             record["doc"],
        "forward_articles": forward_articles,
        "verify_articles":  verify_articles,
        "agreed_articles":  agreed_articles,
        "evidence_span":    evidence_span,
        "parse_error":      None,
        "elapsed_s":        round(time.time() - t0, 2),
    }


# ── 파일럿 실행 ──────────────────────────────────────────────────────────────

def run_pilot(population: list[dict], model_key: str, gpu: int) -> dict:
    logger.info(f"=== E9 파일럿 {PILOT_N}건 [{model_key}] ===")
    model, tokenizer, device_str = load_model(model_key, gpu)

    pilot_pop = population[:PILOT_N]
    results = []
    for i, rec in enumerate(pilot_pop, 1):
        r = infer_one(rec, model, tokenizer, device_str)
        results.append(r)
        if i % 10 == 0 or i == len(pilot_pop):
            errors = sum(1 for x in results if x.get("parse_error"))
            mean_s = np.mean([x["elapsed_s"] for x in results])
            logger.info(f"  [{i}/{PILOT_N}] parse_err={errors} mean_s={mean_s:.1f}")

    total   = len(results)
    errors  = sum(1 for r in results if r.get("parse_error"))
    mean_s  = float(np.mean([r["elapsed_s"] for r in results]))
    est_h   = mean_s * len(population) * 2 / 3600  # forward+verify, rough

    summary = {
        "model":          model_key,
        "pilot_n":        total,
        "parse_fail_n":   errors,
        "parse_fail_rate": round(errors / total, 4) if total else 0,
        "mean_s_per_rec": round(mean_s, 2),
        "est_full_hours": round(est_h, 1),
    }
    logger.info(f"  파일럿 결과: {summary}")
    return summary


# ── 전체 실행 ────────────────────────────────────────────────────────────────

def run_full(population: list[dict], model_key: str, gpu: int) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{model_key}.jsonl"
    logger.info(f"=== E9 전체 실행 [{model_key}] → {out_path} ===")

    # 체크포인트
    done_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done_ids.add(json.loads(line)["chunk_id"])
                except Exception:
                    pass
        logger.info(f"  체크포인트 복원: {len(done_ids)}건 건너뜀")

    pending = [r for r in population if r["chunk_id"] not in done_ids]
    logger.info(f"  처리 대상: {len(pending)}건 / 전체 {len(population)}건")

    model, tokenizer, device_str = load_model(model_key, gpu)

    with open(out_path, "a", encoding="utf-8") as f:
        for i, rec in enumerate(pending, 1):
            result = infer_one(rec, model, tokenizer, device_str)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            if i % 50 == 0 or i == len(pending):
                logger.info(f"  [{i}/{len(pending)}]")

    logger.info(f"  완료: {out_path}")
    return out_path


# ── 분석 ─────────────────────────────────────────────────────────────────────

def _f1(pred: set, ref: set) -> float:
    if not pred and not ref:
        return 1.0
    i = len(pred & ref)
    p = i / len(pred) if pred else 0.0
    r = i / len(ref)  if ref  else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def _score_set(rows: list[dict], key: str, ref: dict) -> np.ndarray:
    vals = [_f1(set(r[key]), ref[r["doc"]]) for r in rows if r.get(key)]
    return np.array(vals, dtype=float)


def _ci(a: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(a) < 2:
        return float("nan"), float("nan")
    boots = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(BOOT_CI)]
    boots = np.sort(boots)
    return float(boots[int(BOOT_CI * 0.025)] * 100), float(boots[int(BOOT_CI * 0.975)] * 100)


def _e1_verdict(rows_all: list[dict], ref: dict, key: str, rng: np.random.Generator) -> dict:
    """E1과 같은 판정 규칙: local-CLEAN(non-empty agreed)의 F1 > 레코드 수준 같은 수 무작위 97.5%ile.

    2026-09-30 수정: 옛 구현은 빈 라벨을 뺀 배열에서 뽑아 표본 크기가 필터 쪽과 달랐다
    (label_ablation.random_keep_baseline 주석 참고).
    """
    has    = np.array([bool(r.get(key)) for r in rows_all])
    scores = np.array([_f1(set(r[key]), ref[r["doc"]]) if r.get(key) else 0.0 for r in rows_all])
    keep   = np.array([bool(r.get("agreed_articles")) for r in rows_all])
    if not has.any() or not keep.any():
        return {"verdict": "집계_불가(데이터_없음)"}
    r = random_keep_baseline(scores, has, keep, rng, BOOT_N)
    rs = random_keep_baseline(scores, has, keep, rng, BOOT_N, groups=[x["doc"] for x in rows_all])
    lo, hi = _ci(scores[keep & has], rng)
    return {
        "n_all":    int(has.sum()),
        "n_clean":  r["n_scored"],
        "A_f1":     round(float(scores[has].mean()) * 100, 2),
        "C_f1":     round(r["mean_f1"] * 100, 2),
        "C_ci95":   [round(lo, 1), round(hi, 1)],
        "B_p975":   round(r["B_p975"] * 100, 2),
        "diff_pp":  round(r["diff_pp"], 2),
        "perm_p":   round(r["perm_p"], 4),
        "verdict":  "CLEAN > 무작위 97.5%ile" if r["pass"] else "미판정(C ≤ B 97.5%ile)",
        "docstrat": {"diff_pp": round(rs["diff_pp"], 2), "perm_p": round(rs["perm_p"], 4),
                     "verdict": "CLEAN > 문서 층화 무작위" if rs["pass"] else "미판정"},
    }


def _fleiss_kappa(labeler_agreed: dict[str, list[dict]], rng: np.random.Generator) -> dict:
    """조별 Fleiss kappa across all labelers on agreed_articles.

    각 레코드 × 조에 대해 labeler가 동일하게 표기한 비율.
    """
    model_keys = list(labeler_agreed.keys())
    if len(model_keys) < 2:
        return {"note": "모델이 2개 미만"}

    # 공통 chunk_id 집합
    id_sets = [set(r["chunk_id"] for r in labeler_agreed[k]) for k in model_keys]
    common_ids = id_sets[0]
    for s in id_sets[1:]:
        common_ids = common_ids & s
    if not common_ids:
        return {"note": "공통 레코드 없음"}

    common_ids_sorted = sorted(common_ids)
    results_by_key = {
        k: {r["chunk_id"]: r for r in labeler_agreed[k]}
        for k in model_keys
    }

    kappas: dict[str, float] = {}
    for art in ARTICLES:
        # 각 레코드에서 라벨러별 binary 라벨 모으기
        ratings = []  # shape (n_records,), each entry = list of 0/1 per rater
        for cid in common_ids_sorted:
            row_ratings = []
            for k in model_keys:
                rec = results_by_key[k].get(cid, {})
                label = 1 if art in (rec.get("agreed_articles") or []) else 0
                row_ratings.append(label)
            ratings.append(row_ratings)

        # Fleiss kappa 계산
        ratings_arr = np.array(ratings, dtype=float)  # (N, R)
        N, R = ratings_arr.shape
        if R < 2:
            continue
        n_j1 = ratings_arr.sum(axis=1)  # 각 레코드에서 라벨=1인 라벨러 수
        n_j0 = R - n_j1

        # 카테고리별 proportion
        p1 = n_j1.mean() / R
        p0 = 1 - p1
        if p1 == 0 or p1 == 1:
            continue  # 모두 같은 라벨 → kappa 정의 불가

        # 관측 합의 비율 P_obs
        p_obs = (n_j1 * (n_j1 - 1) + n_j0 * (n_j0 - 1)).sum() / (N * R * (R - 1))
        # 기대 합의 비율 P_exp
        p_exp = p1 ** 2 + p0 ** 2
        kappa = (p_obs - p_exp) / (1 - p_exp) if p_exp < 1 else 0.0
        kappas[art] = round(float(kappa), 4)

    if not kappas:
        return {"note": "kappa 계산 가능한 조 없음"}

    vals = list(kappas.values())
    return {
        "n_records":    len(common_ids),
        "n_labelers":   len(model_keys),
        "per_article":  kappas,
        "mean_kappa":   round(float(np.mean(vals)), 4),
        "min_kappa":    round(float(np.min(vals)),  4),
        "max_kappa":    round(float(np.max(vals)),  4),
    }


def analyze() -> None:
    """모든 모델 raw/*.jsonl 읽어 E9 최종 결과 산출."""
    logger.info("=== E9 분석 ===")
    population, ref = load_population()

    rng = np.random.default_rng(SEED)
    per_model: dict[str, dict] = {}
    labeler_results: dict[str, list[dict]] = {}

    raw_files = sorted(OUT_DIR.glob("*.jsonl")) if OUT_DIR.exists() else []
    if not raw_files:
        logger.error(f"raw 파일 없음: {OUT_DIR}")
        return

    for raw_path in raw_files:
        model_key = raw_path.stem
        rows = []
        for line in raw_path.read_text("utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass    # 중단된 실행이 남긴 잘린 마지막 줄 — run_full 체크포인트도 같은 규칙으로 건너뛴다
        logger.info(f"  모델: {model_key} — {len(rows)}건")

        parse_errors = sum(1 for r in rows if r.get("parse_error"))
        rows_valid   = [r for r in rows if not r.get("parse_error") and r.get("doc") in ref]
        rows_clean   = [r for r in rows_valid if r.get("agreed_articles")]  # non-empty agreed
        if len(rows) < len(population):
            logger.warning(f"  {model_key}: {len(rows)}/{len(population)} — 미완료라 집계에서 뺀다")
            continue

        forward_verdict = _e1_verdict(rows_valid, ref, "forward_articles", rng)
        # ⚠ agreed 기준은 퇴화한 검정이다: 남기는 기준(agreed 비어 있지 않음)과 채점 대상(agreed 비어
        # 있지 않음)이 같은 집합이라 C == A, 차가 구조적으로 0이다. 기록만 남기고 판정에 쓰지 않는다
        agreed_verdict  = {**_e1_verdict(rows_valid, ref, "agreed_articles", rng),
                           "note": "퇴화 검정(C ≡ A) — 판정에 쓰지 않음. 선별력은 q1_forward_f1(forward 기준)이 E1 틀이다"}

        per_model[model_key] = {
            "n_total":        len(rows),
            "n_parse_error":  parse_errors,
            "n_valid":        len(rows_valid),
            "n_local_clean":  len(rows_clean),
            "clean_rate":     round(len(rows_clean) / len(rows_valid), 4) if rows_valid else 0,
            "q1_forward_f1":  forward_verdict,
            "q2_fbcheck_generalization": agreed_verdict,
        }
        labeler_results[model_key] = rows_valid

    kappa = _fleiss_kappa(labeler_results, rng)

    # ── 질문 1 비교표 (2026-09-30 수정) ──
    # 옛 비교 기준 `gpt_f1_ref = 46.7`은 GPT **CLEAN**의 forward F1이었다 — 로컬 라벨러의
    # **전체** forward F1과 모집단이 다르다. 같은 2,305건에서 GPT forward를 다시 채점하고,
    # 빈 예측 비율(coverage)과 빈 포함 F1을 같이 적는다. non-empty F1만 보면 빈 배열을
    # 많이 내는 라벨러가 확신 있는 레코드만 채점받아 유리해진다.
    gpt_rows = {}
    for path in (CLEAN_PATH, NOISE_PATH):
        for r in load_jsonl(path):
            if r.get("source") == "ftc_case" and _doc_of(r.get("chunk_id", "")) in ref:
                gpt_rows[r["chunk_id"]] = {"doc": _doc_of(r["chunk_id"]),
                                           "forward_articles": list(_label_set(r.get("forward_articles")))}
    ids = [r["chunk_id"] for r in population]

    def _q1_row(rows_by_id: dict) -> dict:
        rs = [rows_by_id[i] for i in ids if i in rows_by_id]
        full = np.array([_f1(set(r.get("forward_articles") or []), ref[r["doc"]]) for r in rs])
        ne = np.array([_f1(set(r["forward_articles"]), ref[r["doc"]]) for r in rs if r.get("forward_articles")])
        return {"n": len(rs), "coverage": round(len(ne) / len(rs), 4),
                "f1_nonempty_pp": round(float(ne.mean()) * 100, 2),
                "f1_incl_empty_pp": round(float(full.mean()) * 100, 2)}

    from eval.analysis.confound_articles import best_constant
    c_f1, c_set = best_constant([("ftc_case", frozenset(ref[gpt_rows[i]["doc"]])) for i in ids])
    const = {"articles": sorted(c_set), "f1_pp": round(c_f1 * 100, 2)}
    q1_table = {"gpt-4o": _q1_row(gpt_rows)}
    for k, rows_valid in labeler_results.items():
        q1_table[k] = _q1_row({r["chunk_id"]: r for r in rows_valid})

    report = {
        "experiment":     "E9",
        "description":    "로컬 오픈모델 라벨러 — FB-Check 일반화 (C1'')",
        "population":     "CLEAN 1473 + NOISE 832 = 2305 (source=ftc_case)",
        "seed":           SEED,
        "boot_n":         BOOT_N,
        "judgment_rule":  "local-CLEAN(non-empty agreed_articles) mean F1 > B_random 97.5%ile",
        "q1_table":       q1_table,
        "q1_constant":    const,
        "per_model":      per_model,
        "q3_fleiss_kappa": kappa,
    }

    save_json(report, OUT_PATH)
    logger.info(f"저장: {OUT_PATH}")

    for model_key, m in per_model.items():
        fwd = m["q1_forward_f1"]
        agr = m["q2_fbcheck_generalization"]
        logger.info(
            f"  [{model_key}]  forward F1={fwd.get('A_f1','?')}%  "
            f"CLEAN={agr.get('C_f1','?')}% Δ={agr.get('diff_pp','?'):+}%p  {agr.get('verdict','?')}"
        )
    logger.info(f"  Fleiss κ 평균={kappa.get('mean_kappa','?')}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  choices=list(MODELS), default=None)
    parser.add_argument("--pilot",  action="store_true", help="100건 파일럿만 실행")
    parser.add_argument("--gpu",    type=int, default=0)
    parser.add_argument("--analyze", action="store_true", help="raw 파일 집계만 실행")
    args = parser.parse_args()

    if args.analyze:
        analyze()
        return

    if not args.model:
        parser.error("--model 또는 --analyze 필요")

    population, _ = load_population()
    logger.info(f"모집단: {len(population)}건")

    if args.pilot:
        summary = run_pilot(population, args.model, args.gpu)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        run_full(population, args.model, args.gpu)


if __name__ == "__main__":
    main()
