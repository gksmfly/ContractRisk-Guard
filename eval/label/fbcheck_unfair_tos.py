# eval/label/fbcheck_unfair_tos.py
"""FB-Check를 공개 사람 라벨(UNFAIR-ToS, 영문 약관)에서 시험 — Qwen (사전 등록: eval/paper_plan.md §7-28, 2026-10-15, API 0)

LexGLUE `unfair_tos`(data/eval/public/unfair_tos/*.jsonl) 9,414문장에서 위반 1,032 + 공정 1,032(무작위)를 뽑아
Qwen2.5-14B가 붙인 불공정 유형 라벨을 FB-Check 2진 규칙으로 걸러 보고, 남은 라벨이 같은 수 무작위보다 정답에 가까운지 본다.

    HF_HUB_OFFLINE=1 .venv/bin/python -m eval.label.fbcheck_unfair_tos --smoke --gpu 1     # 형식 점검(24문장, 결과 비보고)
    HF_HUB_OFFLINE=1 .venv/bin/python -m eval.label.fbcheck_unfair_tos --run --gpu 1         # 이어서 가능
    .venv/bin/python -m eval.label.fbcheck_unfair_tos --analyze
"""

import argparse
import json
import random

import numpy as np

from backend.fb_check.backward_grounding import check_snippet_exists
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("fbcheck_unfair_tos.log")
DATA = PROJECT_ROOT / "data/eval/public/unfair_tos"
RAW = PROJECT_ROOT / "data/eval/label/fbcheck_unfair_tos_raw.jsonl"
SMOKE = PROJECT_ROOT / "data/eval/label/fbcheck_unfair_tos_smoke.jsonl"
OUT = PROJECT_ROOT / "data/eval/label/fbcheck_unfair_tos.json"
SEED, BATCH, MAX_CHARS, BOOT_N, SMOKE_N = 20261015, 8, 1500, 10_000, 24

NAMES = ["Limitation of liability", "Unilateral termination", "Unilateral change", "Content removal",
         "Contract by using", "Choice of law", "Jurisdiction", "Arbitration"]
_CANON = {n.lower(): n for n in NAMES}

_CATEGORIES = """1. Limitation of liability - the provider limits or excludes its liability for damages, losses or harm.
2. Unilateral termination - the provider may suspend or terminate the service, an account or the contract at its discretion (e.g. at any time, without notice or without a stated reason).
3. Unilateral change - the provider may modify the terms or the service at any time, without the user's explicit consent.
4. Content removal - the provider may delete or remove user content at its discretion, without notice or a stated reason.
5. Contract by using - the user is bound by the terms merely by using the service or visiting the site.
6. Choice of law - the contract is governed by the law of a specified country or state.
7. Jurisdiction - disputes must be brought before specified courts (forum selection).
8. Arbitration - disputes must be resolved by arbitration instead of in court."""

_NAMES_LINE = " | ".join(NAMES)

FWD_SYSTEM = f"""You are an expert in consumer-contract law reviewing ONE sentence taken from an online Terms of Service.
Decide whether the sentence contains a clause that is potentially unfair to consumers, using ONLY these categories:

{_CATEGORIES}

Rules:
- Choose a category only if the sentence itself supports it. Do not guess from the topic alone. If none applies, return an empty list.
- "evidence_span" must be copied EXACTLY (same words, same order) from the sentence, at least 10 characters long, and must be the part that supports your categories. Use an empty string when the list is empty.
- Category names must be exactly one of: {_NAMES_LINE}
- Answer with one JSON object only: {{"categories": [...], "evidence_span": "...", "reasoning": "one short sentence"}}"""

FWD_FEWSHOT = [
    {"role": "user", "content": "Sentence:\nwe may terminate your account at any time without notice for any reason ."},
    {"role": "assistant", "content": json.dumps({"categories": ["Unilateral termination"],
                                                 "evidence_span": "terminate your account at any time without notice for any reason",
                                                 "reasoning": "The provider can end the account at any time without notice."})},
    {"role": "user", "content": "Sentence:\nyou can change your password at any time in the settings page ."},
    {"role": "assistant", "content": json.dumps({"categories": [], "evidence_span": "",
                                                 "reasoning": "This only describes a user feature, not an unfair clause."})},
    {"role": "user", "content": "Sentence:\nthese terms are governed by the laws of the state of california and any dispute shall be resolved by binding arbitration ."},
    {"role": "assistant", "content": json.dumps({"categories": ["Choice of law", "Arbitration"],
                                                 "evidence_span": "governed by the laws of the state of california and any dispute shall be resolved by binding arbitration",
                                                 "reasoning": "It names the governing law and forces arbitration."})},
]

VER_SYSTEM_SPAN = f"""You are an expert in consumer-contract law. You see ONLY a short evidence fragment taken from an online Terms of Service (not the whole sentence).
List which of these categories the fragment, on its own, clearly supports:

{_CATEGORIES}

If the fragment does not clearly support any category, return an empty list. Category names must be exactly one of: {_NAMES_LINE}
Answer with one JSON object only: {{"categories": [...]}}"""

VER_SYSTEM_CLAUSE = f"""You are an expert in consumer-contract law reviewing ONE sentence taken from an online Terms of Service.
List which of these categories the sentence clearly supports:

{_CATEGORIES}

If none applies, return an empty list. Category names must be exactly one of: {_NAMES_LINE}
Answer with one JSON object only: {{"categories": [...]}}"""

VER_FEWSHOT_SPAN = [
    {"role": "user", "content": "Evidence:\nterminate your account at any time without notice for any reason"},
    {"role": "assistant", "content": json.dumps({"categories": ["Unilateral termination"]})},
    {"role": "user", "content": "Evidence:\nyou can change your password at any time"},
    {"role": "assistant", "content": json.dumps({"categories": []})},
]


def _read_all() -> list[dict]:
    rows = []
    for sp in ("train", "validation", "test"):
        rows += load_jsonl(DATA / f"{sp}.jsonl")
    for r in rows:
        r["text"] = r["text"].strip()
        r["gold"] = sorted({NAMES[i] for i in r["labels"]})
    return rows


def population() -> tuple[list[dict], list[dict]]:
    """모집단(위반 전부 + 공정 같은 수 무작위)과 스모크용 공정 문장(모집단 밖)."""
    rows = [r for r in _read_all() if 10 <= len(r["text"]) <= MAX_CHARS]
    pos = sorted([r for r in rows if r["gold"]], key=lambda r: r["id"])
    fair = sorted([r for r in rows if not r["gold"]], key=lambda r: r["id"])
    rng = random.Random(SEED)
    pick = rng.sample(range(len(fair)), len(pos))
    chosen = set(pick)
    pop = pos + [fair[i] for i in sorted(pick)]
    rest = [fair[i] for i in range(len(fair)) if i not in chosen]
    smoke = random.Random(SEED + 1).sample(rest, SMOKE_N)
    return pop, smoke


def _cats(v) -> list[str]:
    xs = v.get("categories") if isinstance(v, dict) and isinstance(v.get("categories"), list) else []
    out = []
    for x in xs:
        if isinstance(x, str) and x.strip().lower() in _CANON:
            out.append(_CANON[x.strip().lower()])
    return list(dict.fromkeys(out))


def _gen_batch(model, tok, dev, sys_prompt: str, fewshot: list, users: list[str], max_new: int) -> list[dict | None]:
    import torch

    from eval.label.e9_local_labeling import _extract_json
    prompts = [tok.apply_chat_template([{"role": "system", "content": sys_prompt}, *fewshot,
                                        {"role": "user", "content": u}], tokenize=False, add_generation_prompt=True) + "{"
               for u in users]
    enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    return [_extract_json("{" + tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)) for seq in out]


def _process(model, tok, dev, chunk: list[dict]) -> list[dict]:
    fwd = _gen_batch(model, tok, dev, FWD_SYSTEM, FWD_FEWSHOT, [f"Sentence:\n{r['text']}" for r in chunk], 220)
    stage = []
    for r, v in zip(chunk, fwd):
        cats = _cats(v)
        span = v.get("evidence_span", "") if isinstance(v, dict) and isinstance(v.get("evidence_span"), str) else ""
        # 데이터가 소문자라 두 문자열을 모두 소문자로 맞춰 대소문자 불일치로 억울하게 탈락하는 것을 막는다(§7-28)
        stage.append({"r": r, "ok": v is not None, "f": cats, "span": span,
                      "span_ok": bool(cats) and bool(span) and check_snippet_exists(r["text"].lower(), span.lower())})
    span_jobs = [s for s in stage if s["ok"] and s["f"] and s["span_ok"]]
    clause_jobs = [s for s in stage if s["ok"] and not s["f"]]
    if span_jobs:
        for s, v in zip(span_jobs, _gen_batch(model, tok, dev, VER_SYSTEM_SPAN, VER_FEWSHOT_SPAN,
                                              [f"Evidence:\n{s['span']}" for s in span_jobs], 80)):
            s["v"] = v
    if clause_jobs:
        for s, v in zip(clause_jobs, _gen_batch(model, tok, dev, VER_SYSTEM_CLAUSE, [],
                                                [f"Sentence:\n{s['r']['text']}" for s in clause_jobs], 80)):
            s["v"] = v
    rows = []
    for s in stage:
        r = s["r"]
        row = {"id": r["id"], "label": int(bool(r["gold"])), "gold": r["gold"], "forward": s["f"], "span": s["span"]}
        if not s["ok"]:
            row.update(status="ERROR", reason="forward_parse")
        elif s["f"] and not s["span_ok"]:
            row.update(status="NOISE", reason="snippet_not_found")
        elif s.get("v") is None:
            row.update(status="ERROR", reason="verify_parse")
        else:
            va = _cats(s["v"])
            agreed = [a for a in s["f"] if a in set(va)]
            row.update(verify=va, agreed=agreed)
            if bool(s["f"]) != bool(va):
                row.update(status="NOISE", reason="label_mismatch")
            elif s["f"] and not agreed:
                row.update(status="NOISE", reason="article_mismatch")
            else:
                row.update(status="CLEAN", reason="")
        rows.append(row)
    return rows


def _load():
    from eval.label.e9_local_labeling import load_model
    model, tok, _ = load_model("qwen2.5-14b", GPU[0])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok, next(model.parameters()).device


GPU = [1]


def smoke() -> None:
    _, smoke_rows = population()
    model, tok, dev = _load()
    out = []
    with open(SMOKE, "w", encoding="utf-8") as f:
        for i in range(0, len(smoke_rows), BATCH):
            for row in _process(model, tok, dev, smoke_rows[i:i + BATCH]):
                out.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    err = sum(r["status"] == "ERROR" for r in out)
    logger.info(f"스모크 {len(out)}문장: 파싱 실패 {err} ({err / len(out) * 100:.1f}%) — 형식 점검용, 결과는 보고하지 않음")
    logger.info("형식 확인 예시: " + json.dumps(out[:2], ensure_ascii=False))


def run() -> None:
    rows, _ = population()
    done = {r["id"] for r in load_jsonl(RAW)} if RAW.exists() else set()
    todo = sorted([r for r in rows if r["id"] not in done], key=lambda r: len(r["text"]))
    logger.info(f"=== FB-Check(Qwen) on UNFAIR-ToS: 전체 {len(rows)} · 남음 {len(todo)} ===")
    if not todo:
        return
    model, tok, dev = _load()
    with open(RAW, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), BATCH):
            for row in _process(model, tok, dev, todo[i:i + BATCH]):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if (i // BATCH) % 25 == 0:
                logger.info(f"  [{i + min(BATCH, len(todo) - i)}/{len(todo)}]")


def _f1(pred: set, ref: set) -> float:
    if not pred:
        return 0.0
    return 2 * len(pred & ref) / (len(pred) + len(ref))


def stratified_gain(sc: np.ndarray, has: np.ndarray, keep: np.ndarray, key: np.ndarray, rng) -> dict:
    """CLEAN이 남긴 라벨의 평균 F1 − (forward 유형 집합별 층마다 같은 수를 무작위로 남긴 평균 F1).

    §7-28의 주 추정량. UNFAIR-ToS 전체·분할별과 AI Hub(`--analyze-aihub`)에 **같은 함수**를 쓴다(전체 실험 통일)."""
    strata = []
    for k in sorted(set(key[has])):
        idx = np.where(has & (key == k))[0]
        n_k = int(keep[idx].sum())
        if n_k > 0:
            strata.append((idx, n_k))
    n_sel = sum(n for _, n in strata)
    if not n_sel:
        return {"n_labeled": int(has.sum()), "n_kept_labeled": 0, "gain_pp": float("nan"), "perm_p_one_sided": float("nan")}
    kept_total = float(sum(sc[i][keep[i]].sum() for i, _ in strata))
    exp_total = float(sum(sc[i].mean() * n for i, n in strata))
    sums = np.zeros(BOOT_N)
    for idx, n_k in strata:
        s = sc[idx]
        if n_k >= len(idx):
            sums += s.sum()
        else:
            r = rng.random((BOOT_N, len(idx)))
            pick = np.argpartition(r, n_k - 1, axis=1)[:, :n_k]
            sums += s[pick].sum(1)
    return {"n_labeled": int(has.sum()), "n_kept_labeled": n_sel, "f1_kept_pp": round(kept_total / n_sel * 100, 2),
            "f1_random_stratified_pp": round(exp_total / n_sel * 100, 2), "gain_pp": round((kept_total - exp_total) / n_sel * 100, 2),
            "perm_p_one_sided": float((sums >= kept_total).mean()), "mde80_pp": round(float((sums / n_sel).std()) * 2.8 * 100, 2),
            "n_strata": len(strata)}


def analyze_aihub() -> None:
    """AI Hub(§7-14, Qwen) 기존 행에 **UNFAIR-ToS와 같은 추정량**을 적용한다 — LLM 재호출 없음."""
    rows = [r for r in load_jsonl(PROJECT_ROOT / "data/eval/label/fbcheck_aihub_raw.jsonl") if r["status"] != "ERROR"]
    rng = np.random.default_rng(42)
    has = np.array([bool(r["forward"]) for r in rows])
    sc = np.array([_f1(set(r["forward"]), set(r["ref"])) for r in rows])
    keep = np.array([r["status"] == "CLEAN" for r in rows])
    key = np.array([",".join(sorted(r["forward"])) for r in rows])
    rep = {"rule": "eval/paper_plan.md §7-28 개정 — AI Hub에 같은 추정량(조 집합 층화 무작위 기준선) 적용",
           "note": "AI Hub 기존 §7-14 주 결과는 단순 무작위(+7.0%p)·등급 층화(+2.9%p)였다. 여기는 UNFAIR-ToS와 같은 '라벨 집합 층화'",
           "★_stratified_by_forward_article_set": stratified_gain(sc, has, keep, key, rng)}
    save_json(rep, PROJECT_ROOT / "data/eval/label/fbcheck_aihub_p1.json")
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


def analyze() -> None:
    allrows = load_jsonl(RAW)
    errors = sum(r["status"] == "ERROR" for r in allrows)
    rows = [r for r in allrows if r["status"] != "ERROR"]
    rng = np.random.default_rng(42)
    has = np.array([bool(r["forward"]) for r in rows])
    sc = np.array([_f1(set(r["forward"]), set(r["gold"])) for r in rows])
    keep = np.array([r["status"] == "CLEAN" for r in rows])
    lab = np.array([r["label"] for r in rows])
    key = np.array([",".join(sorted(r["forward"])) for r in rows])

    # --- P1: 유형 층화 무작위 기준선 (층마다 CLEAN 개수를 유지) — 모든 데이터·분할에 같은 함수를 쓴다(§7-28 개정) ---
    p1 = stratified_gain(sc, has, keep, key, rng)
    gain = p1["gain_pp"] / 100
    verdict = ("UNFAIR-ToS에서도 FB-Check 효과 확인(사전 기준: gain>0 · p<0.01)"
               if (gain > 0 and p1["perm_p_one_sided"] < 0.01) else "미판정")
    # 데이터셋 안 일관성: 공식 분할(train/validation/test)별로 같은 추정량을 따로 계산 (판정 아님)
    split = np.array([r["id"].split("-")[0] for r in rows])
    by_split = {s: stratified_gain(sc[split == s], has[split == s], keep[split == s], key[split == s], rng)
                for s in ("train", "validation", "test")}
    signs = [v["gain_pp"] > 0 for v in by_split.values() if v["n_kept_labeled"] > 0]
    consistency = {"gain_pp_by_split": {s: v["gain_pp"] for s, v in by_split.items()},
                   "same_sign_all_splits": bool(signs) and (all(signs) or not any(signs))}

    # --- P2: 단순 무작위(§7-14 방식: 전 레코드에서 같은 수를 무작위로 남김) ---
    n_keep = int(keep.sum())
    b2 = np.empty(BOOT_N)
    for b in range(BOOT_N):
        m = np.zeros(len(rows), bool)
        m[rng.choice(len(rows), n_keep, replace=False)] = True
        b2[b] = sc[m & has].mean()
    kept_mean = float(sc[keep & has].mean())
    p2 = {"n_keep": n_keep, "f1_kept_pp": round(kept_mean * 100, 2), "f1_random_pp": round(float(b2.mean()) * 100, 2),
          "gain_pp": round((kept_mean - float(b2.mean())) * 100, 2), "perm_p": float((b2 >= kept_mean).mean())}

    # --- 문장 단위 부트스트랩 CI (문서 군집 아님) ---
    def gain_of(ix: np.ndarray) -> float:
        h, k, s, kk = has[ix], keep[ix], sc[ix], key[ix]
        num = den = 0.0
        sel = h & k
        if not sel.any():
            return float("nan")
        for g in set(kk[h]):
            m = h & (kk == g)
            n_k = int((m & k).sum())
            if n_k:
                num += s[m].mean() * n_k
                den += n_k
        return float(s[sel].mean() - num / den)
    gb = np.array([gain_of(rng.integers(0, len(rows), len(rows))) for _ in range(2000)])
    ci = [round(float(x) * 100, 2) for x in np.nanpercentile(gb, [2.5, 97.5])]

    # --- 상수 기준선: 위반 문장에서 항상 가장 흔한 유형 하나 ---
    pos = lab == 1
    from collections import Counter
    top = Counter(g for r in rows if r["label"] for g in r["gold"]).most_common(1)[0][0]
    const_f1 = float(np.mean([_f1({top}, set(r["gold"])) for r in rows if r["label"]]))
    fwd_f1_pos_all = float(sc[pos].mean())
    fwd_f1_pos_has = float(sc[pos & has].mean()) if (pos & has).any() else float("nan")
    const = {"constant_category": top, "constant_f1_pos_pp": round(const_f1 * 100, 2),
             "forward_f1_pos_all_pp(empty=0)": round(fwd_f1_pos_all * 100, 2), "forward_f1_pos_labeled_only_pp": round(fwd_f1_pos_has * 100, 2),
             "forward_below_constant": bool(fwd_f1_pos_all < const_f1)}

    # --- 보고 ---
    noise = ~keep
    fp, tp = (lab == 0) & has, (lab == 1) & has
    reasons = {}
    for r in rows:
        if r["status"] != "CLEAN":
            reasons[r.get("reason") or r["status"]] = reasons.get(r.get("reason") or r["status"], 0) + 1

    def label_pr(sel_fn):
        att = ok = gold_n = hit = 0
        for r in rows:
            if not r["label"]:
                continue
            f = set(r["forward"]) if sel_fn(r) else set()
            g = set(r["gold"])
            att += len(f)
            ok += len(f & g)
            gold_n += len(g)
            hit += len(f & g)
        return {"precision_pp": round(ok / att * 100, 1) if att else None, "recall_pp": round(hit / gold_n * 100, 1)}
    f_nofilter = float(np.mean([_f1(set(r["forward"]), set(r["gold"])) for r in rows if r["label"]]))
    f_filter = float(np.mean([_f1(set(r["forward"]) if r["status"] == "CLEAN" else set(), set(r["gold"])) for r in rows if r["label"]]))
    per_cat = {}
    for c in NAMES:
        m = [r for r in rows if c in r["forward"]]
        per_cat[c] = {"n_forward": len(m), "clean_pct": round(sum(r["status"] == "CLEAN" for r in m) / len(m) * 100, 1) if m else None}
    rep = {"rule": "eval/paper_plan.md §7-28", "n": {"records": len(allrows), "scored_records": len(rows), "errors": errors,
                                                    "violations": int(lab.sum()), "fair": int((lab == 0).sum()),
                                                    "forward_nonempty": int(has.sum()), "clean": n_keep},
           "★_P1_stratified": p1, "verdict": verdict, "dataset_consistency_by_split": consistency, "P2_simple_random": p2, "bootstrap_ci95_pp_sentence_level": ci,
           "constant_baseline": const,
           "fp_labels": {"n_fair_with_label": int(fp.sum()), "removed_pct": round(float(noise[fp].mean()) * 100, 1) if fp.any() else None,
                         "n_violation_with_label": int(tp.sum()), "violation_labels_removed_pct": round(float(noise[tp].mean()) * 100, 1) if tp.any() else None},
           "noise_reasons": reasons,
           "tradeoff_violations_only": {"no_filter_f1_pp": round(f_nofilter * 100, 2), "filter_as_empty_f1_pp": round(f_filter * 100, 2),
                                        "no_filter": label_pr(lambda r: True), "filter_as_empty": label_pr(lambda r: r["status"] == "CLEAN")},
           "per_category": per_cat}
    save_json(rep, OUT)
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--analyze-aihub", action="store_true")
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()
    GPU[0] = args.gpu
    if args.smoke:
        smoke()
    if args.run:
        run()
    if args.analyze:
        analyze()
    if args.analyze_aihub:
        analyze_aihub()


if __name__ == "__main__":
    main()
