# eval/label/fbcheck_aihub.py
"""FB-Check를 조항 단위 전문가 정답(AI Hub)에서 시험 — Qwen (사전 등록: eval/paper_plan.md §7-14, 2026-10-08, API 0)

운영 FB-Check와 같은 프롬프트·판정 규칙을 Qwen2.5-14B로 돌린다. 생성은 8건씩 묶어서 한다.

    .venv/bin/python -m eval.label.fbcheck_aihub --run --gpu 0     # forward → verify (캐시, 이어서 가능)
    .venv/bin/python -m eval.label.fbcheck_aihub --analyze
"""

import argparse
import json
import random

import numpy as np

from backend.fb_check.backward_grounding import check_snippet_exists
from backend.fb_check.consistency_verification import _FEW_SHOT_EXAMPLES as VERIFY_FEWSHOT
from backend.fb_check.consistency_verification import _SYSTEM_CLAUSE, _SYSTEM_SPAN, MAX_VERIFY_CHARS
from backend.fb_check.forward_labeling import _FEW_SHOT_EXAMPLES as FWD_FEWSHOT
from backend.fb_check.forward_labeling import _SYSTEM as FWD_SYSTEM
from backend.labeling.articles import ARTICLES
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json
from eval.model.aihub_eval import testset

logger = load_logger("fbcheck_aihub.log")
RAW = PROJECT_ROOT / "data/eval/label/fbcheck_aihub_raw.jsonl"
OUT = PROJECT_ROOT / "data/eval/label/fbcheck_aihub.json"
SEED, BATCH, MAX_CHARS, BOOT_N = 20261008, 8, 1500, 10_000


def population() -> list[dict]:
    items = [it for it in testset() if len(it["text"]) <= MAX_CHARS]
    pos = [it for it in items if it["label"] == 1]
    neg = [it for it in items if it["label"] == 0]
    neg = random.Random(SEED).sample(neg, min(len(pos), len(neg)))
    rows = pos + neg
    for i, r in enumerate(rows):
        r["key"] = f"{r['group']}#{i}"
    return rows


def _arts(v) -> list[str]:
    xs = v.get("articles") if isinstance(v, dict) and isinstance(v.get("articles"), list) else []
    return list(dict.fromkeys(a for a in xs if isinstance(a, str) and a in ARTICLES))


def _label(v, arts) -> str:
    lab = v.get("risk_level") if isinstance(v, dict) and isinstance(v.get("risk_level"), str) else "Low"
    return (lab or "Low") if arts else "Low"


def _gen_batch(model, tok, dev, sys_prompt: str, fewshot: list, users: list[str], max_new: int) -> list[dict | None]:
    import torch

    from eval.label.e9_local_labeling import _extract_json
    # 운영 GPT는 response_format=json_object로 JSON만 낸다. Qwen은 그 장치가 없어 조각 조항에서 줄글로 답하므로
    # (첫 56건 중 26.8% 파싱 실패, 2026-10-08) 답변을 "{"로 시작하게 고정해 같은 조건을 만든다
    prompts = [tok.apply_chat_template([{"role": "system", "content": sys_prompt}, *fewshot,
                                        {"role": "user", "content": u}], tokenize=False, add_generation_prompt=True) + "{"
               for u in users]
    enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    return [_extract_json("{" + tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)) for seq in out]


def run(gpu: int) -> None:
    from eval.label.e9_local_labeling import load_model
    rows = population()
    done = {r["key"] for r in load_jsonl(RAW)} if RAW.exists() else set()
    todo = sorted([r for r in rows if r["key"] not in done], key=lambda r: len(r["text"]))
    logger.info(f"=== FB-Check(Qwen) on AI Hub: 전체 {len(rows)} · 남음 {len(todo)} ===")
    if not todo:
        return
    model, tok, _ = load_model("qwen2.5-14b", gpu)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = next(model.parameters()).device
    with open(RAW, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            fwd = _gen_batch(model, tok, dev, FWD_SYSTEM, FWD_FEWSHOT, [f"계약 조항:\n{r['text'][:3000]}" for r in chunk], 300)
            stage = []
            for r, v in zip(chunk, fwd):
                arts = _arts(v)
                span = v.get("evidence_span", "") if isinstance(v, dict) and isinstance(v.get("evidence_span"), str) else ""
                stage.append({"r": r, "ok": v is not None, "f_arts": arts, "f_label": _label(v, arts), "span": span,
                              "span_ok": bool(arts) and bool(span) and check_snippet_exists(r["text"], span)})
            # verify: 조가 있고 근거가 원문에 있으면 span 모드, 조가 없으면 clause 모드. 근거 없음은 NOISE로 끝(운영과 같음)
            span_jobs = [s for s in stage if s["ok"] and s["f_arts"] and s["span_ok"]]
            clause_jobs = [s for s in stage if s["ok"] and not s["f_arts"]]
            if span_jobs:
                for s, v in zip(span_jobs, _gen_batch(model, tok, dev, _SYSTEM_SPAN, VERIFY_FEWSHOT,
                                                      [f"근거 문구:\n{s['span']}" for s in span_jobs], 200)):
                    s["v"] = v
            if clause_jobs:
                for s, v in zip(clause_jobs, _gen_batch(model, tok, dev, _SYSTEM_CLAUSE, [],
                                                        [f"계약 조항:\n{s['r']['text'][:MAX_VERIFY_CHARS]}" for s in clause_jobs], 200)):
                    s["v"] = v
            for s in stage:
                r = s["r"]
                row = {"key": r["key"], "label": r["label"], "ref": r["articles"], "forward": s["f_arts"], "f_label": s["f_label"]}
                if not s["ok"]:
                    row.update(status="ERROR", reason="forward_parse")
                elif s["f_arts"] and not s["span_ok"]:
                    row.update(status="NOISE", reason="snippet_not_found")
                elif s.get("v") is None:
                    row.update(status="ERROR", reason="verify_parse")
                else:
                    va = _arts(s["v"])
                    vl = _label(s["v"], va)
                    agreed = [a for a in s["f_arts"] if a in set(va)]
                    row.update(verify=va, v_label=vl, agreed=agreed)
                    if s["f_label"] != vl:
                        row.update(status="NOISE", reason="label_mismatch")
                    elif s["f_label"] != "Low" and not agreed:
                        row.update(status="NOISE", reason="article_mismatch")
                    else:
                        row.update(status="CLEAN", reason="")
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if (i // BATCH) % 25 == 0:
                logger.info(f"  [{i + len(chunk)}/{len(todo)}]")


def _f1(pred: set, ref: set) -> float:
    if not pred:
        return 0.0
    i = len(pred & ref)
    return 2 * i / (len(pred) + len(ref)) if (pred or ref) else 1.0


def analyze() -> None:
    rows = load_jsonl(RAW)
    rng = np.random.default_rng(42)
    has = np.array([bool(r["forward"]) for r in rows])
    sc = np.array([_f1(set(r["forward"]), set(r["ref"])) for r in rows])
    keep = np.array([r["status"] == "CLEAN" for r in rows])
    lab = np.array([r["label"] for r in rows])

    def perm(mask_pop: np.ndarray) -> dict:
        idx = np.where(mask_pop)[0]
        s, h, k = sc[idx], has[idx], keep[idx]
        n_keep = int(k.sum())
        f = float(s[k & h].mean())
        boots = np.empty(BOOT_N)
        for b in range(BOOT_N):
            m = np.zeros(len(idx), bool)
            m[rng.choice(len(idx), n_keep, replace=False)] = True
            boots[b] = s[m & h].mean()
        return {"n_pop": int(len(idx)), "n_keep": n_keep, "n_scored_kept": int((k & h).sum()),
                "f1_all_pp": round(float(s[h].mean()) * 100, 2), "f1_kept_pp": round(f * 100, 2),
                "f1_random_pp": round(float(boots.mean()) * 100, 2), "gain_pp": round((f - float(boots.mean())) * 100, 2),
                "perm_p": float((boots >= f).mean()), "mde80_pp": round(float(boots.std()) * 2.8 * 100, 2)}

    allpop = np.ones(len(rows), bool)
    main = perm(allpop)
    verdict = "FB-Check 효과 확인 (조항 단위 전문가 정답, Qwen)" if (main["gain_pp"] > 0 and main["perm_p"] < 0.05) else "미판정"

    # ① 공정 조항에 붙은 잘못된 위반 라벨을 FB-Check가 걸러내는 비율 vs 위반 조항 라벨이 걸러지는 비율
    fp = (lab == 0) & has
    tp = (lab == 1) & has
    noise = ~keep
    fp_removed = float(noise[fp].mean()) if fp.any() else float("nan")
    tp_removed = float(noise[tp].mean()) if tp.any() else float("nan")
    # 무작위 기대: 라벨 있는 레코드 전체의 NOISE 비율
    exp_removed = float(noise[has].mean())
    reasons = {}
    for r in rows:
        if r["status"] != "CLEAN":
            reasons[r.get("reason") or r["status"]] = reasons.get(r.get("reason") or r["status"], 0) + 1
    # ④ gold §25 방식: 거른 라벨을 빈 예측으로 셀 때(위반 조항만), 필터 없음 대비
    pos_idx = lab == 1
    f_nofilter = float(np.mean([_f1(set(r["forward"]), set(r["ref"])) for r, p in zip(rows, pos_idx) if p]))
    f_filter = float(np.mean([_f1(set(r["forward"]) if r["status"] == "CLEAN" else set(), set(r["ref"]))
                              for r, p in zip(rows, pos_idx) if p]))
    rep = {"rule": "eval/paper_plan.md §7-14", "n": {"total": len(rows), "violations": int(lab.sum()), "fair": int((lab == 0).sum()),
                                                    "forward_nonempty": int(has.sum()), "clean": int(keep.sum()),
                                                    "errors": sum(r["status"] == "ERROR" for r in rows)},
           "★_main_kept_vs_random": main, "verdict": verdict,
           "②_violations_only": perm(lab == 1),
           "①_false_positive_labels": {"n_fair_with_label": int(fp.sum()), "removed_by_fbcheck_pct": round(fp_removed * 100, 1),
                                       "n_violation_with_label": int(tp.sum()), "violation_labels_removed_pct": round(tp_removed * 100, 1),
                                       "overall_removed_pct": round(exp_removed * 100, 1)},
           "③_noise_reasons": reasons,
           "④_gold_style_violations": {"no_filter_f1_pp": round(f_nofilter * 100, 2), "fbcheck_as_empty_f1_pp": round(f_filter * 100, 2)}}
    save_json(rep, OUT)
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    if args.run:
        run(args.gpu)
    if args.analyze:
        analyze()


if __name__ == "__main__":
    main()
