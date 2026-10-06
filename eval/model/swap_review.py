# eval/model/swap_review.py
"""배포 모델 교체 검토 — article_v2(KoELECTRA) → KLUE-RoBERTa (2026-10-05)

판정 규칙은 `eval/paper_plan.md` §7에 **실행 전에** 적은 그대로다. 이 스크립트는 그 규칙을 코드로 옮긴 것이고,
규칙을 바꾸려면 문서부터 고친다.

    ① 무해      gold 조항 재현 · 홀드아웃 오경보 (`retrain.evaluate_pair` + `harm_verdict` 그대로)
    ② 교체 이유  paraphrase 위반 조항 재현 하락폭 — 후보가 현행보다 유의하게 덜 떨어져야 한다
    ③ 부작용    paraphrase 표준계약서 · B4 문체 변환 273건 · 온전 지층 908 오경보 — 5%p 넘게 높고 CI 하한 > 0이면 보류
    ④ 행동 변화  실제 기업 약관 지목률 (보고만)
    ⑤ 서빙      토크나이저 호환 · CPU 지연·메모리

실행:
    .venv/bin/python -m eval.model.swap_review --gpu 0                 # 원 후보 RB_s42
    .venv/bin/python -m eval.model.swap_review --gpu 0 --cand ens      # 재시도 (b) 3시드 앙상블
    .venv/bin/python -m eval.model.swap_review --gpu 0 --cand rsn_ens  # 말투 편향 줄이기 통과 팔 (§7-2 4번)

후보는 모델 디렉터리 하나이거나, 디렉터리 목록(= 확률 평균 앙상블, 임계값은 조별 평균)이다.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from backend.training.retrain import evaluate_pair, harm_verdict, paired_ci, predict
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("swap_review.log")

BASE    = PROJECT_ROOT / "models/article_v2"
CAND    = PROJECT_ROOT / "models/_e8_RB_s42"          # dev 최고 시드 (paper_plan §7)
CONTROL = PROJECT_ROOT / "models/_e8_C_s7"            # KoELECTRA · 현행 데이터, dev 최고 시드
EXTRA   = [PROJECT_ROOT / "models/_e8_RB_s7", PROJECT_ROOT / "models/_e8_RB_s123"]
HOLDOUT = PROJECT_ROOT / "backend/training/negative_holdout_docs.json"
PARA    = PROJECT_ROOT / "data/eval/paraphrase_raw.jsonl"
B4      = PROJECT_ROOT / "data/eval/style_transfer_fpr.cache.json"
REAL    = PROJECT_ROOT / "data/eval/real_tos"
OUT     = PROJECT_ROOT / "data/eval/swap_review.json"
NOISE_FLOOR_PP = 5.0


RB_SEEDS = [PROJECT_ROOT / f"models/_e8_RB_s{s}" for s in (42, 7, 123)]
RSN_SEEDS = [PROJECT_ROOT / f"models/_e8_RSN_s{s}" for s in (42, 7, 123)]
# rsn_ens: 말투 편향 줄이기(§7-2)를 통과한 RoBERTa + 회사 말투 음성, 3시드 앙상블 — 사전 등록 4번
CANDS = {"s42": CAND, "ens": RB_SEEDS, "rsn_ens": RSN_SEEDS}


def predict_spec(spec, texts: list[str], gpu: int):
    """spec이 목록이면 앙상블: sigmoid 확률 평균 + dev 임계값 조별 평균 (재시도 사전 등록 (b))."""
    if isinstance(spec, Path):
        return predict(spec, texts, gpu)
    outs = [predict(d, texts, gpu) for d in spec]
    names = outs[0][2]
    assert all(o[2] == names for o in outs), "앙상블 멤버의 조 목록이 다르다"
    return np.mean([o[0] for o in outs], axis=0), np.mean([o[1] for o in outs], axis=0), names


def _flags(spec, texts: list[str], gpu: int) -> np.ndarray:
    P, thr, _ = predict_spec(spec, texts, gpu)
    return (P >= thr).any(1)


def evaluate_pair_spec(base: Path, cand, holdout: list[str], gpu: int) -> dict:
    """`retrain.evaluate_pair`와 같은 계산을 spec(앙상블 포함)으로."""
    from backend.training.retrain import eval_sets, f1_rows
    gold, neg = eval_sets(holdout)
    groups = [":".join(str(r["chunk_id"]).split(":")[:2]) for r in neg]
    rows, out = {}, {"n_gold": len(gold), "n_holdout": len(neg)}
    for tag, spec in (("base", base), ("cand", cand)):
        P, thr, names = predict_spec(spec, [g["text"] for g in gold], gpu)
        Pn, *_ = predict_spec(spec, [r["text"] for r in neg], gpu)
        lab = np.array([[a in g["articles"] for a in names] for g in gold])
        rows[tag] = {"recall": (P >= thr).any(1).astype(float), "fpr": (Pn >= thr).any(1).astype(float),
                     "f1": f1_rows(P >= thr, lab)}
        out[tag] = {k: float(v.mean()) for k, v in (("gold_recall", rows[tag]["recall"]),
                                                     ("holdout_fpr", rows[tag]["fpr"]), ("gold_f1", rows[tag]["f1"]))}
    b, c = rows["base"], rows["cand"]
    out["recall_diff_ci_pp"] = paired_ci(c["recall"] - b["recall"])
    out["fpr_diff_ci_pp"] = paired_ci(c["fpr"] - b["fpr"], groups)
    out["f1_diff_ci_pp"] = paired_ci(c["f1"] - b["f1"])
    return out


def _cmp(cand: np.ndarray, base: np.ndarray, groups: list[str]) -> dict:
    d, lo, hi = paired_ci(cand.astype(float) - base.astype(float), groups)
    return {"base": round(float(base.mean()) * 100, 2), "cand": round(float(cand.mean()) * 100, 2),
            "diff_pp": round(d, 2), "ci95_pp": [round(lo, 2), round(hi, 2)]}


def _side_effect(c: dict) -> bool:
    """③ 보류 조건: 5%p 넘게 높고 CI 하한 > 0."""
    return c["diff_pp"] > NOISE_FLOOR_PP and c["ci95_pp"][0] > 0


# ── ② ③ paraphrase ───────────────────────────────────────────────────────────

def paraphrase_axes(models: dict[str, Path], gpu: int) -> dict:
    from eval.channel.paraphrase_consistency import _items
    items = {r["key"]: r for r in _items()}
    rows = [{**items[c["key"]], "para": c["paraphrase"]} for c in load_jsonl(PARA)
            if c.get("paraphrase") and c["quality"]["pass"]]
    out, fl = {"n_pairs": len(rows)}, {}
    for name, d in models.items():
        fo = _flags(d, [r["text"] for r in rows], gpu)
        fp = _flags(d, [r["para"] for r in rows], gpu)
        fl[name] = (fo, fp)
    gold = np.array([r["kind"] == "gold" for r in rows])
    neg = ~gold
    groups = [r["group"] for r in rows]
    g_groups = [g for g, m in zip(groups, gold) if m]
    n_groups = [g for g, m in zip(groups, neg) if m]
    for name, (fo, fp) in fl.items():
        out[name] = {"gold_recall_orig": round(float(fo[gold].mean()) * 100, 2),
                     "gold_recall_para": round(float(fp[gold].mean()) * 100, 2),
                     "neg_fpr_para": round(float(fp[neg].mean()) * 100, 2)}
    # ② 하락폭 차: (후보 변화) − (현행 변화). 양수 = 후보가 덜 떨어진다
    b, c = fl["base"], fl["cand"]
    change_b = b[1][gold].astype(float) - b[0][gold].astype(float)
    change_c = c[1][gold].astype(float) - c[0][gold].astype(float)
    d, lo, hi = paired_ci(change_c - change_b, g_groups)
    out["②_drop_gap"] = {"base_change_pp": round(float(change_b.mean()) * 100, 2),
                         "cand_change_pp": round(float(change_c.mean()) * 100, 2),
                         "cand_minus_base_pp": round(d, 2), "ci95_pp": [round(lo, 2), round(hi, 2)],
                         "pass": lo > 0}
    out["③_para_neg_fpr"] = _cmp(c[1][neg], b[1][neg], n_groups)
    # 참고: 원문 위반 재현이 둘 다 맞힌 것 중 변형에서 놓친 비율(조건부 하락)
    for name, (fo, fp) in fl.items():
        m = gold & fo
        out[name]["miss_after_para_given_hit"] = round(float((~fp[m]).mean()) * 100, 2)
    return out


# ── ③ B4 · 온전 지층 ─────────────────────────────────────────────────────────

def b4_axis(models: dict[str, Path], gpu: int) -> dict:
    from backend.training.train_article import load_article_records, split_negative_holdout
    recs = load_article_records(PROJECT_ROOT / "data/fb_check/clean.jsonl", label_source="agreed")
    _, h = split_negative_holdout(recs, 12, 42)
    clauses = [r for r in h if r.get("source") == "standard_contract" and not r.get("articles")]
    xfer = json.loads(B4.read_text(encoding="utf-8"))
    assert len(xfer) == len(clauses), "B4 캐시와 원문 개수가 다르다"
    groups = [r["group"] for r in clauses]
    out = {"n": len(clauses)}
    fl = {name: (_flags(d, [r["text"] for r in clauses], gpu), _flags(d, xfer, gpu)) for name, d in models.items()}
    for name, (fo, fx) in fl.items():
        out[name] = {"fpr_orig": round(float(fo.mean()) * 100, 2), "fpr_xfer": round(float(fx.mean()) * 100, 2)}
    out["③_b4_xfer_fpr"] = _cmp(fl["cand"][1], fl["base"][1], groups)
    return out


def intact_axis(models: dict[str, Path], gpu: int) -> dict:
    from eval.channel.gpt_channel_fpr import _doc, intact_stratum, load_negative_pool
    pool = intact_stratum(load_negative_pool())
    groups = [_doc(r["chunk_id"]) for r in pool]
    held = set(json.loads(HOLDOUT.read_text(encoding="utf-8")))
    # ⚠ 오염 측정: 이 풀은 학습 음성과 같은 코퍼스다. 후보·대조군은 홀드아웃 12문서 밖을 전부 학습했다
    in_train = float(np.mean([g not in held for g in groups]))
    fl = {name: _flags(d, [r["text"] for r in pool], gpu) for name, d in models.items()}
    out = {"n": len(pool), "share_outside_holdout_docs": round(in_train, 4),
           "note": "현행(article_v2, 옛 데이터 963건)과 후보(현행 데이터)가 학습에서 본 표준계약서 범위가 달라 이 축은 암기 정도 차이가 섞인다"}
    for name, f in fl.items():
        out[name] = round(float(f.mean()) * 100, 2)
    out["③_intact_fpr"] = _cmp(fl["cand"], fl["base"], groups)
    return out


# ── ④ 실제 약관 ──────────────────────────────────────────────────────────────

def real_tos_axis(models: dict[str, Path], gpu: int) -> dict:
    out = {}
    for f in sorted(REAL.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        texts = [c["text"] for c in d["clauses"]]
        if not texts:
            continue
        out[f.stem] = {"n": len(texts), **{name: round(float(_flags(m, texts, gpu).mean()) * 100, 1)
                                           for name, m in models.items()}}
    allb = np.mean([v["base"] for v in out.values()])
    allc = np.mean([v["cand"] for v in out.values()])
    out["_mean"] = {"base": round(float(allb), 1), "cand": round(float(allc), 1),
                    "warn": bool(abs(allc - allb) > 10)}
    return out


# ── ⑤ 서빙 ──────────────────────────────────────────────────────────────────

def serving_axis(texts: list[str], cand) -> dict:
    import torch
    from transformers import AutoTokenizer, ElectraTokenizerFast

    from backend.model.electra import ArticleMultiLabelElectra
    out = {}
    # 판정 경로(judgment_agent)는 ElectraTokenizerFast로 고정해 읽는다 — 후보 체크포인트에서도 같은 토큰이 나오나
    first = cand if isinstance(cand, Path) else cand[0]
    a = ElectraTokenizerFast.from_pretrained(str(first))
    b = AutoTokenizer.from_pretrained(str(first))
    same = sum(a(t, truncation=True, max_length=512)["input_ids"] == b(t, truncation=True, max_length=512)["input_ids"]
               for t in texts)
    out["tokenizer_identical"] = {"same": same, "n": len(texts), "auto_class": type(b).__name__}
    torch.set_num_threads(4)
    for name, spec in (("base", BASE), ("cand", cand)):
        dirs = [spec] if isinstance(spec, Path) else spec
        tok = AutoTokenizer.from_pretrained(str(dirs[0]))
        t0 = time.time()
        ms = [ArticleMultiLabelElectra.load(d).eval() for d in dirs]
        load_s = time.time() - t0
        lat = []
        with torch.no_grad():
            for t in texts[:40]:
                e = tok(t, truncation=True, max_length=512, return_tensors="pt")
                s = time.time()
                for m in ms:
                    m(e["input_ids"], e["attention_mask"], e.get("token_type_ids", torch.zeros_like(e["input_ids"])))
                lat.append(time.time() - s)
        out[name] = {"n_models": len(ms), "load_s": round(load_s, 2),
                     "cpu_ms_per_clause_median": round(float(np.median(lat)) * 1000, 1),
                     "params_m": round(sum(p.numel() for m in ms for p in m.parameters()) / 1e6, 1)}
        del ms
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cand", choices=list(CANDS), default="s42")
    args = ap.parse_args()
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    cand = CANDS[args.cand]
    models = {"base": BASE, "cand": cand, "control": CONTROL}
    out_path = OUT if args.cand == "s42" else OUT.with_name(f"swap_review_{args.cand}.json")

    rep = {"base": str(BASE.relative_to(PROJECT_ROOT)), "cand": str(cand),
           "control": str(CONTROL.relative_to(PROJECT_ROOT)), "rules": "eval/paper_plan.md §7 (사전 등록)"}
    ev = evaluate_pair_spec(BASE, cand, holdout, args.gpu)
    rep["①_harm"] = {"eval": ev, "verdict": harm_verdict(ev["recall_diff_ci_pp"], ev["fpr_diff_ci_pp"])}
    rep["①_control_vs_base"] = evaluate_pair(BASE, CONTROL, holdout, args.gpu)
    if args.cand == "s42":
        rep["①_extra_seeds"] = {d.name: evaluate_pair(BASE, d, holdout, args.gpu) for d in EXTRA}
    rep["paraphrase"] = paraphrase_axes(models, args.gpu)
    rep["b4"] = b4_axis(models, args.gpu)
    rep["intact"] = intact_axis(models, args.gpu)
    rep["④_real_tos"] = real_tos_axis({"base": BASE, "cand": cand}, args.gpu)
    from backend.training.retrain import eval_sets
    gold, _ = eval_sets(holdout)
    rep["⑤_serving"] = serving_axis([g["text"] for g in gold][:120], cand)

    side = {"para_neg": rep["paraphrase"]["③_para_neg_fpr"], "b4": rep["b4"]["③_b4_xfer_fpr"],
            "intact": rep["intact"]["③_intact_fpr"]}
    holds = [k for k, v in side.items() if _side_effect(v)]
    passed = rep["①_harm"]["verdict"]["passed"] and rep["paraphrase"]["②_drop_gap"]["pass"] and not holds
    rep["decision"] = {"①_pass": rep["①_harm"]["verdict"]["passed"], "②_pass": rep["paraphrase"]["②_drop_gap"]["pass"],
                       "③_holds": holds, "recommend_swap": passed}
    save_json(rep, out_path)
    logger.info(json.dumps(rep["decision"], ensure_ascii=False))
    logger.info(f"저장: {out_path}")


if __name__ == "__main__":
    main()
