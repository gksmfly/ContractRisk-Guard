# eval/gold/aihub_checks.py
"""AI Hub를 외부 평가셋으로 쓰기 전 두 가지 확인 (2026-10-07, API 0)

1. 정의 일치 — AI Hub 전문가 조문(relateLaword)과 공정위 근거_법령이 같은 조항에서 얼마나 같은가.
   AI Hub 조항과 우리 공정위 레코드가 공백·기호 제거 30자 조각을 50% 이상 공유하면 같은 조항으로 짝짓는다.
   gold clean 255(사건당 근거 1개 = 조항 귀속이 깨끗한 층)와 짝지은 것을 주 수치로, 공정위 전체(사건 단위 근거)를 참고로 낸다.
2. 비슷한 사건 누수 — 겹침을 뺀 AI Hub 조항이 우리 학습 데이터와 의미상 얼마나 가까운가(KoE5 최대 코사인).
   기준선: §7-3에서 정식으로 떼어 둔 새 시험 문서(공정위 위반 428 · 표준계약서 360)가 그 분할의 학습 데이터와 가까운 정도.

    .venv/bin/python -m eval.gold.aihub_checks --gpu 0
"""

import argparse
import glob
import json
import os
import re
from collections import Counter

import numpy as np

from backend.labeling.articles import ARTICLE_IDS
from backend.training.train_article import load_ftc_gold
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json
from eval.label.label_ablation import _ref_by_doc

logger = load_logger("aihub_checks.log")
OUT = PROJECT_ROOT / "data/eval/gold/aihub_checks.json"
_norm = lambda t: re.sub(r"[\s\W_]+", "", t or "")  # noqa: E731


def _sh(t: str, k: int = 30, step: int = 1) -> set[str]:
    t = _norm(t)
    return {t[i:i + k] for i in range(0, max(1, len(t) - k + 1), step)}


def aihub_items() -> list[dict]:
    out = []
    for f in sorted({os.path.realpath(x) for x in glob.glob(str(PROJECT_ROOT / "data/raw/aihub_tos/**/*.json"), recursive=True)}):
        d = json.load(open(f, encoding="utf-8"))
        arts = sorted({f"제{m}조" for law in d.get("relateLaword", []) for m in re.findall(r"제(\d+)조", law)
                       if f"제{m}조" in ARTICLE_IDS})
        for t in d.get("clauseArticle", []):
            if len((t or "").strip()) >= 30:
                out.append({"text": t, "adv": d.get("dvAntageous"), "articles": arts})
    return out


def definition_check(items: list[dict]) -> dict:
    pos = [it for it in items if it["adv"] == "2" and it["articles"]]
    gold = load_ftc_gold("clean")
    ref = _ref_by_doc()
    ftc = [r for p in ("clean", "noise") for r in load_jsonl(PROJECT_ROOT / f"data/fb_check/{p}.jsonl")
           if r.get("source") == "ftc_case"]

    def match(targets: list[tuple[str, set]]) -> list[tuple[set, set]]:
        index: dict[str, list[int]] = {}
        shs = [_sh(t) for t, _ in targets]
        for j, s in enumerate(shs):
            for x in s:
                index.setdefault(x, []).append(j)
        pairs = []
        for it in pos:
            a = _sh(it["text"])
            cnt = Counter(j for x in a for j in index.get(x, ()))
            if not cnt:
                continue
            j, c = cnt.most_common(1)[0]
            if c / max(min(len(a), len(shs[j])), 1) >= 0.5:
                pairs.append((set(it["articles"]), set(targets[j][1])))
        return pairs

    def agree(pairs):
        if not pairs:
            return {"n": 0}
        exact = np.mean([a == b for a, b in pairs])
        jac = np.mean([len(a & b) / len(a | b) for a, b in pairs])
        any_overlap = np.mean([bool(a & b) for a, b in pairs])
        ftc_in_aihub = np.mean([b <= a for a, b in pairs])
        extra = Counter(x for a, b in pairs for x in a - b)
        missing = Counter(x for a, b in pairs for x in b - a)
        return {"n": len(pairs), "exact_pct": round(exact * 100, 1), "jaccard_pct": round(jac * 100, 1),
                "any_overlap_pct": round(any_overlap * 100, 1), "ftc_subset_of_aihub_pct": round(ftc_in_aihub * 100, 1),
                "aihub_mean_n": round(float(np.mean([len(a) for a, _ in pairs])), 2),
                "ftc_mean_n": round(float(np.mean([len(b) for _, b in pairs])), 2),
                "aihub_extra_top": dict(extra.most_common(5)), "aihub_missing_top": dict(missing.most_common(5))}

    gold_pairs = match([(g["text"], set(g["articles"])) for g in gold])
    ftc_pairs = match([(r["text"], ref[":".join(r["chunk_id"].split(":")[:2])]) for r in ftc
                       if ":".join(r["chunk_id"].split(":")[:2]) in ref])
    return {"vs_gold_clean_255 (주)": agree(gold_pairs), "vs_ftc_all_case_level (참고)": agree(ftc_pairs)}


def leakage_check(items: list[dict], gpu: int) -> dict:
    from sentence_transformers import SentenceTransformer
    ours_rows = [r for p in ("clean", "noise") for r in load_jsonl(PROJECT_ROOT / f"data/fb_check/{p}.jsonl")]
    ours = [r["text"] for r in ours_rows] + [g["text"] for g in load_ftc_gold("all")]
    banned = set().union(*(_sh(t) for t in ours))
    usable = [it for it in items if not (_sh(it["text"], step=3) & banned)]
    pos_u = [it["text"] for it in usable if it["adv"] == "2" and it["articles"]]
    neg_u = [it["text"] for it in usable if it["adv"] == "1"]

    fresh = PROJECT_ROOT / "data/eval/fresh"
    train_nc = [r["text"] for r in load_jsonl(fresh / "labels_NC.jsonl")]
    f_pos = [r["text"] for r in load_jsonl(fresh / "test_pos.jsonl")]
    f_neg = [r["text"] for r in load_jsonl(fresh / "test_neg.jsonl")]
    train_all = [r["text"] for r in load_jsonl(PROJECT_ROOT / "data/fb_check/clean.jsonl")]

    m = SentenceTransformer("nlpai-lab/KoE5", device=f"cuda:{gpu}")
    enc = lambda xs: m.encode([f"passage: {x}" for x in xs], batch_size=64, normalize_embeddings=True,  # noqa: E731
                              show_progress_bar=False)
    E_tr_all, E_tr_nc = enc(train_all), enc(train_nc)

    def nn_stats(X: list[str], T) -> dict:
        E = enc(X)
        mx = (E @ T.T).max(1)
        return {"n": len(X), "median": round(float(np.median(mx)), 4), "p90": round(float(np.percentile(mx, 90)), 4),
                "share_ge_0.95": round(float((mx >= 0.95).mean()) * 100, 1),
                "share_ge_0.90": round(float((mx >= 0.90).mean()) * 100, 1)}

    return {"usable": {"pos": len(pos_u), "neg": len(neg_u)},
            "aihub_pos_vs_train(clean.jsonl)": nn_stats(pos_u, E_tr_all),
            "aihub_neg_vs_train(clean.jsonl)": nn_stats(neg_u, E_tr_all),
            "baseline_fresh_pos_vs_its_train(NC)": nn_stats(f_pos, E_tr_nc),
            "baseline_fresh_neg_vs_its_train(NC)": nn_stats(f_neg, E_tr_nc),
            "note": "최대 코사인이 기준선(정식 문서 분할)보다 높지 않으면 '비슷한 사건 누수'는 정식 분할 수준"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    items = aihub_items()
    rep = {"definition": definition_check(items), "leakage": leakage_check(items, args.gpu)}
    save_json(rep, OUT)
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
