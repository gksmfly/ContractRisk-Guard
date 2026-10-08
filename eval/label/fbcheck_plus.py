# eval/label/fbcheck_plus.py
"""FB-Check 향상 — 탐색 절반에서 고르고 확인 절반에서 판정 (사전 등록: eval/paper_plan.md §7-6, 2026-10-07, API 0)

    후보   MV-Q  FB-Check CLEAN ∧ Qwen 재판정 CLEAN
           MV-L  FB-Check CLEAN ∧ Llama 재판정 CLEAN
           MV-QL FB-Check CLEAN ∧ (Qwen ∨ Llama CLEAN)
           NB    FB-Check CLEAN ∧ 이웃 합의 상위 2/3 (KoE5 kNN, 같은 문서 제외, k=10, 이웃 forward 조 집합과 Jaccard 평균)

    지표   FB-Check CLEAN 안에서 후보가 남긴 것 vs 버린 것, 문서 층화 순열 이득 (수와 무관)
    순서   탐색 절반에서 이득 최대 후보 하나 선택 → 확인 절반에서 그 후보만 판정

    .venv/bin/python -m eval.label.fbcheck_plus --gpu 0
"""

import argparse
import json
import random

import numpy as np

from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json
from eval.label.label_ablation import _doc_of, _f1, _labels, _ref_by_doc, random_keep_baseline

logger = load_logger("fbcheck_plus.log")

XV_DIR  = PROJECT_ROOT / "data/eval/label/e_cross_verify_raw"
OUT     = PROJECT_ROOT / "data/eval/label/fbcheck_plus.json"
EMB     = PROJECT_ROOT / "data/eval/label/fbcheck_plus_koe5.npy"
SPLIT_SEED, K_NN, NB_DROP = 20261007, 10, 1 / 3
BOOT_N = 10_000


def load_rows() -> list[dict]:
    ref = _ref_by_doc()
    rows = []
    for p, st in (("clean", "CLEAN"), ("noise", "NOISE")):
        for r in load_jsonl(PROJECT_ROOT / f"data/fb_check/{p}.jsonl"):
            if r.get("source") == "ftc_case" and _doc_of(r.get("chunk_id", "")) in ref:
                rows.append({"chunk_id": r["chunk_id"], "doc": _doc_of(r["chunk_id"]), "text": r["text"], "status": st,
                             "forward": _labels(r.get("forward_articles"))})
    for r in rows:
        r["ref"] = ref[r["doc"]]
    return rows


def neighbor_scores(rows: list[dict], gpu: int) -> np.ndarray:
    """SimiFeat식 이웃 합의: 의미가 가까운 다른 문서 조항들의 forward 조 집합과 얼마나 겹치나."""
    if EMB.exists():
        E = np.load(EMB)
    else:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("nlpai-lab/KoE5", device=f"cuda:{gpu}")
        E = m.encode([f"passage: {r['text']}" for r in rows], batch_size=64, normalize_embeddings=True,
                     show_progress_bar=False)
        np.save(EMB, E)
    S = E @ E.T
    docs = np.array([r["doc"] for r in rows])
    S[docs[:, None] == docs[None, :]] = -np.inf                     # 같은 문서는 이웃에서 뺀다
    nn = np.argsort(-S, axis=1)[:, :K_NN]
    out = np.zeros(len(rows))
    for i, r in enumerate(rows):
        a = r["forward"]
        js = []
        for j in nn[i]:
            b = rows[j]["forward"]
            js.append(len(a & b) / len(a | b) if (a | b) else 1.0)
        out[i] = float(np.mean(js))
    return out


def xv_clean(rows: list[dict], model_key: str) -> np.ndarray:
    res = {json.loads(line)["chunk_id"]: json.loads(line)["status"]
           for line in (XV_DIR / f"{model_key}.jsonl").read_text("utf-8").splitlines() if line.strip()}
    return np.array([res.get(r["chunk_id"]) == "CLEAN" for r in rows])


def within(scores, has, pop, keep, docs, rng) -> dict:
    idx = np.where(pop)[0]
    s, h, k = scores[idx], has[idx], keep[idx]
    r = random_keep_baseline(s, h, k, rng, BOOT_N, groups=[docs[i] for i in idx])
    kept, dropped = s[k & h], s[~k & h]
    return {"n_pop": int(len(idx)), "n_kept": int(k.sum()), "n_dropped": int((~k).sum()),
            "f1_kept_pp": round(float(kept.mean()) * 100, 2) if len(kept) else None,
            "f1_dropped_pp": round(float(dropped.mean()) * 100, 2) if len(dropped) else None,
            "docstrat_gain_pp": round(r["diff_pp"], 2), "perm_p": r["perm_p"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    rows = load_rows()
    docs = [r["doc"] for r in rows]
    has = np.array([bool(r["forward"]) for r in rows])
    scores = np.array([_f1(r["forward"], r["ref"]) if r["forward"] else 0.0 for r in rows])
    fb = np.array([r["status"] == "CLEAN" for r in rows])

    # 문서 단위 절반 분할 (사전 등록 시드)
    udocs = sorted(set(docs))
    random.Random(SPLIT_SEED).shuffle(udocs)
    disc_docs = set(udocs[: len(udocs) // 2])
    disc = np.array([d in disc_docs for d in docs])

    q, ll = xv_clean(rows, "qwen2.5-14b"), xv_clean(rows, "llama-3.1-8b")
    nb = neighbor_scores(rows, args.gpu)
    # NB 컷: FB-Check CLEAN 중 이웃 합의 하위 1/3 제거. 컷 값은 탐색 절반의 CLEAN에서만 정한다
    cut = float(np.quantile(nb[fb & disc], NB_DROP))
    cands = {"MV-Q": q, "MV-L": ll, "MV-QL": q | ll, "NB": nb > cut}

    rng = np.random.default_rng(42)
    discovery = {name: within(scores, has, fb & disc, keep, docs, rng) for name, keep in cands.items()}
    best = max(discovery, key=lambda n: (discovery[n]["docstrat_gain_pp"], discovery[n]["n_kept"]))
    logger.info(f"탐색 절반 선택: {best} {discovery[best]}")

    # ── 확인 절반: 선택한 후보만 판정. 나머지는 선택 뒤에 보고용으로만 계산
    conf = within(scores, has, fb & ~disc, cands[best], docs, rng)
    verdict = "FB-Check 향상 확인" if (conf["docstrat_gain_pp"] > 0 and conf["perm_p"] < 0.05) else "미판정"
    others = {n: within(scores, has, fb & ~disc, keep, docs, rng) for n, keep in cands.items() if n != best}

    rep = {"rule": "eval/paper_plan.md §7-6", "split_seed": SPLIT_SEED,
           "n_docs": {"discovery": len(disc_docs), "confirmation": len(udocs) - len(disc_docs)},
           "n_rows": {"discovery": int(disc.sum()), "confirmation": int((~disc).sum())},
           "nb_cut_from_discovery": round(cut, 4),
           "discovery": discovery, "selected": best,
           "confirmation_selected": conf, "verdict": verdict,
           "confirmation_others_report_only": others}
    save_json(rep, OUT)
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
