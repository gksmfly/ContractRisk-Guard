# eval/rag/explain_quality.py
"""Qwen 설명 대 GPT-4o 저장 설명 — 기술적 비교 (탐색적, 사전 등록 아님, 2026-10-12, API 0)

사람이 읽고 판단하는 평가를 못 하므로 자동으로 잴 수 있는 것만 잰다. GPT-4o 결과는 이미 저장된 것(forward_labeling)이다.
GPT-4o는 서비스가 쓰는 GPT-4o-mini보다 좋은 모델이라 "GPT 쪽 상한"에 가깝다.

    HF_HUB_OFFLINE=1 .venv/bin/python -m eval.rag.explain_quality
"""

import json
import re

import numpy as np

from backend.agents.evidence_selection_agent import _build_fallback
from backend.db.loader import embed_texts, get_embedder
from backend.training.retrain import eval_sets
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("explain_quality.log")
OUT = PROJECT_ROOT / "data/eval/rag/explain_quality.json"
_WS = re.compile(r"\s+")
_TOK = re.compile(r"[가-힣A-Za-z0-9]{2,}")
rng = np.random.default_rng(42)


def toks(t: str) -> set[str]:
    return set(_TOK.findall(t or ""))


def f1(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    i = len(a & b)
    return 0.0 if i == 0 else 2 * i / (len(a) + len(b))


def boot(vals: np.ndarray, n: int = 5000) -> list[float]:
    b = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
    return [round(float(x), 3) for x in np.percentile(b, [2.5, 97.5])]


def main() -> None:
    holdout = json.loads((PROJECT_ROOT / "backend/training/negative_holdout_docs.json").read_text(encoding="utf-8"))
    gold, _ = eval_sets(holdout)
    recs = {}
    for r in load_jsonl(PROJECT_ROOT / "data/fb_check/clean.jsonl") + load_jsonl(PROJECT_ROOT / "data/fb_check/noise.jsonl"):
        recs.setdefault(_WS.sub("", r["text"]), r)
    pairs = []
    for row in (json.loads(l) for l in open(PROJECT_ROOT / "data/eval/rag/explain_A0_raw.jsonl")):
        if not row["raw"] or not row["audit"]["format_ok"]:
            continue
        clause = gold[row["i"]]["text"]
        g = recs.get(_WS.sub("", clause))
        if not g or not g.get("reasoning") or not g.get("evidence_span"):
            continue
        arts = " ".join(b["description"] for b in _build_fallback(row["pred"]))
        pairs.append({"clause": clause, "arts": arts, "q_span": row["raw"]["evidence_span"],
                      "q_exp": f"{row['raw']['explanation']} {row['raw'].get('suggestion', '')}".strip(),
                      "g_span": g["evidence_span"], "g_exp": g["reasoning"]})
    n = len(pairs)
    logger.info(f"짝 {n}건")

    # 1. 하이라이트 일치
    span_f1 = np.array([f1(toks(p["q_span"]), toks(p["g_span"])) for p in pairs])
    overlap = np.array([float(bool(toks(p["q_span"]) & toks(p["g_span"]))) for p in pairs])
    # 2. 설명 의미 유사도 (+ 다른 조항 설명과 짝지은 기준선)
    emb = get_embedder()
    Q = np.array(embed_texts(emb, [p["q_exp"] for p in pairs]))
    G = np.array(embed_texts(emb, [p["g_exp"] for p in pairs]))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    same = (Q * G).sum(1)
    shuf = np.mean([(Q * G[rng.permutation(n)]).sum(1) for _ in range(20)], axis=0)
    # 3. 조항 특이성: 같은 쪽 다른 설명들과의 평균 유사도(낮을수록 조항에 맞춘 문장)
    def generic(X):
        S = X @ X.T
        return (S.sum(1) - 1) / (n - 1)
    gq, gg = generic(Q), generic(G)
    # 4. 원문 근거성: 설명의 단어가 조항 원문 또는 제공된 조문에 있는 비율 / 조항 원문에만 있는 비율
    def grounded(key, only_clause=False):
        out = []
        for p in pairs:
            t = toks(p[key])
            src = toks(p["clause"]) if only_clause else toks(p["clause"]) | toks(p["arts"])
            out.append(len(t & src) / max(len(t), 1))
        return np.array(out)

    rep = {"n_pairs": n,
           "GPT-4o는 서비스의 GPT-4o-mini보다 상한": True,
           "1_하이라이트_토큰F1_평균": round(float(span_f1.mean()), 3), "1_ci": boot(span_f1),
           "1_하이라이트_단어가_하나라도_겹치는_비율": round(float(overlap.mean()), 3),
           "2_같은조항_설명유사도": round(float(same.mean()), 3), "2_다른조항과_짝지은_기준선": round(float(shuf.mean()), 3),
           "2_차이_ci": boot(same - shuf),
           "3_조항간_평균유사도_Qwen(낮을수록_특이)": round(float(gq.mean()), 3),
           "3_조항간_평균유사도_GPT4o": round(float(gg.mean()), 3), "3_차이_Qwen-GPT_ci": boot(gq - gg),
           "4_원문+조문에_있는_단어비율_Qwen": round(float(grounded("q_exp").mean()), 3),
           "4_원문+조문에_있는_단어비율_GPT4o": round(float(grounded("g_exp").mean()), 3),
           "4_원문에만_있는_단어비율_Qwen": round(float(grounded("q_exp", True).mean()), 3),
           "4_원문에만_있는_단어비율_GPT4o": round(float(grounded("g_exp", True).mean()), 3),
           "평균_글자수_Qwen": round(float(np.mean([len(p["q_exp"]) for p in pairs])), 1),
           "평균_글자수_GPT4o": round(float(np.mean([len(p["g_exp"]) for p in pairs])), 1)}
    save_json(rep, OUT)
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
