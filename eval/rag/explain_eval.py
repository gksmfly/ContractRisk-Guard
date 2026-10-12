# eval/rag/explain_eval.py
"""근거 기반 설명 생성(RAG 안 A) 평가 (사전 등록: eval/paper_plan.md §7-21, 2026-10-12, API 0)

FTC 단일 조 gold 중 article_v3가 조를 지목한 조항에서, 지목한 조의 조문 원문(A0) 또는
조문 + 유사 판례 2건(A1)을 로컬 Qwen에 읽혀 설명을 만들고, **후처리 전 원출력**을 자동 검사한다.

    HF_HOME=/dev/shm/gksmfly_hf HF_HUB_OFFLINE=1 LOCAL_LLM_DEVICE=cuda:1 JUDGMENT_DEVICE=cuda:1 \
      .venv/bin/python -m eval.rag.explain_eval --arm A0
    ... --arm A1          # 판례 포함(DB 필요)
    ... --analyze
"""

import argparse
import json
import os
import time

import numpy as np

from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("explain_eval.log")
OUT_DIR = PROJECT_ROOT / "data/eval/rag"
OUT = OUT_DIR / "explain_eval.json"
MODEL = "local:qwen2.5-14b"
THRESH = {"format_ok": 0.95, "span_verbatim": 0.95, "violation_max": 0.02}


def _raw_path(arm: str):
    return OUT_DIR / f"explain_{arm}_raw.jsonl"


def run(arm: str, limit: int) -> None:
    from backend.agents import explain_agent as ex
    from backend.agents.evidence_selection_agent import _build_fallback
    from backend.agents.judgment_agent import predict_articles
    from backend.retrieval.schemas import LegalBasis
    from backend.training.retrain import eval_sets

    os.environ["EXPLAIN_MODEL"] = MODEL
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gold, _ = eval_sets(json.loads((PROJECT_ROOT / "backend/training/negative_holdout_docs.json").read_text(encoding="utf-8")))
    path = _raw_path(arm)
    done = {r["i"] for r in load_jsonl(path)} if path.exists() else set()

    if arm == "A1":
        from backend.agents.evidence_selection_agent import evidence_selection_node
        from backend.agents.retrieval_strategy_agent import retrieval_strategy_node
        from backend.db.connection import init_pool
        init_pool()

    n_eval = 0
    with open(path, "a", encoding="utf-8") as f:
        for i, g in enumerate(gold):
            if limit and n_eval >= limit:
                break
            arts = predict_articles(g["text"])
            if not arts:
                continue                              # 서비스에서는 범위 밖으로 빠지는 조항
            n_eval += 1
            if i in done:
                continue
            basis = [LegalBasis(**b) for b in _build_fallback(arts)]
            precedents = None
            if arm == "A1":
                st = {"clause": g["text"], "evidence_span": "", "model_articles": arts, "retry_count": 0}
                st.update(retrieval_strategy_node(st))
                precedents = evidence_selection_node(st)["precedent_refs"]
            t0 = time.time()
            raw = ex.generate(g["text"], basis, precedents, MODEL)
            dt = time.time() - t0
            a = ex.audit(g["text"], raw, [b.article for b in basis], precedents)
            f.write(json.dumps({"i": i, "doc": g["doc_id"], "pred": arts, "gold": g["articles"], "sec": round(dt, 2),
                                "raw": raw, "audit": a}, ensure_ascii=False) + "\n")
            f.flush()
            if n_eval % 20 == 0:
                logger.info(f"  [{arm}] {n_eval}건")


def _ci(vals: np.ndarray, groups: list[str]) -> list[float]:
    idx: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    cl = [np.array(v) for v in idx.values()]
    rng = np.random.default_rng(42)
    b = [vals[np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])].mean() for _ in range(3000)]
    return [round(float(x) * 100, 1) for x in np.percentile(b, [2.5, 97.5])]


def gpt_reference() -> dict:
    """운영 GPT-4o forward 저장분: 조를 지목한 레코드의 근거 문구 존재율. 조항이 달라 참고만."""
    rows = [r for r in load_jsonl(PROJECT_ROOT / "data/fb_check/clean.jsonl") + load_jsonl(PROJECT_ROOT / "data/fb_check/noise.jsonl")
            if r.get("forward_articles")]
    return {"n": len(rows), "span_exists_pct": round(float(np.mean([bool(r.get("snippet_exists")) for r in rows])) * 100, 1)}


def analyze() -> None:
    rep: dict = {"rule": "eval/paper_plan.md §7-21", "thresholds": THRESH, "gpt4o_reference": gpt_reference(), "arms": {}}
    for arm in ("A0", "A1"):
        path = _raw_path(arm)
        if not path.exists():
            continue
        rows = load_jsonl(path)
        g = [r["doc"] for r in rows]
        col = lambda k: np.array([float(r["audit"][k]) for r in rows])  # noqa: E731
        fmt = col("format_ok")
        ok = fmt == 1
        viol = np.array([float(r["audit"]["bad_cited"] or r["audit"]["bad_mention"]) for r in rows])
        span = col("span_verbatim")
        sec = np.array([r["sec"] for r in rows])
        res = {"n": len(rows), "format_ok_pct": round(fmt.mean() * 100, 1), "format_ok_ci": _ci(fmt, g),
               "span_verbatim_pct": round(span.mean() * 100, 1), "span_verbatim_ci": _ci(span, g),
               "span_verbatim_among_format_ok_pct": round(span[ok].mean() * 100, 1) if ok.any() else None,
               "citation_violation_pct": round(viol.mean() * 100, 1), "citation_violation_ci": _ci(viol, g),
               "bad_cited_pct": round(col("bad_cited").mean() * 100, 1), "bad_mention_pct": round(col("bad_mention").mean() * 100, 1),
               "bad_case_no_n": int(col("bad_case_no").sum()),
               "sec_p50": round(float(np.median(sec)), 1), "sec_p95": round(float(np.percentile(sec, 95)), 1)}
        res["pass"] = bool(fmt.mean() >= THRESH["format_ok"] and span.mean() >= THRESH["span_verbatim"]
                           and viol.mean() <= THRESH["violation_max"])
        if arm == "A1":
            res["pass"] = bool(res["pass"] and res["bad_case_no_n"] == 0)
        rep["arms"][arm] = res
    save_json(rep, OUT)
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A0", "A1"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.arm:
        run(a.arm, a.limit)
    if a.analyze:
        analyze()


if __name__ == "__main__":
    main()
