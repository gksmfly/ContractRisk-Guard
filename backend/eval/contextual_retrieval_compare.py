# backend/eval/contextual_retrieval_compare.py
"""
Contextual Retrieval(Anthropic 방식) 근사 구현 — 청크마다 소속 문맥(법령명·조문번호)을
본문 앞에 명시적으로 붙여서 임베딩한다. 원래 기법은 LLM으로 문맥 요약을 생성하지만,
법령 조문은 이미 정확한 소속 메타데이터(law_name, article_no)를 갖고 있어 LLM 요약보다
더 정확한 "정답 문맥"을 결정론적으로 만들 수 있다 — 그래서 여기서는 LLM 호출 없이
"[법령: {law_name} 제{article_no}조]\n" 프리픽스를 그대로 붙인다.

가설: LightRAG의 실패 원인(그래프 희석)과 RRF의 저조한 성능 둘 다, "청크 본문 자체에
소속 법령이 안 드러나 있어서 Dense 임베딩이 이 조문이 어느 법에 속하는지 반영을 못 한다"는
공통 원인을 갖고 있을 수 있다 — 이 프리픽스로 그 가설을 직접 검증한다.

Dense-only 비교(프리픽스 유무)로 순수하게 이 효과만 격리한다 — production DB의 저장된
임베딩을 재사용하지 않고, 같은 인코딩 호출로 프리픽스 버전/비프리픽스 버전을 둘 다
새로 계산해 방법론 차이를 없앤다. 법령 3,323청크 전체를 메모리에 올려 numpy 코사인
유사도로 비교(로컬 연산, API 비용 없음).

실행: .venv/bin/python -m backend.eval.contextual_retrieval_compare
"""

from typing import Any

import numpy as np

from backend.api.services.retrieval import _get_cached_embedder
from backend.db.loader import embed_texts
from backend.eval.lightrag_compare import _N_QUERIES, LAWS_PATH, build_ground_truth
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("contextual_retrieval_compare.log")
OUT_PATH = PROJECT_ROOT / "data/eval/contextual_retrieval_vs_baseline_report.json"

_TOP_K = 5
_BATCH = 64


def _prefixed_text(rec: dict) -> str:
    meta = rec["metadata"]
    return f"[법령: {meta['law_name']} 제{meta['article_no']}조]\n{rec['text']}"


def _encode_all(embedder: Any, texts: list[str]) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), _BATCH):
        vecs.extend(embed_texts(embedder, texts[i:i + _BATCH], prefix="passage: "))
        logger.info(f"    임베딩 진행: {min(i + _BATCH, len(texts))}/{len(texts)}")
    return np.array(vecs, dtype=np.float32)


def _cosine_top_k(query_vec: np.ndarray, corpus_vecs: np.ndarray, k: int) -> list[int]:
    q = query_vec / np.linalg.norm(query_vec)
    c = corpus_vecs / np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
    sims = c @ q
    return list(np.argsort(-sims)[:k])


def main() -> None:
    law_recs = load_jsonl(LAWS_PATH)
    queries = build_ground_truth(law_recs, n_cases=_N_QUERIES)
    logger.info(f"  평가 쿼리 {len(queries)}건 생성(법령 전체 대상, 동일 seed/로직)")

    embedder = _get_cached_embedder()

    logger.info("  베이스라인(프리픽스 없음) 임베딩 계산 중...")
    baseline_vecs = _encode_all(embedder, [r["text"] for r in law_recs])

    logger.info("  Contextual(프리픽스 있음) 임베딩 계산 중...")
    contextual_vecs = _encode_all(embedder, [_prefixed_text(r) for r in law_recs])

    baseline_hits = contextual_hits = 0
    per_query = []
    for i, q in enumerate(queries):
        query_vec = np.array(embed_texts(embedder, [q["clause"]], prefix="query: ")[0], dtype=np.float32)
        correct = set(map(tuple, q["correct_pairs"]))

        b_idx = _cosine_top_k(query_vec, baseline_vecs, _TOP_K)
        b_found = {(law_recs[j]["metadata"]["law_name"], law_recs[j]["metadata"]["article_no"]) for j in b_idx}
        b_hit = bool(b_found & correct)

        c_idx = _cosine_top_k(query_vec, contextual_vecs, _TOP_K)
        c_found = {(law_recs[j]["metadata"]["law_name"], law_recs[j]["metadata"]["article_no"]) for j in c_idx}
        c_hit = bool(c_found & correct)

        baseline_hits += b_hit
        contextual_hits += c_hit
        per_query.append({"case_name": q["case_name"], "correct_pairs": q["correct_pairs"], "baseline_dense_hit": b_hit, "contextual_dense_hit": c_hit})
        if (i + 1) % 25 == 0:
            logger.info(f"  평가 진행: {i + 1}/{len(queries)} | 누적 Baseline={baseline_hits} Contextual={contextual_hits}")

    n = len(queries)
    result = {
        "n_queries": n,
        "baseline_dense_hit_rate": baseline_hits / n,
        "contextual_dense_hit_rate": contextual_hits / n,
        "baseline_dense_hits": baseline_hits,
        "contextual_dense_hits": contextual_hits,
        "per_query": per_query,
        "note": "Dense-only(sparse/RRF 미포함) 순수 비교 — 프리픽스 유무 효과만 격리. LLM 호출 없음(결정론적 메타데이터 프리픽스).",
    }
    save_json(result, OUT_PATH)
    logger.info(f"========== 최종 결과 (n={n}) ==========")
    logger.info(f"  Dense-only(프리픽스 없음): {baseline_hits}/{n} ({baseline_hits/n*100:.1f}%)")
    logger.info(f"  Dense-only(Contextual):    {contextual_hits}/{n} ({contextual_hits/n*100:.1f}%)")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
