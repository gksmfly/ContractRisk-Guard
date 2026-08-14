# backend/eval/lightrag_scoped_compare.py
"""
옵션3(GraphRAG 서브그래프 스코핑)을 "안 해봄"으로 남기지 말고 실제로 시도한다.

`lightrag.QueryParam`/`BaseVectorStorage.query()` 어디에도 소스(법령명) 필터 파라미터가
없다는 건 이미 확인했다(project_retrieval_alternatives_eval.md 참고) — 그래프 순회
자체(엔티티/관계 추론)를 스코핑하는 건 내부 스토리지를 몽키패치해야 해서 여전히 시도
안 한다. 대신 **그래프가 만든 최종 청크 벡터스토어(`rag.chunks_vdb`)를 직접 top_k=30으로
넉넉히 조회한 뒤, RAPTOR-lite와 같은 로컬 EXAONE 라우팅으로 예측한 법령에 속하는 청크만
남기고 top-5를 뽑는다** — "그래프 자체의 순회 범위"는 못 좁히지만 "그래프가 후보로 올린
청크 풀"은 스코핑할 수 있다는 절충안. 진짜 서브그래프 스코핑이 아니라 근사임을 분명히
한다.

청크 텍스트만으론 소속 법령을 못 가려서(reference_id가 비어있음, lightrag_compare.py의
이유와 동일) RAPTOR-lite처럼 정답 조문 본문 텍스트 매칭이 아니라, law_recs에서 텍스트로
역매핑해 law_name을 붙인다.

주의: rag.chunks_vdb.query()는 LightRAG 설정대로 OpenAI 임베딩(openai_embed)을 쓴다 —
KoE5가 아니라서 다른 옵션들과 임베딩 모델 자체가 다르다는 confound가 있다(이 실험은
"스코핑 효과"만 보려는 것이지 "임베딩 모델 비교"가 아님을 결과 해석 시 감안해야 함).
API 비용 발생(쿼리당 임베딩 1회, 100건 — 소액).

실행: .venv/bin/python -m backend.eval.lightrag_scoped_compare
"""

import asyncio

from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc

from backend.eval.domain_filter_compare import _law_names
from backend.eval.lightrag_compare import LAWS_PATH, WORKING_DIR, _N_QUERIES, build_ground_truth
from backend.eval.raptor_lite_compare import _FEWSHOT, _SYSTEM_TMPL
from backend.eval.local_llm import generate_json
from backend.db.connection import get_conn
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("lightrag_scoped_compare.log")
OUT_PATH = PROJECT_ROOT / "data/eval/lightrag_scoped_report.json"

_RAW_TOP_K = 30
_TOP_K = 5


def _text_to_law(chunk_text: str, article_text_index: dict[str, tuple]) -> tuple | None:
    """청크 본문(앞 60자)으로 law_recs 역매핑 — lightrag_compare.py의 정답판정과 동일 원리."""
    for prefix, pair in article_text_index.items():
        if prefix and prefix in chunk_text:
            return pair
    return None


async def main() -> None:
    law_recs = load_jsonl(LAWS_PATH)
    queries = build_ground_truth(law_recs, n_cases=_N_QUERIES)
    logger.info(f"  평가 쿼리 {len(queries)}건 생성(법령 전체 대상, 동일 seed/로직)")

    article_text_index = {
        r["text"][:60]: (r["metadata"]["law_name"], r["metadata"]["article_no"])
        for r in law_recs if r["text"][:60].strip()
    }

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=EmbeddingFunc(embedding_dim=1536, max_token_size=8192, func=openai_embed),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    with get_conn() as conn, conn.cursor() as cur:
        law_names = _law_names(cur)
        system = _SYSTEM_TMPL.format(law_list=", ".join(law_names))

    raw_hits = scoped_hits = 0
    per_query = []
    for i, q in enumerate(queries):
        results = await rag.chunks_vdb.query(q["clause"], top_k=_RAW_TOP_K)
        correct_texts = set(q["correct_texts"])

        # 스코핑 없는 baseline: 그래프 벡터스토어 top-5 그대로
        raw_top5_content = "".join(r.get("content", "") for r in results[:_TOP_K])
        raw_hit = any(t in raw_top5_content for t in correct_texts)

        # 로컬 EXAONE 라우팅으로 예측 법령만 남기고 top-5
        content = f"계약 조항:\n{q['clause'][:800]}"
        route_result = generate_json(system, content, max_new_tokens=100, fewshot=_FEWSHOT)
        predicted = [law for law in (route_result or {}).get("laws", []) if law in law_names]

        scoped = []
        for r in results:
            pair = _text_to_law(r.get("content", ""), article_text_index)
            if pair and pair[0] in predicted:
                scoped.append(r)
            if len(scoped) >= _TOP_K:
                break
        scoped_content = "".join(r.get("content", "") for r in scoped)
        scoped_hit = any(t in scoped_content for t in correct_texts)

        raw_hits += raw_hit
        scoped_hits += scoped_hit
        per_query.append({
            "case_name": q["case_name"], "correct_pairs": q["correct_pairs"],
            "predicted_laws": predicted, "raw_hit": raw_hit, "scoped_hit": scoped_hit,
        })
        if (i + 1) % 10 == 0:
            logger.info(f"  평가 진행: {i + 1}/{len(queries)} | 누적 Raw={raw_hits} Scoped={scoped_hits}")

    n = len(queries)
    result = {
        "n_queries": n,
        "raw_hit_rate": raw_hits / n,
        "scoped_hit_rate": scoped_hits / n,
        "raw_hits": raw_hits,
        "scoped_hits": scoped_hits,
        "per_query": per_query,
        "note": (
            "진짜 그래프 순회 스코핑이 아니라 chunks_vdb 청크 풀을 라우팅 예측 법령으로 "
            "사후 필터링한 근사. LightRAG 설정상 OpenAI 임베딩(openai_embed) 사용 — "
            "다른 옵션들(KoE5)과 임베딩 모델 자체가 다름(confound)."
        ),
    }
    save_json(result, OUT_PATH)
    logger.info(f"========== 최종 결과 (n={n}) ==========")
    logger.info(f"  LightRAG chunks_vdb(스코핑 없음): {raw_hits}/{n} ({raw_hits/n*100:.1f}%)")
    logger.info(f"  LightRAG chunks_vdb(라우팅 스코핑): {scoped_hits}/{n} ({scoped_hits/n*100:.1f}%)")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
