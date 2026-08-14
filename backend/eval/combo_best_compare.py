# backend/eval/combo_best_compare.py
"""
가장 성능이 좋았던 두 방법(LegalMALR-lite 쿼리 재구성, RAPTOR-lite 법령 라우팅)에
Cross-Encoder 재랭킹까지 얹은 "종합 콤보" — 각각 33%, 33%까지 올렸던 두 기법을 합치면
더 올라가는지 확인한다. 로컬 EXAONE 호출 1회로 재구성 쿼리 + 라우팅 예측을 동시에
받아 쿼리당 LLM 호출 수를 늘리지 않는다.

파이프라인: EXAONE 1회 호출(재구성 쿼리 + 예측 법령 top-2) → 예측 법령 파티션만
재구성 쿼리로 dense+sparse 검색(top-10/파티션) → Cross-Encoder로 최종 top-5 재정렬.

동일 ground truth(FTC 근거_법령, 법령 전체 3,323청크, seed=42, 100건).

실행: .venv/bin/python -m backend.eval.combo_best_compare
"""

from sentence_transformers import CrossEncoder

from backend.api.services.retrieval import _get_cached_embedder
from backend.db.connection import get_conn
from backend.db.loader import embed_texts
from backend.eval.domain_filter_compare import _dense_with_score, _law_names, _sparse_with_score, rrf_baseline_hit
from backend.eval.lightrag_compare import LAWS_PATH, _N_QUERIES, build_ground_truth
from backend.eval.local_llm import generate_json
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("combo_best_compare.log")
OUT_PATH = PROJECT_ROOT / "data/eval/combo_best_report.json"

_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_CANDIDATE_K_PER_PARTITION = 10
_TOP_K = 5

_SYSTEM_TMPL = (
    "너는 한국 법률 전문가다. 계약 조항을 읽고 두 가지를 답해라.\n"
    "1) 이 조항이 다투는 법적 쟁점을 간결한 법률 검색 쿼리로 재구성(조항에 없어도 관련 "
    "법률 용어를 명시적으로 포함).\n"
    "2) 근거로 삼을 가능성이 높은 법령을 아래 목록에서 정확히 2개 선택.\n"
    "목록: {law_list}\n"
    '반드시 JSON만 출력: {{"reformed_query": "...", "laws": ["법령명1", "법령명2"]}}'
)
_FEWSHOT = [
    {"role": "user", "content": "계약 조항:\n제5조 을은 계약기간 중 언제든지 갑에게 서면 통지만으로 계약을 해지할 수 있으며, 갑은 이에 대해 이의를 제기할 수 없다."},
    {"role": "assistant", "content": '{"reformed_query": "사업자의 일방적 계약 해지권, 고객의 이의제기권 배제, 약관의 규제에 관한 법률상 해지 조항의 불공정성", "laws": ["약관의 규제에 관한 법률", "민법"]}'},
    {"role": "user", "content": "계약 조항:\n제9조 회원이 이용약관을 위반한 경우 회사는 손해배상과 별도로 위약금으로 계약금의 3배를 청구할 수 있다."},
    {"role": "assistant", "content": '{"reformed_query": "손해배상액의 예정, 과중한 위약금 조항, 약관의 규제에 관한 법률상 손해배상액 예정의 부당성", "laws": ["약관의 규제에 관한 법률", "민법"]}'},
]


def combo_best_hit(cur, embedder, reranker: CrossEncoder, clause: str, law_names: list[str], system: str, correct_pairs: list[tuple]) -> bool:
    content = f"계약 조항:\n{clause[:800]}"
    result = generate_json(system, content, max_new_tokens=200, fewshot=_FEWSHOT)
    reformed = (result or {}).get("reformed_query", "").strip() or clause
    predicted = [law for law in (result or {}).get("laws", []) if law in law_names]
    if not predicted:
        return False

    query_vec = embed_texts(embedder, [reformed], prefix="query: ")[0]
    vec_literal = "[" + ",".join(repr(x) for x in query_vec) + "]"

    pool: dict[str, dict] = {}
    for law_name in predicted:
        for row in _dense_with_score(cur, law_name, vec_literal, _CANDIDATE_K_PER_PARTITION):
            chunk_id, source, metadata, text, score = row
            pool[chunk_id] = {"chunk_id": chunk_id, "metadata": metadata, "text": text}
        for row in _sparse_with_score(cur, law_name, reformed, _CANDIDATE_K_PER_PARTITION):
            chunk_id, source, metadata, text, score = row
            pool.setdefault(chunk_id, {"chunk_id": chunk_id, "metadata": metadata, "text": text})

    candidates = list(pool.values())
    if not candidates:
        return False

    scores = reranker.predict([(clause, c["text"]) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:_TOP_K]
    found = {(c["metadata"].get("law_name"), c["metadata"].get("article_no")) for c, _ in ranked}
    return bool(found & set(map(tuple, correct_pairs)))


def main() -> None:
    law_recs = load_jsonl(LAWS_PATH)
    queries = build_ground_truth(law_recs, n_cases=_N_QUERIES)
    logger.info(f"  평가 쿼리 {len(queries)}건 생성(법령 전체 대상, 동일 seed/로직)")

    logger.info(f"  Cross-Encoder 로드: {_RERANKER_MODEL}")
    reranker = CrossEncoder(_RERANKER_MODEL, max_length=512)

    embedder = _get_cached_embedder()
    with get_conn() as conn, conn.cursor() as cur:
        law_names = _law_names(cur)
        system = _SYSTEM_TMPL.format(law_list=", ".join(law_names))

        rrf_hits = combo_hits = 0
        per_query = []
        for i, q in enumerate(queries):
            r_hit = rrf_baseline_hit(cur, embedder, q["clause"], q["correct_pairs"])
            c_hit = combo_best_hit(cur, embedder, reranker, q["clause"], law_names, system, q["correct_pairs"])
            rrf_hits += r_hit
            combo_hits += c_hit
            per_query.append({"case_name": q["case_name"], "correct_pairs": q["correct_pairs"], "rrf_hit": r_hit, "combo_best_hit": c_hit})
            if (i + 1) % 10 == 0:
                logger.info(f"  평가 진행: {i + 1}/{len(queries)} | 누적 RRF={rrf_hits} ComboBest={combo_hits}")

    n = len(queries)
    result = {
        "n_queries": n,
        "rrf_hit_rate": rrf_hits / n,
        "combo_best_hit_rate": combo_hits / n,
        "rrf_hits": rrf_hits,
        "combo_best_hits": combo_hits,
        "per_query": per_query,
        "note": "EXAONE 1회 호출(쿼리 재구성+법령 라우팅 동시) + 파티션 검색 + Cross-Encoder 재랭킹. LegalMALR-lite+RAPTOR-lite+Cross-Encoder 종합.",
    }
    save_json(result, OUT_PATH)
    logger.info(f"========== 최종 결과 (n={n}) ==========")
    logger.info(f"  RRF(파티션 없음):  {rrf_hits}/{n} ({rrf_hits/n*100:.1f}%)")
    logger.info(f"  종합 콤보:         {combo_hits}/{n} ({combo_hits/n*100:.1f}%)")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
