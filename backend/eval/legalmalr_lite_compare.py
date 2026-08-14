# backend/eval/legalmalr_lite_compare.py
"""
LegalMALR(법령 검색 특화 멀티에이전트 쿼리 이해 + 리랭킹)의 "쿼리 재구성" 부분을
경량 근사한다 — 원 논문을 이 세션에서 직접 검증하지 못했으므로(reference 확인 불가,
backend/eval/retrieval_alternatives_survey.md 참고) 논문 그대로의 재현이 아니라
"법률 쿼리 재구성이 원문 그대로 검색하는 것보다 나은가"라는 핵심 가설만 근사 검증한다.

계약 조항 원문(번호·형식이 섞인 긴 텍스트)을 그대로 임베딩/검색하는 대신, 로컬 LLM
(EXAONE-3.5-7.8B-Instruct, OpenAI API 비용 없음)으로 "이 조항이 다투는 법적 쟁점을
명시적인 법률 용어로 재구성"한 뒤 그 재구성 쿼리로 검색한다 — 원 조항엔 없는 법률
용어(예: "계약 해지", "손해배상액의 예정")를 명시적으로 드러내면 Dense/Sparse 검색
둘 다에 도움이 될 수 있다는 가설.

동일 ground truth(FTC 근거_법령, 법령 전체 3,323청크, seed=42, 100건), RRF 검색은
domain_filter_compare.py의 rrf_baseline_hit()와 동일 로직(파티션 없음, top_k=8) 재사용.

실행: .venv/bin/python -m backend.eval.legalmalr_lite_compare
"""

from backend.api.services.retrieval import _get_cached_embedder
from backend.db.connection import get_conn
from backend.eval.domain_filter_compare import rrf_baseline_hit
from backend.eval.lightrag_compare import LAWS_PATH, _N_QUERIES, build_ground_truth
from backend.eval.local_llm import generate_json
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("legalmalr_lite_compare.log")
OUT_PATH = PROJECT_ROOT / "data/eval/legalmalr_lite_vs_rrf_report.json"

_SYSTEM = (
    "너는 한국 법률 전문가다. 계약 조항 원문을 읽고, 이 조항이 실제로 다투는 법적 쟁점을 "
    "간결한 법률 검색 쿼리로 재구성해라. 조항에 없더라도 관련 법률 용어(해지/해제, 손해배상액의 "
    "예정, 면책조항, 무효, 개별약정 우선 등)를 명시적으로 포함시켜라. "
    '반드시 JSON만 출력: {"reformed_query": "..."}'
)
_FEWSHOT = [
    {"role": "user", "content": "계약 조항:\n제5조 을은 계약기간 중 언제든지 갑에게 서면 통지만으로 계약을 해지할 수 있으며, 갑은 이에 대해 이의를 제기할 수 없다."},
    {"role": "assistant", "content": '{"reformed_query": "사업자의 일방적 계약 해지권, 고객의 이의제기권 배제, 약관의 규제에 관한 법률상 해지 조항의 불공정성"}'},
    {"role": "user", "content": "계약 조항:\n제9조 회원이 이용약관을 위반한 경우 회사는 손해배상과 별도로 위약금으로 계약금의 3배를 청구할 수 있다."},
    {"role": "assistant", "content": '{"reformed_query": "손해배상액의 예정, 과중한 위약금 조항, 약관의 규제에 관한 법률상 손해배상액 예정의 부당성"}'},
]


def main() -> None:
    law_recs = load_jsonl(LAWS_PATH)
    queries = build_ground_truth(law_recs, n_cases=_N_QUERIES)
    logger.info(f"  평가 쿼리 {len(queries)}건 생성(법령 전체 대상, 동일 seed/로직)")

    embedder = _get_cached_embedder()
    with get_conn() as conn, conn.cursor() as cur:
        rrf_hits = reformed_hits = 0
        per_query = []
        for i, q in enumerate(queries):
            r_hit = rrf_baseline_hit(cur, embedder, q["clause"], q["correct_pairs"])

            content = f"계약 조항:\n{q['clause'][:800]}"
            result = generate_json(_SYSTEM, content, max_new_tokens=150, fewshot=_FEWSHOT)
            reformed = (result or {}).get("reformed_query", "").strip()
            rf_hit = rrf_baseline_hit(cur, embedder, reformed, q["correct_pairs"]) if reformed else False

            rrf_hits += r_hit
            reformed_hits += rf_hit
            per_query.append({
                "case_name": q["case_name"], "correct_pairs": q["correct_pairs"],
                "reformed_query": reformed, "rrf_hit": r_hit, "reformed_query_hit": rf_hit,
            })
            if (i + 1) % 10 == 0:
                logger.info(f"  평가 진행: {i + 1}/{len(queries)} | 누적 RRF(원문)={rrf_hits} RRF(재구성쿼리)={reformed_hits}")

    n = len(queries)
    result = {
        "n_queries": n,
        "rrf_original_hit_rate": rrf_hits / n,
        "rrf_reformed_hit_rate": reformed_hits / n,
        "rrf_original_hits": rrf_hits,
        "rrf_reformed_hits": reformed_hits,
        "per_query": per_query,
        "note": (
            "LegalMALR 원 논문 미검증 — '법률 쿼리 재구성' 아이디어만 근사(로컬 EXAONE-3.5-7.8B, "
            "few-shot 2건, API 비용 0). 논문 자체의 재현이 아님."
        ),
    }
    save_json(result, OUT_PATH)
    logger.info(f"========== 최종 결과 (n={n}) ==========")
    logger.info(f"  RRF(원문 그대로):     {rrf_hits}/{n} ({rrf_hits/n*100:.1f}%)")
    logger.info(f"  RRF(재구성 쿼리):     {reformed_hits}/{n} ({reformed_hits/n*100:.1f}%)")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
