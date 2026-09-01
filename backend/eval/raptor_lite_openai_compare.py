# backend/eval/raptor_lite_openai_compare.py
"""
RAPTOR-lite 법령 라우팅을 **OpenAI gpt-4o-mini**로 수행해, 로컬 EXAONE-3.5-7.8B 결과
(33%)와 직접 비교한다.

왜 이 실험이 필요한가: `raptor_lite_compare.py`가 EXAONE을 쓴 건 "평가 스크립트
~100건에 OpenAI 예산을 쓰지 말자"는 **실험 단계의 제약** 때문이었다. 그런데 그 코드가
그대로 프로덕션(`backend/agents/query_router.py`)에 반영되면서, 요청 경로에 7.8B 로컬
모델을 상주시키는 운영 부담(GPU 16GB 상주, 콜드스타트, accelerate 의존성, Docker GPU
설정)이 따라왔다. 정작 파이프라인은 **이미 조항마다 OpenAI를 호출한다**(Analysis 노드의
FORWARD_MODEL, Red-team의 gpt-4o-mini) — 즉 "API 비용 0원"이라는 EXAONE의 최대 장점이
프로덕션에서는 거의 의미가 없다. 그럼에도 13종 대안 비교(`retrieval_alternatives_survey.md`)에
gpt-4o-mini 라우팅은 **한 번도 포함된 적이 없다**. 이 스크립트가 그 공백을 메운다.

비교 가능성 보장: 프롬프트(`_SYSTEM_TMPL`)·few-shot(`_FEWSHOT`)·검색 로직(`routed_hit`)을
`raptor_lite_compare.py`에서 **import해서 그대로 쓴다** — 복사하지 않으므로 문구가
갈라질 수 없다. ground truth도 같은 `build_ground_truth(seed=42, n_cases=100)`이라
EXAONE 결과와 쿼리 단위로 짝지어 비교(McNemar)할 수 있다. 바뀌는 건 라우팅을 수행하는
모델 하나뿐이다.

실행:
    .venv/bin/python -m backend.eval.raptor_lite_openai_compare --limit 5   # 스모크
    .venv/bin/python -m backend.eval.raptor_lite_openai_compare             # 전체 100건
"""

import argparse
import json
import os
import time
from math import comb

from openai import OpenAI

from backend.api.services.retrieval import _get_cached_embedder
from backend.db.connection import get_conn
from backend.eval.domain_filter_compare import _law_names, rrf_baseline_hit
from backend.eval.lightrag_compare import _N_QUERIES, LAWS_PATH, build_ground_truth
from backend.eval.raptor_lite_compare import _FEWSHOT, _SYSTEM_TMPL, routed_hit
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("raptor_lite_openai_compare.log")
OUT_PATH = PROJECT_ROOT / "data/eval/raptor_lite_openai_vs_rrf_report.json"
EXAONE_REPORT = PROJECT_ROOT / "data/eval/raptor_lite_vs_rrf_report.json"

ROUTER_EVAL_MODEL = os.environ.get("ROUTER_EVAL_MODEL", "gpt-4o-mini")


def _route_openai(
    client: OpenAI, system: str, user_content: str, retries: int = 3
) -> tuple[dict | None, int, int]:
    """gpt-4o-mini로 법령 top-2를 예측한다. 반환: (파싱된 JSON, prompt_tokens, completion_tokens).

    local_llm.generate_json()과 같은 역할이되 백엔드만 OpenAI다. 온도 0·JSON 모드로
    고정해 로컬 쪽 greedy decoding(do_sample=False)과 조건을 맞춘다.
    """
    messages = [{"role": "system", "content": system}, *_FEWSHOT, {"role": "user", "content": user_content}]
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=ROUTER_EVAL_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=100,
            )
            usage = resp.usage
            return (
                json.loads(resp.choices[0].message.content),
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            )
        except Exception as e:
            logger.warning(f"  라우팅 호출 실패 ({attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None, 0, 0


def _mcnemar_exact(b: int, c: int) -> float:
    """불일치 쌍(b, c)에 대한 McNemar 정확검정 양측 p-value — 기존 비교 스크립트들과 동일한 방식."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _load_exaone_per_query() -> dict[str, bool] | None:
    """EXAONE 실행 결과를 case_name 기준으로 읽어 짝지은 비교에 쓴다."""
    if not EXAONE_REPORT.exists():
        return None
    data = json.loads(EXAONE_REPORT.read_text())
    return {q["case_name"]: bool(q["routed_hit"]) for q in data.get("per_query", [])}


def main(limit: int | None = None) -> None:
    law_recs = load_jsonl(LAWS_PATH)
    queries = build_ground_truth(law_recs, n_cases=_N_QUERIES)
    if limit:
        queries = queries[:limit]
    logger.info(f"  평가 쿼리 {len(queries)}건 (모델={ROUTER_EVAL_MODEL}, 동일 seed/ground truth)")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    embedder = _get_cached_embedder()

    with get_conn() as conn, conn.cursor() as cur:
        law_names = _law_names(cur)
        system = _SYSTEM_TMPL.format(law_list=", ".join(law_names))
        logger.info(f"  파티션 {len(law_names)}개, 라우팅 대상")

        rrf_hits = routed_hits = routing_correct = 0
        prompt_tokens = completion_tokens = 0
        per_query = []
        for i, q in enumerate(queries):
            r_hit = rrf_baseline_hit(cur, embedder, q["clause"], q["correct_pairs"])

            content = f"계약 조항:\n{q['clause'][:800]}"
            result, p_tok, c_tok = _route_openai(client, system, content)
            prompt_tokens += p_tok
            completion_tokens += c_tok

            predicted = [law for law in (result or {}).get("laws", []) if law in law_names]
            correct_laws = {p[0] for p in q["correct_pairs"]}
            is_routing_correct = bool(set(predicted) & correct_laws)

            rt_hit = routed_hit(cur, embedder, q["clause"], predicted, q["correct_pairs"])

            rrf_hits += r_hit
            routed_hits += rt_hit
            routing_correct += is_routing_correct
            per_query.append({
                "case_name": q["case_name"], "correct_pairs": q["correct_pairs"],
                "predicted_laws": predicted, "routing_correct": is_routing_correct,
                "rrf_hit": r_hit, "routed_hit": rt_hit,
            })
            if (i + 1) % 10 == 0:
                logger.info(
                    f"  평가 진행: {i + 1}/{len(queries)} | 누적 RRF={rrf_hits} "
                    f"Routed={routed_hits} 라우팅정답률={routing_correct}/{i + 1}"
                )

    n = len(queries)

    # (1) gpt-4o-mini 라우팅 vs 필터 없는 RRF — EXAONE 리포트와 같은 형태의 1차 비교
    b_rrf = sum(1 for q in per_query if q["routed_hit"] and not q["rrf_hit"])
    c_rrf = sum(1 for q in per_query if q["rrf_hit"] and not q["routed_hit"])

    result = {
        "n_queries": n,
        "router_model": ROUTER_EVAL_MODEL,
        "rrf_hit_rate": rrf_hits / n,
        "routed_hit_rate": routed_hits / n,
        "routing_accuracy": routing_correct / n,
        "rrf_hits": rrf_hits,
        "routed_hits": routed_hits,
        "mcnemar_vs_rrf": {"b_routed_only": b_rrf, "c_rrf_only": c_rrf, "p_value": _mcnemar_exact(b_rrf, c_rrf)},
        "token_usage": {"prompt": prompt_tokens, "completion": completion_tokens},
        "per_query": per_query,
        "note": (
            f"raptor_lite_compare.py와 프롬프트·few-shot·ground truth·검색 로직 동일, "
            f"라우팅 모델만 로컬 EXAONE-3.5-7.8B → {ROUTER_EVAL_MODEL}로 교체."
        ),
    }

    # (2) gpt-4o-mini vs EXAONE — 같은 쿼리에 대한 짝지은 비교(가능할 때만)
    exaone = _load_exaone_per_query()
    if exaone:
        paired = [(q["case_name"], q["routed_hit"]) for q in per_query if q["case_name"] in exaone]
        b = sum(1 for name, hit in paired if hit and not exaone[name])
        c = sum(1 for name, hit in paired if not hit and exaone[name])
        result["vs_exaone"] = {
            "n_paired": len(paired),
            "exaone_hits": sum(exaone[name] for name, _ in paired),
            "openai_hits": sum(hit for _, hit in paired),
            "b_openai_only": b,
            "c_exaone_only": c,
            "p_value": _mcnemar_exact(b, c),
        }

    save_json(result, OUT_PATH)

    logger.info(f"========== 최종 결과 (n={n}, 모델={ROUTER_EVAL_MODEL}) ==========")
    logger.info(f"  RRF(파티션 없음):   {rrf_hits}/{n} ({rrf_hits / n * 100:.1f}%)")
    logger.info(f"  라우팅 후 검색:     {routed_hits}/{n} ({routed_hits / n * 100:.1f}%)")
    logger.info(f"  라우팅 자체 정답률: {routing_correct}/{n} ({routing_correct / n * 100:.1f}%)")
    logger.info(f"  vs RRF McNemar p={result['mcnemar_vs_rrf']['p_value']:.4g} (b={b_rrf}, c={c_rrf})")
    if "vs_exaone" in result:
        v = result["vs_exaone"]
        logger.info(
            f"  vs EXAONE(짝지은 {v['n_paired']}건): OpenAI {v['openai_hits']} vs EXAONE {v['exaone_hits']} "
            f"| b={v['b_openai_only']} c={v['c_exaone_only']} p={v['p_value']:.4g}"
        )
    logger.info(f"  토큰: prompt={prompt_tokens:,} completion={completion_tokens:,}")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="쿼리 수 제한(스모크 테스트용)")
    args = parser.parse_args()
    main(limit=args.limit)
