# backend/api/services/latency_benchmark.py
"""
run_analyze() 처리 속도 측정

조항마다 GPT(FORWARD_MODEL) 호출 + KoELECTRA 추론 + pgvector 검색을
순차로(for 루프) 처리하는 구조라, 조항 수가 많은 계약서일수록 선형으로
느려질 것으로 예상된다. 실제로 몇 초 걸리는지 조항 수별로 측정해서
사용자가 "분석하기"를 누르고 얼마나 기다리는지 가늠한다.

실행:
  python -m backend.api.services.latency_benchmark
  python -m backend.api.services.latency_benchmark --sizes 1,5,10,20
"""

import argparse
import asyncio
import time

from backend.api.services.analyze import run_analyze
from backend.utils import load_logger

logger = load_logger("latency_benchmark.log")

# 실제 조항 스타일 문구를 domain·risk_level 다양하게 순환시켜 표본 계약서를 구성한다
_CLAUSE_TEMPLATES = [
    "제{n}조(해지) 회사는 이용자에게 사전 통보 없이 언제든지 본 서비스 이용계약을 즉시 해지할 수 있으며, "
    "이로 인해 발생하는 손해에 대하여 회사는 어떠한 책임도 지지 아니한다.",
    "제{n}조(손해배상) 회사의 귀책사유로 인한 손해에 대해서는 직접 손해에 한하여 배상하며, "
    "간접손해·특별손해·기대이익 손실에 대해서는 배상하지 아니한다.",
    "제{n}조(계약 해지) 이용자는 언제든지 서비스 해지를 신청할 수 있으며, 회사는 30일 이내에 처리한다. "
    "단, 이용자가 요금을 미납한 경우 회사는 14일 이상의 유예기간을 부여한 후 해지할 수 있다.",
    "제{n}조(개인정보) 회사는 관련 법령에 따라 이용자의 개인정보를 안전하게 관리하며, "
    "수집 목적 달성 시 지체 없이 파기한다.",
]


def _build_contract(n_clauses: int) -> str:
    clauses = [
        _CLAUSE_TEMPLATES[i % len(_CLAUSE_TEMPLATES)].format(n=i + 1)
        for i in range(n_clauses)
    ]
    return "\n\n".join(clauses)


async def _measure(n_clauses: int) -> dict:
    text = _build_contract(n_clauses)
    t0 = time.time()
    result = await run_analyze(text)
    elapsed = time.time() - t0
    return {
        "n_clauses_input":    n_clauses,
        "n_clauses_analyzed": result.total_clauses,
        "elapsed_sec":        round(elapsed, 2),
        "sec_per_clause":     round(elapsed / max(result.total_clauses, 1), 2),
    }


async def main_async(sizes: list[int]) -> None:
    logger.info(f"{'조항 수(입력)':>12} {'조항 수(분석됨)':>14} {'총 소요(초)':>12} {'조항당(초)':>10}")
    logger.info("-" * 55)
    for n in sizes:
        r = await _measure(n)
        logger.info(f"{r['n_clauses_input']:>12} {r['n_clauses_analyzed']:>14} "
                    f"{r['elapsed_sec']:>12.2f} {r['sec_per_clause']:>10.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="run_analyze() 처리 속도 측정")
    parser.add_argument("--sizes", type=str, default="1,5,10,20",
                        help="측정할 조항 개수 목록, 콤마로 구분 (기본: 1,5,10,20)")
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    asyncio.run(main_async(sizes))


if __name__ == "__main__":
    main()
