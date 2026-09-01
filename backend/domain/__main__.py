# backend/domain/__main__.py
import argparse
import json
from typing import Any

from backend.domain.common import logger
from backend.domain.config import DOMAIN_DIR, EXPC_DIR, LAW_DIR, PREC_DIR
from backend.domain.filter_interpretation import filter_interpretations
from backend.domain.filter_law import filter_laws
from backend.domain.filter_precedent import filter_precedents
from backend.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="도메인 필터링 — 법령명 메타데이터 + Dense Retrieval")
    parser.add_argument("--device", default="cuda:1", help="임베딩 디바이스 (예: cpu, cuda:0, cuda:1)")
    parser.add_argument("--source", default="all",
                        choices=["all", "law", "interpretation", "precedent"],
                        help="처리할 소스 (기본: all)")
    args = parser.parse_args()

    run_law  = args.source in ("all", "law")
    run_expc = args.source in ("all", "interpretation")
    run_prec = args.source in ("all", "precedent")

    logger.info("========== 도메인 필터링 시작 ==========")
    report_path = DOMAIN_DIR / "filter_report.json"
    existing = {}
    if report_path.exists():
        try:
            existing = json.loads(report_path.read_text(encoding="utf-8")).get("필터링_결과", {})
        except Exception:
            pass
    report: dict[str, Any] = existing

    if run_law:
        if LAW_DIR.exists():
            report["law"] = filter_laws(LAW_DIR, DOMAIN_DIR / "law")
        else:
            logger.warning(f"  법령 디렉토리 없음: {LAW_DIR}")

    if run_expc:
        if EXPC_DIR.exists():
            report["interpretation"] = filter_interpretations(
                EXPC_DIR, DOMAIN_DIR / "commentary",
                device=args.device,
            )
        else:
            logger.warning(f"  해석례 디렉토리 없음: {EXPC_DIR}")

    if run_prec:
        if PREC_DIR.exists():
            report["precedent"] = filter_precedents(
                PREC_DIR, DOMAIN_DIR / "case",
                device=args.device,
            )
        else:
            logger.warning(f"  판례 디렉토리 없음: {PREC_DIR}")

    save_json({"필터링_결과": report}, DOMAIN_DIR / "filter_report.json")
    logger.info("========== 도메인 필터링 완료 ==========")


if __name__ == "__main__":
    main()
