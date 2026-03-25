"""
국가법령정보 API를 사용하여 도메인별 법령 데이터를 수집하는 스크립트

분석 도메인:
    1. 해지 조항: 일방적 해지, 해지 통보 기간, 위약금, 해지 사유 모호성
    2. 책임 제한 조항: 전면 면책, 간접손해 배제, 손해배상 상한, 고의·중과실 배제 여부

사용법:
    python scripts/crawl_law.py --key <인증키>                          # 전체 도메인 법령 수집
    python scripts/crawl_law.py --key <인증키> --domain termination     # 해지 조항 관련 법령만
    python scripts/crawl_law.py --key <인증키> --domain liability       # 책임 제한 관련 법령만
    python scripts/crawl_law.py --key <인증키> --query 약관             # 키워드로 법령 검색
    python scripts/crawl_law.py --key <인증키> --mst 253527            # 특정 MST 법령 수집
"""

import requests
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Optional

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/crawl.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

BASE_URL = "http://www.law.go.kr/DRF/lawService.do"
LAW_LIST_URL = "http://www.law.go.kr/DRF/lawSearch.do"
RAW_DATA_DIR = Path("data/raw")

# ──────────────────────────────────────────────
# 도메인별 관련 법령 정의 (B2C 기업-개인 서비스 기준)
# ──────────────────────────────────────────────
DOMAIN_LAWS = {
    "termination": {
        "name": "해지 조항",
        "description": "일방적 해지, 해지 통보 기간, 위약금, 해지 사유 모호성",
        "laws": [
            {"name": "약관의 규제에 관한 법률", "query": "약관의 규제에 관한 법률", "sections": "§9 (계약의 해제·해지)"},
            {"name": "민법", "query": "민법", "sections": "§543~553 (계약의 해지·해제)"},
            {"name": "전자상거래 등에서의 소비자보호에 관한 법률", "query": "전자상거래 등에서의 소비자보호에 관한 법률", "sections": "§17 (청약철회)"},
            {"name": "상법", "query": "상법", "sections": "§64~69 (상행위 해지 규정)"},
        ],
    },
    "liability": {
        "name": "책임 제한 조항",
        "description": "전면 면책, 간접손해 배제, 손해배상 상한, 고의·중과실 배제 여부",
        "laws": [
            {"name": "약관의 규제에 관한 법률", "query": "약관의 규제에 관한 법률", "sections": "§7 (면책조항의 금지)"},
            {"name": "민법", "query": "민법", "sections": "§750~766 (불법행위·손해배상)"},
            {"name": "전자상거래 등에서의 소비자보호에 관한 법률", "query": "전자상거래 등에서의 소비자보호에 관한 법률", "sections": "§21 (금지행위·면책제한)"},
            {"name": "소비자기본법", "query": "소비자기본법", "sections": "§19 (사업자의 책임)"},
            {"name": "제조물 책임법", "query": "제조물 책임법", "sections": "§3~4 (제조물 결함 책임)"},
        ],
    },
}


def get_domain_law_queries(domain: Optional[str] = None) -> list[dict]:
    """
    도메인에 해당하는 법령 목록을 반환합니다.
    도메인 미지정 시 전체 도메인의 법령을 중복 없이 반환합니다.

    Args:
        domain: 도메인 키 ("termination", "liability") 또는 None(전체)

    Returns:
        법령 정보 딕셔너리 리스트 (중복 제거)
    """
    if domain:
        if domain not in DOMAIN_LAWS:
            raise ValueError(f"알 수 없는 도메인: {domain}. 사용 가능: {list(DOMAIN_LAWS.keys())}")
        return DOMAIN_LAWS[domain]["laws"]

    # 전체 도메인: 법령명 기준 중복 제거
    seen = set()
    all_laws = []
    for d in DOMAIN_LAWS.values():
        for law in d["laws"]:
            if law["name"] not in seen:
                seen.add(law["name"])
                all_laws.append(law)
    return all_laws


def fetch_law_list(auth_key: str, query: str = "", page: int = 1, display: int = 100) -> dict:
    """
    법령 목록을 키워드로 검색합니다.

    Args:
        auth_key: API 인증키
        query: 검색 키워드 (법령명)
        page: 페이지 번호 (1부터 시작)
        display: 페이지당 결과 수 (최대 100)

    Returns:
        법령 목록 JSON 딕셔너리
    """
    params = {
        "OC": auth_key,
        "target": "law",
        "type": "JSON",
        "page": page,
        "display": display,
        "query": query,
    }
    response = requests.get(LAW_LIST_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_law_by_mst(auth_key: str, mst_id: str) -> dict | None:
    """
    법령일련번호(MST)로 특정 법령의 전체 내용을 조회합니다.

    Args:
        auth_key: API 인증키
        mst_id: 법령일련번호

    Returns:
        법령 전체 내용 JSON 딕셔너리, 실패 시 None
    """
    params = {
        "OC": auth_key,
        "target": "law",
        "MST": mst_id,
        "type": "JSON",
    }
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def save_json(data: dict, filepath: Path) -> None:
    """JSON 데이터를 파일로 저장합니다."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def search_law_mst(auth_key: str, query: str) -> list[dict]:
    """
    키워드로 법령을 검색하여 MST와 법령명 목록을 반환합니다.

    Args:
        auth_key: API 인증키
        query: 검색 키워드

    Returns:
        [{"mst": "...", "name": "..."}, ...] 형태의 리스트
    """
    result = fetch_law_list(auth_key, query=query, page=1, display=20)
    laws = result.get("LawSearch", {}).get("law", [])

    if isinstance(laws, dict):
        laws = [laws]

    found = []
    for law in laws:
        mst = law.get("법령일련번호") or law.get("MST")
        name = law.get("법령명한글") or law.get("법령명") or ""
        if mst:
            found.append({"mst": str(mst), "name": name})

    return found


def crawl_single_law(auth_key: str, mst_id: str) -> None:
    """
    단일 법령을 MST 번호로 수집하여 저장합니다.

    Args:
        auth_key: API 인증키
        mst_id: 법령일련번호
    """
    logger.info(f"법령 수집 시작 - MST: {mst_id}")
    filepath = RAW_DATA_DIR / f"{mst_id}.json"

    if filepath.exists():
        logger.info(f"이미 수집된 법령입니다 - MST: {mst_id}, 건너뜁니다.")
        return

    data = fetch_law_by_mst(auth_key, mst_id)
    if data:
        save_json(data, filepath)
        logger.info(f"저장 완료 - {filepath}")
    else:
        logger.warning(f"데이터 없음 - MST: {mst_id}")


def crawl_domain_laws(auth_key: str, domain: Optional[str] = None, delay: float = 1.0) -> None:
    """
    도메인별 관련 법령을 검색하여 수집합니다.

    1단계: 각 법령명으로 API 검색 → MST 번호 확보
    2단계: MST 번호로 법령 상세 내용 수집

    Args:
        auth_key: API 인증키
        domain: 도메인 키 ("termination", "liability") 또는 None(전체)
        delay: API 호출 간격 (초)
    """
    law_entries = get_domain_law_queries(domain)
    domain_label = DOMAIN_LAWS[domain]["name"] if domain else "전체 도메인"
    logger.info(f"[{domain_label}] 법령 수집 시작 - 대상 법령 {len(law_entries)}건")

    # 1단계: 법령명으로 검색하여 MST 확보
    mst_targets = []
    seen_mst = set()

    for entry in law_entries:
        logger.info(f"법령 검색 중: {entry['name']} (관련 조문: {entry['sections']})")
        try:
            results = search_law_mst(auth_key, entry["query"])
            # 정확히 일치하는 법령 우선, 없으면 첫 번째 결과 사용
            matched = None
            for r in results:
                if entry["name"] in r["name"]:
                    matched = r
                    break
            if not matched and results:
                matched = results[0]

            if matched and matched["mst"] not in seen_mst:
                seen_mst.add(matched["mst"])
                mst_targets.append({
                    "mst": matched["mst"],
                    "name": matched["name"],
                    "sections": entry["sections"],
                })
                logger.info(f"  → 찾음: {matched['name']} (MST: {matched['mst']})")
            elif matched:
                logger.info(f"  → 이미 목록에 포함됨: {matched['name']}")
            else:
                logger.warning(f"  → 검색 결과 없음: {entry['name']}")
        except requests.RequestException as e:
            logger.error(f"  → 검색 실패: {entry['name']} | 오류: {e}")

        time.sleep(delay)

    logger.info(f"검색 완료 - 수집 대상 법령 {len(mst_targets)}건")

    # 수집 대상 목록 저장
    domain_key = domain or "all"
    list_path = RAW_DATA_DIR / f"domain_{domain_key}_targets.json"
    save_json({"domain": domain_label, "total": len(mst_targets), "laws": mst_targets}, list_path)

    # 2단계: 각 법령 상세 내용 수집
    success, skipped, failed = 0, 0, 0

    for i, target in enumerate(mst_targets, 1):
        mst_id = target["mst"]
        filepath = RAW_DATA_DIR / f"{mst_id}.json"

        if filepath.exists():
            logger.info(f"[{i}/{len(mst_targets)}] 건너뜀 (이미 존재) - {target['name']}")
            skipped += 1
            continue

        try:
            data = fetch_law_by_mst(auth_key, mst_id)
            if data:
                save_json(data, filepath)
                logger.info(f"[{i}/{len(mst_targets)}] 저장 완료 - {target['name']} (MST: {mst_id})")
                success += 1
            else:
                logger.warning(f"[{i}/{len(mst_targets)}] 데이터 없음 - {target['name']}")
                failed += 1
        except requests.RequestException as e:
            logger.error(f"[{i}/{len(mst_targets)}] 요청 실패 - {target['name']} | 오류: {e}")
            failed += 1

        time.sleep(delay)

    logger.info(
        f"[{domain_label}] 수집 완료 - 성공: {success}건, 건너뜀: {skipped}건, 실패: {failed}건"
    )


def crawl_by_query(auth_key: str, query: str, delay: float = 1.0) -> None:
    """
    키워드로 법령을 검색하고 결과를 수집합니다.
    새로운 관련 법령을 탐색할 때 사용합니다.

    Args:
        auth_key: API 인증키
        query: 검색 키워드
        delay: API 호출 간격 (초)
    """
    logger.info(f"키워드 검색: '{query}'")
    results = search_law_mst(auth_key, query)

    if not results:
        logger.warning(f"검색 결과 없음: '{query}'")
        return

    logger.info(f"검색 결과 {len(results)}건:")
    for r in results:
        logger.info(f"  - {r['name']} (MST: {r['mst']})")

    for i, r in enumerate(results, 1):
        filepath = RAW_DATA_DIR / f"{r['mst']}.json"
        if filepath.exists():
            logger.info(f"[{i}/{len(results)}] 건너뜀 (이미 존재) - {r['name']}")
            continue

        try:
            data = fetch_law_by_mst(auth_key, r["mst"])
            if data:
                save_json(data, filepath)
                logger.info(f"[{i}/{len(results)}] 저장 완료 - {r['name']}")
        except requests.RequestException as e:
            logger.error(f"[{i}/{len(results)}] 요청 실패 - {r['name']} | 오류: {e}")

        time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description="국가법령정보 API 도메인별 데이터 수집기")
    parser.add_argument("--key", required=True, help="국가법령정보 API 인증키")
    parser.add_argument(
        "--domain",
        choices=["termination", "liability"],
        default=None,
        help="수집 도메인: termination(해지 조항), liability(책임 제한 조항). 미지정 시 전체 도메인.",
    )
    parser.add_argument(
        "--mst",
        default=None,
        help="특정 법령일련번호(MST)로 단일 법령 수집",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="키워드로 법령 검색 및 수집 (새 법령 탐색용)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="API 호출 간격(초), 기본값: 1.0",
    )
    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.mst:
        crawl_single_law(args.key, args.mst)
    elif args.query:
        crawl_by_query(args.key, args.query, delay=args.delay)
    else:
        crawl_domain_laws(args.key, domain=args.domain, delay=args.delay)


if __name__ == "__main__":
    main()
