# scripts/crawl_law.py
"""
국가법령정보 API - 법령 전체 수집 스크립트

사용법:
    python scripts/crawl_law.py --key <인증키>
"""

import os
import requests
import json
import time
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

LOG_DIR = Path(os.environ.get("LOG_DIR", "data/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "crawl_law.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

LIST_URL = os.environ.get("LAW_LIST_URL", "https://www.law.go.kr/DRF/lawSearch.do")
DETAIL_URL = os.environ.get("LAW_DETAIL_URL", "https://www.law.go.kr/DRF/lawService.do")
RAW_DIR = Path(os.environ.get("LAW_RAW_DIR", "data/raw/laws"))


def fetch_list(auth_key: str, page: int = 1, display: int = 100) -> dict:
    """법령 목록을 페이지 단위로 조회합니다."""
    params = {
        "OC": auth_key,
        "target": "law",
        "type": "JSON",
        "page": page,
        "display": display,
    }
    response = requests.get(LIST_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_detail(auth_key: str, mst_id: str) -> dict | None:
    """법령일련번호(MST)로 법령 상세 내용을 조회합니다."""
    params = {
        "OC": auth_key,
        "target": "law",
        "MST": mst_id,
        "type": "JSON",
    }
    response = requests.get(DETAIL_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def save_json(data: dict, filepath: Path) -> None:
    """JSON 데이터를 파일로 저장합니다."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def collect_mst_list(auth_key: str, delay: float) -> list[str]:
    """전체 법령 목록에서 MST 번호를 수집합니다."""
    mst_list = []
    page = 1

    while True:
        logger.info(f"법령 목록 조회 - 페이지 {page}")
        result = fetch_list(auth_key, page=page)

        laws = result.get("LawSearch", {}).get("law", [])
        if not laws:
            break

        if isinstance(laws, dict):
            laws = [laws]

        for law in laws:
            mst = law.get("법령일련번호") or law.get("MST")
            if mst:
                mst_list.append(str(mst))

        logger.info(f"페이지 {page} - {len(laws)}건 (누적: {len(mst_list)}건)")

        total_count = int(result.get("LawSearch", {}).get("totalCnt", 0))
        display = int(result.get("LawSearch", {}).get("numOfRows", 100))
        if page * display >= total_count:
            break

        page += 1
        time.sleep(delay)

    return mst_list


def crawl_all(auth_key: str, delay: float = 1.0) -> None:
    """법령 전체를 수집합니다."""
    logger.info("=== 법령 전체 수집 시작 ===")

    # 1단계: MST 목록 수집
    mst_list = collect_mst_list(auth_key, delay)
    logger.info(f"법령 목록 수집 완료 - 총 {len(mst_list)}건")

    save_json({"total": len(mst_list), "mst_list": mst_list}, RAW_DIR / "mst_list.json")

    # 2단계: 각 법령 상세 수집
    success, skipped, failed = 0, 0, 0

    for i, mst_id in enumerate(mst_list, 1):
        filepath = RAW_DIR / f"{mst_id}.json"

        if filepath.exists():
            skipped += 1
            continue

        for attempt in range(1, 4):
            try:
                data = fetch_detail(auth_key, mst_id)
                if data:
                    save_json(data, filepath)
                    logger.info(f"[{i}/{len(mst_list)}] 저장 완료 - MST: {mst_id}")
                    success += 1
                else:
                    logger.warning(f"[{i}/{len(mst_list)}] 데이터 없음 - MST: {mst_id}")
                    failed += 1
                break
            except requests.RequestException as e:
                if attempt < 3:
                    logger.warning(f"[{i}/{len(mst_list)}] 재시도 {attempt}/3 - MST: {mst_id} | {e}")
                    time.sleep(delay * attempt)
                else:
                    logger.error(f"[{i}/{len(mst_list)}] 최종 실패 - MST: {mst_id} | {e}")
                    failed += 1

        time.sleep(delay)

    logger.info(f"=== 법령 수집 완료 - 성공: {success}, 건너뜀: {skipped}, 실패: {failed} ===")


def main():
    parser = argparse.ArgumentParser(description="국가법령정보 API - 법령 전체 수집")
    parser.add_argument("--key", required=True, help="API 인증키")
    parser.add_argument("--delay", type=float, default=1.0, help="API 호출 간격(초)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    crawl_all(args.key, delay=args.delay)


if __name__ == "__main__":
    main()
