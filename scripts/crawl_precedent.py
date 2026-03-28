# scripts/crawl_precedent.py
"""
국가법령정보 API - 판례 전체 수집 스크립트

사용법:
    python scripts/crawl_precedent.py --key <인증키>
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

LOG_DIR = Path(os.environ["LOG_DIR"])
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "crawl_precedent.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

LIST_URL = os.environ["LAW_LIST_URL"]
DETAIL_URL = os.environ["LAW_DETAIL_URL"]
RAW_DIR = Path(os.environ["PREC_RAW_DIR"])


def fetch_list(auth_key: str, page: int = 1, display: int = 100) -> dict:
    """판례 목록을 페이지 단위로 조회합니다."""
    params = {
        "OC": auth_key,
        "target": "prec",
        "type": "JSON",
        "page": page,
        "display": display,
    }
    response = requests.get(LIST_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_detail(auth_key: str, prec_id: str) -> dict | None:
    """판례일련번호로 판례 상세 내용을 조회합니다."""
    params = {
        "OC": auth_key,
        "target": "prec",
        "ID": prec_id,
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


def collect_id_list(auth_key: str, delay: float) -> list[dict]:
    """전체 판례 목록에서 ID와 사건명을 수집합니다."""
    id_list = []
    page = 1

    while True:
        logger.info(f"판례 목록 조회 - 페이지 {page}")
        result = fetch_list(auth_key, page=page)

        precs = result.get("PrecSearch", {}).get("prec", [])
        if not precs:
            break

        if isinstance(precs, dict):
            precs = [precs]

        for prec in precs:
            prec_id = prec.get("판례일련번호") or prec.get("ID")
            name = prec.get("사건명") or ""
            if prec_id:
                id_list.append({"id": str(prec_id), "name": name})

        logger.info(f"페이지 {page} - {len(precs)}건 (누적: {len(id_list)}건)")

        total_count = int(result.get("PrecSearch", {}).get("totalCnt", 0))
        display = int(result.get("PrecSearch", {}).get("numOfRows", 100))
        if page * display >= total_count:
            break

        page += 1
        time.sleep(delay)

    return id_list


def crawl_all(auth_key: str, delay: float = 1.0) -> None:
    """판례 전체를 수집합니다."""
    logger.info("=== 판례 전체 수집 시작 ===")

    # 1단계: 판례 목록 수집
    id_list = collect_id_list(auth_key, delay)
    logger.info(f"판례 목록 수집 완료 - 총 {len(id_list)}건")

    save_json(
        {"total": len(id_list), "precedents": id_list},
        RAW_DIR / "prec_list.json",
    )

    # 2단계: 각 판례 상세 수집
    success, skipped, failed = 0, 0, 0

    for i, item in enumerate(id_list, 1):
        prec_id = item["id"]
        filepath = RAW_DIR / f"{prec_id}.json"

        if filepath.exists():
            skipped += 1
            continue

        for attempt in range(1, 4):
            try:
                data = fetch_detail(auth_key, prec_id)
                if data:
                    save_json(data, filepath)
                    logger.info(f"[{i}/{len(id_list)}] 저장 완료 - {item['name']} (ID: {prec_id})")
                    success += 1
                else:
                    logger.warning(f"[{i}/{len(id_list)}] 데이터 없음 - ID: {prec_id}")
                    failed += 1
                break
            except requests.RequestException as e:
                if attempt < 3:
                    logger.warning(f"[{i}/{len(id_list)}] 재시도 {attempt}/3 - ID: {prec_id} | {e}")
                    time.sleep(delay * attempt)
                else:
                    logger.error(f"[{i}/{len(id_list)}] 최종 실패 - ID: {prec_id} | {e}")
                    failed += 1

        time.sleep(delay)

    logger.info(f"=== 판례 수집 완료 - 성공: {success}, 건너뜀: {skipped}, 실패: {failed} ===")


def main():
    parser = argparse.ArgumentParser(description="국가법령정보 API - 판례 전체 수집")
    parser.add_argument("--key", required=True, help="API 인증키")
    parser.add_argument("--delay", type=float, default=1.0, help="API 호출 간격(초)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    crawl_all(args.key, delay=args.delay)


if __name__ == "__main__":
    main()
