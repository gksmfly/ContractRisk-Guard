# scripts/crawl_interpretation.py
"""
국가법령정보 API - 법령해석례 전체 수집 스크립트

사용법:
    python scripts/crawl_interpretation.py --key <인증키>
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
        logging.FileHandler(LOG_DIR / "crawl_interpretation.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

LIST_URL = os.environ.get("LAW_LIST_URL", "https://www.law.go.kr/DRF/lawSearch.do")
DETAIL_URL = os.environ.get("LAW_DETAIL_URL", "https://www.law.go.kr/DRF/lawService.do")
RAW_DIR = Path(os.environ.get("EXPC_RAW_DIR", "data/raw/interpretations"))


def fetch_list(auth_key: str, page: int = 1, display: int = 100) -> dict:
    """법령해석례 목록을 페이지 단위로 조회합니다."""
    params = {
        "OC": auth_key,
        "target": "expc",
        "type": "JSON",
        "page": page,
        "display": display,
    }
    response = requests.get(LIST_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_detail(auth_key: str, expc_id: str) -> dict | None:
    """해석례일련번호로 해석례 상세 내용을 조회합니다."""
    params = {
        "OC": auth_key,
        "target": "expc",
        "ID": expc_id,
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
    """전체 해석례 목록에서 ID와 제목을 수집합니다."""
    id_list = []
    page = 1

    while True:
        logger.info(f"해석례 목록 조회 - 페이지 {page}")
        result = fetch_list(auth_key, page=page)

        expcs = result.get("Expc", {}).get("expc", [])
        if not expcs:
            break

        if isinstance(expcs, dict):
            expcs = [expcs]

        for expc in expcs:
            expc_id = expc.get("법령해석례일련번호") or expc.get("id")
            title = expc.get("안건명") or ""
            if expc_id:
                id_list.append({"id": str(expc_id), "title": title})

        logger.info(f"페이지 {page} - {len(expcs)}건 (누적: {len(id_list)}건)")

        total_count = int(result.get("Expc", {}).get("totalCnt", 0))
        display = int(result.get("Expc", {}).get("numOfRows", 100))
        if page * display >= total_count:
            break

        page += 1
        time.sleep(delay)

    return id_list


def crawl_all(auth_key: str, delay: float = 1.0) -> None:
    """법령해석례 전체를 수집합니다."""
    logger.info("=== 법령해석례 전체 수집 시작 ===")

    # 1단계: 해석례 목록 수집
    id_list = collect_id_list(auth_key, delay)
    logger.info(f"해석례 목록 수집 완료 - 총 {len(id_list)}건")

    save_json(
        {"total": len(id_list), "interpretations": id_list},
        RAW_DIR / "expc_list.json",
    )

    # 2단계: 각 해석례 상세 수집
    success, skipped, failed = 0, 0, 0

    for i, item in enumerate(id_list, 1):
        expc_id = item["id"]
        filepath = RAW_DIR / f"{expc_id}.json"

        if filepath.exists():
            skipped += 1
            continue

        for attempt in range(1, 4):
            try:
                data = fetch_detail(auth_key, expc_id)
                if data:
                    save_json(data, filepath)
                    logger.info(f"[{i}/{len(id_list)}] 저장 완료 - {item['title']} (ID: {expc_id})")
                    success += 1
                else:
                    logger.warning(f"[{i}/{len(id_list)}] 데이터 없음 - ID: {expc_id}")
                    failed += 1
                break
            except requests.RequestException as e:
                if attempt < 3:
                    logger.warning(f"[{i}/{len(id_list)}] 재시도 {attempt}/3 - ID: {expc_id} | {e}")
                    time.sleep(delay * attempt)
                else:
                    logger.error(f"[{i}/{len(id_list)}] 최종 실패 - ID: {expc_id} | {e}")
                    failed += 1

        time.sleep(delay)

    logger.info(f"=== 해석례 수집 완료 - 성공: {success}, 건너뜀: {skipped}, 실패: {failed} ===")


def main():
    parser = argparse.ArgumentParser(description="국가법령정보 API - 법령해석례 전체 수집")
    parser.add_argument("--key", required=True, help="API 인증키")
    parser.add_argument("--delay", type=float, default=1.0, help="API 호출 간격(초)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    crawl_all(args.key, delay=args.delay)


if __name__ == "__main__":
    main()
