"""
국가법령정보 API를 사용하여 법령 전체 데이터를 수집하는 스크립트

사용법:
    python scripts/crawl_law.py --key <인증키>
    python scripts/crawl_law.py --key <인증키> --mst 253527
"""

import requests
import json
import time
import argparse
import logging
from pathlib import Path

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


def fetch_law_list(auth_key: str, page: int = 1, display: int = 100) -> dict:
    """
    법령 목록을 페이지 단위로 조회합니다.

    Args:
        auth_key: API 인증키
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
        "query": "",  # 전체 법령 조회
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


def crawl_all_laws(auth_key: str, delay: float = 1.0) -> None:
    """
    국가법령정보 API에서 전체 법령 목록을 수집한 뒤,
    각 법령의 상세 내용을 순차적으로 수집하여 저장합니다.

    Args:
        auth_key: API 인증키
        delay: API 호출 간격 (초), 서버 부하 방지
    """
    logger.info("전체 법령 목록 수집 시작")

    # 1단계: 법령 목록 전체 수집
    mst_list = []
    page = 1
    while True:
        logger.info(f"법령 목록 조회 중 - 페이지 {page}")
        result = fetch_law_list(auth_key, page=page)

        laws = result.get("LawSearch", {}).get("law", [])
        if not laws:
            logger.info("더 이상 법령 목록이 없습니다. 목록 수집 완료.")
            break

        # 단일 결과인 경우 리스트로 변환
        if isinstance(laws, dict):
            laws = [laws]

        for law in laws:
            mst = law.get("법령일련번호") or law.get("MST")
            if mst:
                mst_list.append(str(mst))

        logger.info(f"페이지 {page} - {len(laws)}건 수집 (누적: {len(mst_list)}건)")

        # 마지막 페이지 여부 확인
        total_count = int(result.get("LawSearch", {}).get("totalCnt", 0))
        display = int(result.get("LawSearch", {}).get("numOfRows", 100))
        if page * display >= total_count:
            break

        page += 1
        time.sleep(delay)

    logger.info(f"법령 목록 수집 완료 - 총 {len(mst_list)}건")

    # 목록 저장
    list_path = RAW_DATA_DIR / "mst_list.json"
    save_json({"total": len(mst_list), "mst_list": mst_list}, list_path)
    logger.info(f"MST 목록 저장 완료 - {list_path}")

    # 2단계: 각 법령 상세 내용 수집
    logger.info("법령 상세 내용 수집 시작")
    success, skipped, failed = 0, 0, 0

    for i, mst_id in enumerate(mst_list, 1):
        filepath = RAW_DATA_DIR / f"{mst_id}.json"
        if filepath.exists():
            logger.info(f"[{i}/{len(mst_list)}] 건너뜀 (이미 존재) - MST: {mst_id}")
            skipped += 1
            continue

        try:
            data = fetch_law_by_mst(auth_key, mst_id)
            if data:
                save_json(data, filepath)
                logger.info(f"[{i}/{len(mst_list)}] 저장 완료 - MST: {mst_id}")
                success += 1
            else:
                logger.warning(f"[{i}/{len(mst_list)}] 데이터 없음 - MST: {mst_id}")
                failed += 1
        except requests.RequestException as e:
            logger.error(f"[{i}/{len(mst_list)}] 요청 실패 - MST: {mst_id} | 오류: {e}")
            failed += 1

        time.sleep(delay)

    logger.info(
        f"전체 수집 완료 - 성공: {success}건, 건너뜀: {skipped}건, 실패: {failed}건"
    )


def main():
    parser = argparse.ArgumentParser(description="국가법령정보 API 데이터 수집기")
    parser.add_argument("--key", required=True, help="국가법령정보 API 인증키")
    parser.add_argument(
        "--mst",
        default=None,
        help="특정 법령일련번호(MST). 미입력 시 전체 법령 수집.",
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
    else:
        crawl_all_laws(args.key, delay=args.delay)


if __name__ == "__main__":
    main()
