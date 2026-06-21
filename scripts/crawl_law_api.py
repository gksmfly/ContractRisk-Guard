# scripts/crawl_law_api.py
"""
국가법령정보 API - 법령·판례·법령해석례 수집 스크립트

--source 인자로 수집 대상을 선택합니다.
수집 결과는 source별 디렉터리에 항목당 JSON 파일로 저장됩니다.

사용법:
    python scripts/crawl_law_api.py --source law --key <인증키>
    python scripts/crawl_law_api.py --source precedent --key <인증키>
    python scripts/crawl_law_api.py --source interpretation --key <인증키>
    python scripts/crawl_law_api.py --source law --key <인증키> --query 계약

샘플 출력 데이터:
    # 목록 파일 (data/raw/law/mst_list.json)
    {
        "total": 12,
        "query": "약관",
        "mst_list": ["123456", "234567"]
    }

    # 개별 항목 파일 (data/raw/law/123456.json)
    {
        "법령일련번호": "123456",
        "법령명한글": "약관의 규제에 관한 법률",
        "시행일자": "20240101",
        "조문": [{"조문번호": "1", "조문제목": "목적", "조문내용": "이 법은..."}]
    }
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import save_json, setup_logger, PROJECT_ROOT

LIST_URL = os.environ.get("LAW_LIST_URL", "https://www.law.go.kr/DRF/lawSearch.do")
DETAIL_URL = os.environ.get("LAW_DETAIL_URL", "https://www.law.go.kr/DRF/lawService.do")
DEFAULT_QUERY = "약관"

logger: logging.Logger  # main()에서 source별로 초기화


@dataclass
class SourceConfig:
    name: str
    display: str
    target: str
    list_wrapper: str
    items_key: str
    id_fields: list[str]
    id_param: str
    raw_dir_env: str
    raw_dir_default: str
    list_filename: str
    list_items_key: str
    log_filename: str
    extract_fn: Callable[[dict], dict]

"""법령 상세 응답에서 조항 분석에 필요한 필드만 추출합니다.

Args:
    raw (dict): fetch_detail() 반환값 (LawService 래핑 포함).

Returns:
    dict: 키: "법령일련번호", "법령명한글", "시행일자", "조문"
          조문이 없으면 빈 딕셔너리 반환.
"""

def extract_law_fields(raw: dict) -> dict:
    svc = raw.get("LawService", {})
    if not svc:
        return {}
    raw_articles = svc.get("조문", [])
    if isinstance(raw_articles, dict):
        raw_articles = [raw_articles]
    articles = [
        {
            "조문번호": str(a.get("조문번호", "")),
            "조문제목": str(a.get("조문제목", "")),
            "조문내용": str(a.get("조문내용", "")),
        }
        for a in raw_articles
        if a.get("조문내용")
    ]
    return {
        "법령일련번호": str(svc.get("법령일련번호", "")),
        "법령명한글": str(svc.get("법령명한글", "")),
        "시행일자": str(svc.get("시행일자", "")),
        "조문": articles,
    }


"""판례 상세 응답에서 조항 분석에 필요한 필드만 추출합니다.

Args:
    raw (dict): fetch_detail() 반환값 (PrecService 래핑 포함).

Returns:
    dict: 키: "판례일련번호", "사건명", "선고일자", "판시사항", "판결요지"
          판시사항·판결요지가 모두 없으면 빈 딕셔너리 반환.
"""

def extract_precedent_fields(raw: dict) -> dict:
    svc = raw.get("PrecService", {})
    if not svc:
        return {}
    판시사항 = str(svc.get("판시사항", "")).strip()
    판결요지 = str(svc.get("판결요지", "")).strip()
    판례내용 = str(svc.get("판례내용", "")).strip()
    if not 판시사항 and not 판결요지 and not 판례내용:
        return {}
    return {
        "판례일련번호": str(svc.get("판례일련번호", "")),
        "사건명": str(svc.get("사건명", "")),
        "선고일자": str(svc.get("선고일자", "")),
        "판시사항": 판시사항,
        "판결요지": 판결요지,
        "판례내용": 판례내용,
    }


"""법령해석례 상세 응답에서 조항 분석에 필요한 필드만 추출합니다.

Args:
    raw (dict): fetch_detail() 반환값 (ExpcService 래핑 포함).

Returns:
    dict: 키: "법령해석례일련번호", "안건명", "회신일자", "회신내용"
          회신내용이 없으면 빈 딕셔너리 반환.
"""

def extract_interpretation_fields(raw: dict) -> dict:
    svc = raw.get("ExpcService", {})
    if not svc:
        return {}
    회신내용 = str(svc.get("회신내용", "")).strip()
    if not 회신내용:
        return {}
    return {
        "법령해석례일련번호": str(svc.get("법령해석례일련번호", "")),
        "안건명": str(svc.get("안건명", "")),
        "회신일자": str(svc.get("회신일자", "")),
        "회신내용": 회신내용,
    }


SOURCES: dict[str, SourceConfig] = {
    "law": SourceConfig(
        name="law",
        display="법령",
        target="law",
        list_wrapper="LawSearch",
        items_key="law",
        id_fields=["법령일련번호", "MST"],
        id_param="MST",
        raw_dir_env="LAW_RAW_DIR",
        raw_dir_default=str(PROJECT_ROOT / "data/raw/law"),
        list_filename="mst_list.json",
        list_items_key="mst_list",
        log_filename="crawl_law.log",
        extract_fn=extract_law_fields,
    ),
    "precedent": SourceConfig(
        name="precedent",
        display="판례",
        target="prec",
        list_wrapper="PrecSearch",
        items_key="prec",
        id_fields=["판례일련번호", "ID"],
        id_param="ID",
        raw_dir_env="PREC_RAW_DIR",
        raw_dir_default=str(PROJECT_ROOT / "data/raw/case"),
        list_filename="prec_list.json",
        list_items_key="precedents",
        log_filename="crawl_precedent.log",
        extract_fn=extract_precedent_fields,
    ),
    "interpretation": SourceConfig(
        name="interpretation",
        display="법령해석례",
        target="expc",
        list_wrapper="Expc",
        items_key="expc",
        id_fields=["법령해석례일련번호", "id"],
        id_param="ID",
        raw_dir_env="EXPC_RAW_DIR",
        raw_dir_default=str(PROJECT_ROOT / "data/raw/commentary"),
        list_filename="expc_list.json",
        list_items_key="interpretations",
        log_filename="crawl_interpretation.log",
        extract_fn=extract_interpretation_fields,
    ),
}


"""목록 API를 호출하여 페이지 단위 응답을 반환합니다.

Args:
    auth_key (str): 국가법령정보 API 인증키.
    target (str): API target 파라미터 ("law", "prec", "expc").
    query (str): 검색 키워드.
    page (int): 조회할 페이지 번호 (1부터 시작).
    display (int): 페이지당 결과 수 (최대 100).

Returns:
    dict: API 응답 JSON 딕셔너리.
"""

def fetch_list(
    auth_key: str, target: str, query: str, page: int = 1, display: int = 100
) -> dict:
    params = {
        "OC": auth_key,
        "target": target,
        "type": "JSON",
        "query": query,
        "page": page,
        "display": display,
    }
    response = requests.get(LIST_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


"""항목 ID로 상세 내용을 조회합니다.

Args:
    auth_key (str): 국가법령정보 API 인증키.
    target (str): API target 파라미터.
    item_id (str): 항목 ID (법령일련번호, 판례일련번호 등).
    id_param (str): ID에 해당하는 API 파라미터명 ("MST" 또는 "ID").

Returns:
    dict | None: 상세 정보 JSON 딕셔너리. 실패 시 None.
"""

def fetch_detail(
    auth_key: str, target: str, item_id: str, id_param: str
) -> dict | None:
    params = {
        "OC": auth_key,
        "target": target,
        id_param: item_id,
        "type": "JSON",
    }
    response = requests.get(DETAIL_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


"""목록 API를 페이지 단위로 순회하며 항목 ID와 레이블을 수집합니다.

Args:
    auth_key (str): 국가법령정보 API 인증키.
    cfg (SourceConfig): 수집 대상 설정 객체.
    query (str): 검색 키워드.
    delay (float): 페이지 간 요청 대기 시간(초).

Returns:
    list[str]: 항목 ID 리스트. 예: ["123456", "234567"]
"""

def collect_id_list(
    auth_key: str, cfg: SourceConfig, query: str, delay: float
) -> list[str]:
    id_list: list[str] = []
    page = 1

    while True:
        logger.info(f"{cfg.display} 목록 조회 - 페이지 {page} (query='{query}')")
        result = fetch_list(auth_key, cfg.target, query=query, page=page)

        wrapper = result.get(cfg.list_wrapper, {})
        items = wrapper.get(cfg.items_key, [])
        if not items:
            break
        if isinstance(items, dict):
            items = [items]

        for item in items:
            item_id = next((item.get(f) for f in cfg.id_fields if item.get(f)), None)
            if item_id:
                id_list.append(str(item_id))

        logger.info(f"페이지 {page} - {len(items)}건 (누적: {len(id_list)}건)")

        total_count = int(wrapper.get("totalCnt", 0))
        display_count = int(wrapper.get("numOfRows", 100))
        if page * display_count >= total_count:
            break

        page += 1
        time.sleep(delay)

    return id_list


"""단일 항목을 최대 3회 재시도하며 수집하고 저장합니다.

Args:
    auth_key (str): 국가법령정보 API 인증키.
    cfg (SourceConfig): 수집 대상 설정 객체.
    item_id (str): 항목 ID (법령일련번호, 판례일련번호 등).
    filepath (Path): 저장할 JSON 파일 경로.
    progress (str): 로그 출력용 진행 상황 문자열 (예: "[1/100]").
    delay (float): 재시도 대기 시간 배수(초).

Returns:
    str: "success", "empty", "failed" 중 하나.
"""

def fetch_and_save(
    auth_key: str,
    cfg: SourceConfig,
    item_id: str,
    filepath: Path,
    progress: str,
    delay: float,
) -> str:
    for attempt in range(1, 4):
        try:
            raw = fetch_detail(auth_key, cfg.target, item_id, cfg.id_param)
            if not raw:
                logger.warning(f"{progress} 데이터 없음 - ID: {item_id}")
                return "failed"
            data = cfg.extract_fn(raw)
            if not data:
                logger.warning(f"{progress} 내용 없음 - ID: {item_id}")
                save_json(raw, filepath)
                return "empty"
            save_json(data, filepath)
            logger.info(f"{progress} 저장 완료 - ID: {item_id}")
            return "success"
        except requests.RequestException as e:
            if attempt < 3:
                logger.warning(f"{progress} 재시도 {attempt}/3 - ID: {item_id} | {e}")
                time.sleep(delay * attempt)
            else:
                logger.error(f"{progress} 최종 실패 - ID: {item_id} | {e}")
                return "failed"
    return "failed"


"""키워드로 필터된 항목을 수집하고 저장합니다.

목록을 먼저 수집한 후 각 항목의 상세 데이터를 개별 JSON 파일로 저장합니다.
이미 저장된 파일은 건너뜁니다.

Args:
    auth_key (str): 국가법령정보 API 인증키.
    cfg (SourceConfig): 수집 대상 설정 객체.
    query (str): 검색 키워드 (기본값: "약관").
    delay (float): API 호출 간 대기 시간(초).
"""

def _is_valid(filepath: Path) -> bool:
    try:
        with open(filepath) as f:
            d = json.load(f)
        return "Law" not in d
    except Exception:
        return False


def crawl_all(
    auth_key: str, cfg: SourceConfig, query: str = DEFAULT_QUERY, delay: float = 1.0
) -> None:
    raw_dir = Path(os.environ.get(cfg.raw_dir_env, cfg.raw_dir_default))
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== {cfg.display} 수집 시작 (query='{query}') ===")

    list_file = raw_dir / cfg.list_filename
    if list_file.exists():
        with open(list_file) as f:
            saved = json.load(f)
        id_list = [str(item) if not isinstance(item, dict) else str(item.get("id", ""))
                   for item in saved.get(cfg.list_items_key, [])]
        id_list = [i for i in id_list if i]
        logger.info(f"{cfg.display} 기존 목록 로드 - 총 {len(id_list)}건")
    else:
        id_list = collect_id_list(auth_key, cfg, query=query, delay=delay)
        logger.info(f"{cfg.display} 목록 수집 완료 - 총 {len(id_list)}건")
        save_json(
            {"total": len(id_list), "query": query, cfg.list_items_key: id_list},
            list_file,
        )

    success, skipped, failed = 0, 0, 0
    for i, item_id in enumerate(id_list, 1):
        filepath = raw_dir / f"{item_id}.json"
        if filepath.exists():
            skipped += 1
            continue

        result = fetch_and_save(
            auth_key, cfg, item_id, filepath,
            progress=f"[{i}/{len(id_list)}]",
            delay=delay,
        )
        if result == "success":
            success += 1
        else:
            failed += 1

        time.sleep(delay)

    logger.info(
        f"=== {cfg.display} 수집 완료 - 성공: {success}, 건너뜀: {skipped}, 실패: {failed} ==="
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="국가법령정보 API - 법령·판례·법령해석례 수집"
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=list(SOURCES.keys()),
        help="수집 대상 (law / precedent / interpretation)",
    )
    parser.add_argument("--key", required=True, help="API 인증키")
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help=f"검색 키워드 (기본값: '{DEFAULT_QUERY}')",
    )
    parser.add_argument("--delay", type=float, default=1.0, help="API 호출 간격(초)")
    args = parser.parse_args()

    cfg = SOURCES[args.source]

    global logger
    logger = setup_logger(cfg.log_filename)

    crawl_all(args.key, cfg, query=args.query, delay=args.delay)


if __name__ == "__main__":
    main()
