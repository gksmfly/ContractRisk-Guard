# scripts/crawl_ftc_seed.py
"""
공정거래위원회 심결례 - 불공정약관 시정조치 사례 목록 수집 스크립트 (Seed 데이터)

불공정약관 목록 전체를 순회하며 사건 메타데이터와 PDF 다운로드 식별자를 수집합니다.
검색 키워드에 의존하지 않고, 대표위반유형=불공정약관으로 필터된 전체 페이지를 모읍니다.

사전 설치:
    pip install playwright
    playwright install chromium

사용법:
    python scripts/crawl_ftc_seed.py
    python scripts/crawl_ftc_seed.py --delay 2.0
    python scripts/crawl_ftc_seed.py --max-pages 220
    python scripts/crawl_ftc_seed.py --no-headless
"""

import json
import logging
import os
import re
import time
import argparse
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from dotenv import load_dotenv

load_dotenv()

LOG_DIR = Path(os.environ["LOG_DIR"])
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "crawl_ftc_seed.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

SEED_DIR = Path(os.environ.get("FTC_SEED_DIR", "data/seed"))
BASE_URL = "https://case.ftc.go.kr/ocp/co/ltfr.do"


def save_json(data: Any, filepath: Path) -> None:
    """JSON 데이터를 파일로 저장합니다."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def normalize_text(value: str) -> str:
    """비교를 위해 공백을 정규화합니다."""
    return re.sub(r"\s+", " ", value).strip()


def extract_pdf_info(tr: Any) -> dict[str, str]:
    """행에서 PDF 다운로드용 docId, docSn을 추출합니다."""
    for element in tr.query_selector_all("a, button, input[type='hidden']"):
        onclick = element.get_attribute("onclick") or ""
        values = re.findall(r"['\"]([^'\"]+)['\"]", onclick)
        if "fn_downloadFile" in onclick and len(values) >= 2:
            return {"docId": values[0], "docSn": values[1]}

        data_doc_id = element.get_attribute("data-doc-id") or element.get_attribute("docid") or ""
        data_doc_sn = element.get_attribute("data-doc-sn") or element.get_attribute("docsn") or ""
        if data_doc_id:
            return {"docId": data_doc_id, "docSn": data_doc_sn}

    hidden_values: list[str] = []
    for hidden in tr.query_selector_all("input[type='hidden']"):
        value = hidden.get_attribute("value") or ""
        if value:
            hidden_values.append(value)
    if len(hidden_values) >= 2:
        return {"docId": hidden_values[0], "docSn": hidden_values[1]}

    row_html = tr.inner_html()
    html_match = re.search(
        r"fn_downloadFile\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]",
        row_html,
    )
    if html_match:
        return {"docId": html_match.group(1), "docSn": html_match.group(2)}

    return {}


def find_result_table(page: Any) -> Any | None:
    """검색 결과 테이블을 반환합니다."""
    for selector in [
        "#contents table",
        "table.boardList",
        "table.list",
        "table",
    ]:
        table = page.query_selector(selector)
        if table and table.query_selector("tbody tr"):
            return table
    return None


def get_column_names(table: Any) -> list[str]:
    """결과 테이블의 컬럼명을 추출합니다."""
    th_elements = table.query_selector_all("thead th, tr:first-child th")
    return [normalize_text(th.inner_text()) for th in th_elements]


def is_target_row(cell_data: dict[str, str]) -> bool:
    """불공정약관 사건 행인지 확인합니다."""
    action_type = normalize_text(cell_data.get("대표조치유형", ""))

    if action_type and "불공정약관" not in action_type:
        return False
    return True


def parse_rows(page: Any) -> list[dict[str, Any]]:
    """현재 페이지의 테이블 행을 파싱하여 사건 목록을 반환합니다."""
    rows: list[dict[str, Any]] = []

    table = find_result_table(page)
    if not table:
        tables = page.query_selector_all("table")
        logger.warning(f"테이블 행을 찾지 못함 (페이지 내 table 수: {len(tables)})")
        for idx, t in enumerate(tables):
            cls = t.get_attribute("class") or ""
            tid = t.get_attribute("id") or ""
            row_count = len(t.query_selector_all("tr"))
            logger.warning(f"  table[{idx}] class='{cls}' id='{tid}' rows={row_count}")
        return rows

    tr_elements = table.query_selector_all("tbody tr")
    logger.info(f"테이블 행 수: {len(tr_elements)}")

    col_names = get_column_names(table)
    if col_names:
        logger.info(f"컬럼명: {col_names}")

    for tr in tr_elements:
        tds = tr.query_selector_all("td")
        if len(tds) < 2:
            continue

        first_text = tds[0].inner_text().strip()
        if "없습니다" in first_text or "데이터" in first_text:
            continue

        # 링크(사건명) 추출
        link = tr.query_selector("a")
        title = normalize_text(link.inner_text()) if link else ""

        # 각 td 텍스트
        td_texts = [normalize_text(td.inner_text()) for td in tds]

        # 컬럼명과 매핑
        cell_data: dict[str, str] = {}
        if col_names:
            for i, val in enumerate(td_texts):
                key = col_names[i] if i < len(col_names) else f"col_{i}"
                cell_data[key] = val
        else:
            cell_data = {f"col_{i}": v for i, v in enumerate(td_texts)}

        if title:
            cell_data["사건명"] = title
        if not is_target_row(cell_data):
            continue

        # PDF 다운로드 정보 추출 (docId, docSn)
        pdf_info = extract_pdf_info(tr)
        if not pdf_info:
            logger.warning(
                "PDF 정보 추출 실패: 사건명='%s', 사건번호='%s', 의결번호='%s'",
                title,
                cell_data.get("사건번호", ""),
                cell_data.get("의결번호", ""),
            )

        row: dict[str, Any] = {
            "사건명": title,
            "셀_데이터": cell_data,
            "pdf_info": pdf_info,
        }
        rows.append(row)

    return rows


def build_list_url(page_index: int) -> str:
    """불공정약관 목록 페이지 URL을 생성합니다."""
    query = urlencode(
        {
            "pageIndex": page_index,
            "caseNo": "",
            "caseNm": "",
            "decsnNo": "",
            "startRceptDt": "",
            "endRceptDt": "",
            "reprsntManagtTyCd": "",
            "reprsntViolTy": "10",
            "searchKrwd": "",
        }
    )
    return f"{BASE_URL}?{query}"


def build_case_record(row: dict[str, Any], page_index: int) -> dict[str, Any]:
    """수집 행을 저장용 사건 레코드로 변환합니다."""
    return {
        "사건명": row["사건명"],
        "셀_데이터": row["셀_데이터"],
        "pdf_info": row["pdf_info"],
        "출처": "공정거래위원회 심결례",
        "카테고리": "불공정약관",
        "수집_페이지": page_index,
    }


def crawl_all_pages(page: Any, delay: float = 1.5, max_pages: int = 250) -> list[dict[str, Any]]:
    """불공정약관 전체 페이지를 순회하며 사건 목록을 수집합니다."""
    logger.info("전체(불공정약관) 크롤링 시작")

    all_cases: list[dict[str, Any]] = []
    previous_signature = ""

    for page_index in range(1, max_pages + 1):
        url = build_list_url(page_index)
        logger.info("전체(불공정약관) - 페이지 %s 크롤링 시작: %s", page_index, url)

        page.goto(url, wait_until="networkidle", timeout=30000)
        time.sleep(delay)

        rows = parse_rows(page)
        if not rows:
            logger.info("전체(불공정약관) - 페이지 %s 데이터 없음, 수집 종료", page_index)
            break

        page_signature = " || ".join(
            [
                rows[0]["사건명"],
                rows[-1]["사건명"],
                str(len(rows)),
            ]
        )
        if page_signature == previous_signature:
            logger.warning(
                "전체(불공정약관) - 페이지 %s가 직전 페이지와 동일하여 수집을 중단합니다.",
                page_index,
            )
            break
        previous_signature = page_signature

        for row in rows:
            all_cases.append(build_case_record(row, page_index))

        logger.info(
            "전체(불공정약관) - 페이지 %s 완료: %s건 (누적: %s건)",
            page_index,
            len(rows),
            len(all_cases),
        )

        if len(rows) < 10:
            logger.info(
                "전체(불공정약관) - 페이지 %s가 마지막 페이지로 판단되어 수집을 종료합니다.",
                page_index,
            )
            break

    logger.info("전체(불공정약관) 크롤링 완료 - 총 %s건 수집", len(all_cases))
    return all_cases


def deduplicate(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """사건번호/의결번호/사건명을 조합한 키 기준으로 중복을 제거합니다."""
    seen: dict[str, dict[str, Any]] = {}
    for case in cases:
        title = case.get("사건명", "")
        cell_data = case.get("셀_데이터", {})
        case_number = normalize_text(cell_data.get("사건번호", ""))
        decision_number = normalize_text(cell_data.get("의결번호", ""))
        dedup_key = " | ".join(filter(None, [case_number, decision_number, title]))

        if not dedup_key:
            continue
        if dedup_key not in seen:
            seen[dedup_key] = case
        else:
            # 키워드 병합
            existing_kw = seen[dedup_key].get("검색_키워드", "")
            new_kw = case.get("검색_키워드", "")
            merged_kw = ", ".join(sorted(filter(None, {existing_kw, new_kw})))
            seen[dedup_key]["검색_키워드"] = merged_kw
    return list(seen.values())


def crawl_all(delay: float = 1.5, headless: bool = True, max_pages: int = 250) -> None:
    """불공정약관 전체 페이지를 크롤링 후 중복 제거하여 저장합니다."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error("Playwright가 설치되어 있지 않습니다. 설치 후 재시도하세요:")
        logger.error("  pip install playwright && playwright install chromium")
        return

    logger.info("=== 공정위 불공정약관 시정조치 사례 수집 시작 ===")
    all_cases: list[dict[str, Any]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="ko-KR",
        )
        page = context.new_page()

        try:
            cases = crawl_all_pages(page, delay=delay, max_pages=max_pages)
            all_cases.extend(cases)
            logger.info(f"전체 크롤링 완료: {len(cases)}건")
        except Exception as e:
            logger.error(f"전체 크롤링 실패: {e}")

        browser.close()

    # 3단계: 중복 제거
    before_count = len(all_cases)
    unique_cases = deduplicate(all_cases)
    after_count = len(unique_cases)
    logger.info(f"중복 제거: {before_count}건 → {after_count}건 (중복 {before_count - after_count}건 제거)")

    # 4단계: 저장
    output_path = SEED_DIR / "ftc_cases_raw.json"
    result = {
        "총_건수": after_count,
        "수집_카테고리": "불공정약관(represntViolTy=10)",
        "수집_방식": "불공정약관 전체 페이지 순회",
        "사례": unique_cases,
    }
    save_json(result, output_path)
    logger.info(f"=== 저장 완료: {output_path} ({after_count}건) ===")


def main() -> None:
    """메인 실행 함수입니다."""
    parser = argparse.ArgumentParser(
        description="공정거래위원회 심결례 - 불공정약관 시정조치 사례 목록 수집"
    )
    parser.add_argument(
        "--delay", type=float, default=1.5, help="페이지 요청 간격(초, 기본값: 1.5)"
    )
    parser.add_argument(
        "--no-headless", action="store_true", help="브라우저 화면 표시 (디버깅용)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=250,
        help="최대 순회 페이지 수(기본값: 250)",
    )
    args = parser.parse_args()

    SEED_DIR.mkdir(parents=True, exist_ok=True)
    crawl_all(delay=args.delay, headless=not args.no_headless, max_pages=args.max_pages)


if __name__ == "__main__":
    main()
