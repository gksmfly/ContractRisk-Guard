# backend/scripts/crawl_ftc_cases.py
"""
공정거래위원회 심결례 - 불공정약관 시정조치 사례 목록 수집 스크립트 (Seed 데이터)

불공정약관 심결례 전체를 순회하며 사건 메타데이터와 PDF 다운로드 식별자를 수집합니다.
사건번호에 "약관"이 포함된 케이스만 필터링합니다 (예: 2022약관0758).
reprsntViolTy=10 은 실제로 필터링 되지 않아 caseNo=약관 으로 대체합니다.

사용법:
    python backend/scripts/crawl_ftc_cases.py

샘플 출력 데이터 (data/raw/ftc_cases/ftc_cases_raw.json):
    {
        "총_건수": 350,
        "수집_카테고리": "불공정약관(caseNo=약관)",
        "수집_방식": "사건번호 약관 포함 전체 페이지 순회",
        "사례": [
            {
                "사건명": "㈜OO 불공정약관 시정",
                "셀_데이터": {
                    "사건번호": "2024약관1234",
                    "의결번호": "제2024-XXX호",
                    "사건명": "㈜OO 불공정약관 시정"
                },
                "pdf_info": {"docId": "AB1234", "docSn": "1"},
                "출처": "공정거래위원회 심결례",
                "카테고리": "불공정약관",
                "수집_페이지": 1
            }
        ]
    }
"""

import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import save_json, setup_logger, PROJECT_ROOT

logger = setup_logger("crawl_ftc_cases.log")

FTC_DIR = Path(os.environ.get("FTC_DIR", str(PROJECT_ROOT / "data/raw/ftc_cases")))
BASE_URL = os.environ.get("FTC_CASE_URL", "https://case.ftc.go.kr/ocp/co/ltfr.do")


"""비교를 위해 공백을 정규화합니다.

Args:
    value (str): 정규화할 문자열.

Returns:
    str: 연속된 공백을 단일 공백으로 치환하고 앞뒤 공백을 제거한 문자열.
"""

def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


"""행에서 PDF 다운로드용 docId, docSn을 추출합니다.

onclick 속성, data-* 속성, hidden input, HTML 소스 순으로 시도합니다.

Args:
    tr (Any): Playwright로 조회한 테이블 행(tr) 엘리먼트.

Returns:
    dict[str, str]: PDF 식별자 딕셔너리.
                    추출 성공 시 {"docId": str, "docSn": str}, 실패 시 빈 딕셔너리 {}.
"""

def extract_pdf_info(tr: Any) -> dict[str, str]:
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


"""검색 결과 테이블을 반환합니다.

여러 CSS 선택자를 순서대로 시도합니다.

Args:
    page (Any): Playwright 페이지 객체.

Returns:
    Any | None: tbody tr이 존재하는 첫 번째 테이블 엘리먼트. 없으면 None.
"""

def find_result_table(page: Any) -> Any | None:
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


"""결과 테이블의 컬럼명을 추출합니다.

Args:
    table (Any): Playwright로 조회한 테이블 엘리먼트.

Returns:
    list[str]: 정규화된 컬럼명 문자열 리스트.
               예: ["번호", "사건명", "의결번호", "등록일"]
"""

def get_column_names(table: Any) -> list[str]:
    th_elements = table.query_selector_all("thead th, tr:first-child th")
    return [normalize_text(th.inner_text()) for th in th_elements]


"""현재 페이지의 테이블 행을 파싱하여 사건 목록을 반환합니다.

Args:
    page (Any): Playwright 페이지 객체.

Returns:
    list[dict[str, Any]]: 사건 정보 딕셔너리 리스트.
                          각 항목 구조: {"사건명": str, "셀_데이터": dict, "pdf_info": dict}
"""

def parse_rows(page: Any) -> list[dict[str, Any]]:
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

        link = tr.query_selector("a")
        title = normalize_text(link.inner_text()) if link else ""

        td_texts = [normalize_text(td.inner_text()) for td in tds]

        cell_data: dict[str, str] = {}
        if col_names:
            for i, val in enumerate(td_texts):
                key = col_names[i] if i < len(col_names) else f"col_{i}"
                cell_data[key] = val
        else:
            cell_data = {f"col_{i}": v for i, v in enumerate(td_texts)}

        if title:
            cell_data["사건명"] = title

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

    pdf_count = sum(1 for row in rows if row["pdf_info"].get("docId"))
    logger.info("현재 페이지 유효 행 수: %s, PDF 식별자 추출 성공: %s", len(rows), pdf_count)
    return rows


"""불공정약관 목록 페이지 URL을 생성합니다.

Args:
    page_index (int): 조회할 페이지 번호 (1부터 시작).

Returns:
    str: caseNo=약관 필터가 적용된 목록 페이지 URL 문자열.
"""

def build_list_url(page_index: int) -> str:
    query = urlencode(
        {
            "pageIndex": page_index,
            "caseNo": "약관",
            "caseNm": "",
            "decsnNo": "",
            "startRceptDt": "",
            "endRceptDt": "",
            "reprsntManagtTyCd": "",
            "reprsntViolTy": "",
            "searchKrwd": "",
        }
    )
    return f"{BASE_URL}?{query}"


"""수집 행을 저장용 사건 레코드로 변환합니다.

Args:
    row (dict[str, Any]): parse_rows()가 반환한 단일 행 딕셔너리.
                          키: "사건명", "셀_데이터", "pdf_info"
    page_index (int): 해당 행이 수집된 페이지 번호.

Returns:
    dict[str, Any]: 저장용 사건 레코드 딕셔너리.
                    키: "사건명", "셀_데이터", "pdf_info", "출처", "카테고리", "수집_페이지"
"""

def build_case_record(row: dict[str, Any], page_index: int) -> dict[str, Any]:
    return {
        "사건명": row["사건명"],
        "셀_데이터": row["셀_데이터"],
        "pdf_info": row["pdf_info"],
        "출처": "공정거래위원회 심결례",
        "카테고리": "불공정약관",
        "수집_페이지": page_index,
    }


"""불공정약관 전체 페이지를 순회하며 사건 목록을 수집합니다.

연속된 두 페이지의 내용이 동일하면 수집을 중단합니다.

Args:
    page (Any): Playwright 페이지 객체.
    delay (float): 페이지 요청 간 대기 시간(초).
    max_pages (int): 순회할 최대 페이지 수.

Returns:
    list[dict[str, Any]]: 수집된 사건 레코드 딕셔너리 리스트.
"""

def crawl_all_pages(page: Any, delay: float = 1.5, max_pages: int = 250) -> list[dict[str, Any]]:
    logger.info("약관심결례 크롤링 시작")

    all_cases: list[dict[str, Any]] = []
    previous_signature = ""

    for page_index in range(1, max_pages + 1):
        url = build_list_url(page_index)
        logger.info("약관심결례 - 페이지 %s 크롤링 시작: %s", page_index, url)

        page.goto(url, wait_until="networkidle", timeout=30000)
        time.sleep(delay)

        rows = parse_rows(page)
        if not rows:
            logger.info("약관심결례 - 페이지 %s 데이터 없음, 수집 종료", page_index)
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
                "약관심결례 - 페이지 %s가 직전 페이지와 동일하여 수집을 중단합니다.",
                page_index,
            )
            break
        previous_signature = page_signature

        for row in rows:
            all_cases.append(build_case_record(row, page_index))

        logger.info(
            "약관심결례 - 페이지 %s 완료: %s건 (누적: %s건)",
            page_index,
            len(rows),
            len(all_cases),
        )

    logger.info("약관심결례 크롤링 완료 - 총 %s건 수집", len(all_cases))
    return all_cases


"""사건번호/의결번호/사건명을 조합한 키 기준으로 중복을 제거합니다.

동일 키가 여러 페이지에 존재하면 가장 앞 페이지 항목을 유지합니다.

Args:
    cases (list[dict[str, Any]]): 수집된 사건 레코드 리스트 (build_case_record 반환값들).

Returns:
    list[dict[str, Any]]: 중복이 제거된 사건 레코드 리스트.
"""

def deduplicate(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            existing_page = int(seen[dedup_key].get("수집_페이지", 10**9))
            new_page = int(case.get("수집_페이지", 10**9))
            if new_page < existing_page:
                seen[dedup_key] = case
    return list(seen.values())


"""불공정약관 전체 페이지를 크롤링 후 중복 제거하여 저장합니다.

Args:
    delay (float): 페이지 요청 간 대기 시간(초).
    headless (bool): True이면 브라우저를 백그라운드에서 실행.
    max_pages (int): 순회할 최대 페이지 수.
"""

def crawl_all(delay: float = 1.5, headless: bool = True, max_pages: int = 250) -> None:
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
        except Exception as e:
            logger.error(f"전체 크롤링 실패: {e}")

        browser.close()

    before_count = len(all_cases)
    unique_cases = deduplicate(all_cases)
    after_count = len(unique_cases)
    logger.info(f"중복 제거: {before_count}건 → {after_count}건 (중복 {before_count - after_count}건 제거)")

    FTC_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FTC_DIR / "ftc_cases_raw.json"
    result = {
        "총_건수": after_count,
        "수집_카테고리": "불공정약관(caseNo=약관)",
        "수집_방식": "사건번호 약관 포함 전체 페이지 순회",
        "사례": unique_cases,
    }
    save_json(result, output_path)
    logger.info(f"=== 저장 완료: {output_path} ({after_count}건) ===")


def main() -> None:
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

    FTC_DIR.mkdir(parents=True, exist_ok=True)
    crawl_all(delay=args.delay, headless=not args.no_headless, max_pages=args.max_pages)


if __name__ == "__main__":
    main()
