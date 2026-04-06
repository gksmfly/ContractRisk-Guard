# scripts/crawl_ftc_seed.py
"""
공정거래위원회 심결례 - 불공정약관 시정조치 사례 목록 수집 스크립트 (Seed 데이터)

1단계: 목록 크롤링 → 사건번호, 의결번호, 사건명, 의결일, PDF URL 수집
상세 페이지 클릭 없이 목록 테이블 + PDF 링크만 빠르게 수집합니다.

사전 설치:
    pip install playwright
    playwright install chromium

사용법:
    python scripts/crawl_ftc_seed.py
    python scripts/crawl_ftc_seed.py --delay 2.0
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

KEYWORDS: list[str] = ["면책", "손해배상", "해지", "위약금", "책임"]


def save_json(data: Any, filepath: Path) -> None:
    """JSON 데이터를 파일로 저장합니다."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_total_count(page: Any) -> int:
    """검색 결과 총 건수를 추출합니다."""
    body_text = page.inner_text("body")
    for pattern in [r"총\s*([\d,]+)\s*건", r"전체\s*([\d,]+)\s*건", r"\(\s*([\d,]+)\s*건\s*\)"]:
        match = re.search(pattern, body_text)
        if match:
            return int(match.group(1).replace(",", ""))
    return 0


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


def is_target_row(cell_data: dict[str, str], keyword: str | None) -> bool:
    """불공정약관 조건과 키워드 조건을 만족하는 행인지 확인합니다."""
    action_type = normalize_text(cell_data.get("대표조치유형", ""))
    title = normalize_text(cell_data.get("사건명", ""))

    if action_type and "불공정약관" not in action_type:
        return False
    if keyword and keyword not in title:
        return False
    return True


def parse_rows(page: Any, keyword: str | None = None) -> list[dict[str, Any]]:
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
        if not is_target_row(cell_data, keyword):
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


def go_next_page(page: Any, current_page: int) -> bool:
    """현재 페이지에서 다음 페이지로 이동합니다."""
    next_num = current_page + 1

    # 1) 다음 페이지 번호 링크가 보이면 클릭
    next_link = page.query_selector(
        f".paging a:text-is('{next_num}'), "
        f"a[onclick*='{next_num}']"
    )
    if next_link:
        try:
            next_link.click()
            page.wait_for_load_state("networkidle", timeout=15000)
            time.sleep(1)
            return True
        except Exception:
            pass

    # 2) 다음 그룹 버튼 클릭 (10페이지 단위 그룹 넘김)
    for selector in [
        "a.next", ".paging .next", "a[title='다음']",
        "a:text-is('다음')", "a:text-is('>')",
        ".pagination .next a", "a.page-next",
    ]:
        next_btn = page.query_selector(selector)
        if next_btn:
            try:
                next_btn.click()
                page.wait_for_load_state("networkidle", timeout=15000)
                time.sleep(1)
                return True
            except Exception:
                continue

    return False


def crawl_category(
    page: Any,
    keyword: str | None = None,
    delay: float = 1.5,
) -> list[dict[str, Any]]:
    """불공정약관 카테고리(+ 키워드 검색) 전체 페이지를 크롤링합니다."""
    url = f"{BASE_URL}?represntViolTy=10"
    if keyword:
        url += f"&searchKeyword={keyword}"

    label = f"키워드=[{keyword}]" if keyword else "전체(불공정약관)"
    logger.info(f"{label} 크롤링 시작 - URL: {url}")

    page.goto(url, wait_until="networkidle", timeout=30000)
    time.sleep(delay)

    total = get_total_count(page)
    logger.info(f"{label} 총 {total}건 발견")
    if total == 0:
        logger.warning(f"{label} 검색 결과 총 건수가 0건으로 확인되어 수집을 중단합니다.")
        return []

    all_cases: list[dict[str, Any]] = []
    page_num = 1

    while True:
        logger.info(f"{label} - 페이지 {page_num} 크롤링 중")

        rows = parse_rows(page, keyword=keyword)
        if not rows:
            logger.info(f"{label} - 페이지 {page_num}에 데이터 없음, 크롤링 종료")
            break

        for row in rows:
            case: dict[str, Any] = {
                "사건명": row["사건명"],
                "셀_데이터": row["셀_데이터"],
                "pdf_info": row["pdf_info"],
                "검색_키워드": keyword or "",
                "출처": "공정거래위원회 심결례",
                "카테고리": "불공정약관",
            }
            all_cases.append(case)

        logger.info(f"{label} - 페이지 {page_num} 완료: {len(rows)}건 (누적: {len(all_cases)}건)")

        # 다음 페이지로 이동 (순차 진행)
        if not go_next_page(page, page_num):
            logger.info(f"{label} - 마지막 페이지: {page_num}")
            break
        page_num += 1

    logger.info(f"{label} 크롤링 완료 - 총 {len(all_cases)}건 수집")
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


def crawl_all(delay: float = 1.5, headless: bool = True) -> None:
    """불공정약관 전체 + 키워드별 크롤링 후 중복 제거하여 저장합니다."""
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

        # 1단계: 불공정약관 전체 크롤링
        try:
            cases = crawl_category(page, keyword=None, delay=delay)
            all_cases.extend(cases)
            logger.info(f"전체 크롤링 완료: {len(cases)}건")
        except Exception as e:
            logger.error(f"전체 크롤링 실패: {e}")

        # 2단계: 키워드별 추가 검색
        for kw in KEYWORDS:
            try:
                cases = crawl_category(page, keyword=kw, delay=delay)
                all_cases.extend(cases)
                logger.info(f"키워드 [{kw}] 크롤링 완료: {len(cases)}건")
            except Exception as e:
                logger.error(f"키워드 [{kw}] 크롤링 실패: {e}")

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
        "검색_키워드": KEYWORDS,
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
    args = parser.parse_args()

    SEED_DIR.mkdir(parents=True, exist_ok=True)
    crawl_all(delay=args.delay, headless=not args.no_headless)


if __name__ == "__main__":
    main()
