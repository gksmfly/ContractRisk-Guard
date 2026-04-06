# scripts/crawl_ftc_seed.py
"""
공정거래위원회 심결례 - 불공정약관 시정조치 사례 수집 스크립트 (Seed 데이터)

공정위 심결례 사이트(case.ftc.go.kr)에서 불공정약관 사례를 크롤링합니다.
JS 렌더링이 필요하므로 Playwright를 사용합니다.

사전 설치:
    pip install playwright
    playwright install chromium

사용법:
    python scripts/crawl_ftc_seed.py
    python scripts/crawl_ftc_seed.py --delay 2.0
    python scripts/crawl_ftc_seed.py --no-headless   # 브라우저 화면 표시 (디버깅용)
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

    # "총 N건", "전체 N건", "(N건)" 등 다양한 패턴 시도
    for pattern in [r"총\s*([\d,]+)\s*건", r"전체\s*([\d,]+)\s*건", r"\(\s*([\d,]+)\s*건\s*\)"]:
        match = re.search(pattern, body_text)
        if match:
            return int(match.group(1).replace(",", ""))

    return 0


def parse_rows(page: Any) -> list[dict[str, Any]]:
    """현재 페이지의 테이블 행을 파싱하여 사건 목록을 반환합니다."""
    rows: list[dict[str, Any]] = []

    # 여러 테이블 선택자 시도
    tr_elements = []
    for selector in [
        "table.boardList tbody tr",
        "#contents table tbody tr",
        "table.list tbody tr",
        "table tbody tr",
    ]:
        tr_elements = page.query_selector_all(selector)
        if tr_elements:
            logger.info(f"테이블 선택자 매칭: '{selector}' → {len(tr_elements)}행")
            break

    if not tr_elements:
        # 디버깅: 페이지에 있는 테이블 확인
        tables = page.query_selector_all("table")
        logger.warning(f"테이블 행을 찾지 못함 (페이지 내 table 수: {len(tables)})")
        for idx, t in enumerate(tables):
            cls = t.get_attribute("class") or ""
            tid = t.get_attribute("id") or ""
            row_count = len(t.query_selector_all("tr"))
            logger.warning(f"  table[{idx}] class='{cls}' id='{tid}' rows={row_count}")
        return rows

    for idx, tr in enumerate(tr_elements):
        tds = tr.query_selector_all("td")
        if len(tds) < 2:
            continue

        # "데이터가 없습니다" 류의 안내 행 건너뛰기
        if len(tds) == 1:
            continue
        first_text = tds[0].inner_text().strip()
        if "없습니다" in first_text or "데이터" in first_text:
            continue

        # 링크 추출
        link = tr.query_selector("a")
        title = link.inner_text().strip() if link else ""

        # 각 td에서 텍스트 추출
        td_texts = [td.inner_text().strip() for td in tds]

        row: dict[str, Any] = {
            "제목": title,
            "셀_텍스트": td_texts,
            "행_인덱스": idx,
        }
        rows.append(row)

    return rows


def fetch_detail_by_click(page: Any, row_idx: int, delay: float) -> dict[str, str]:
    """목록에서 링크를 직접 클릭하여 상세 페이지 정보를 추출합니다."""
    detail: dict[str, str] = {}
    try:
        # 현재 목록의 행에서 링크를 다시 찾아 클릭
        tr_elements = page.query_selector_all("table tbody tr")
        target_tr = None
        td_row_idx = 0
        for tr in tr_elements:
            tds = tr.query_selector_all("td")
            if len(tds) < 2:
                continue
            if td_row_idx == row_idx:
                target_tr = tr
                break
            td_row_idx += 1

        if not target_tr:
            logger.warning(f"행 인덱스 {row_idx}에 해당하는 행을 찾지 못함")
            return detail

        link = target_tr.query_selector("a")
        if not link:
            logger.warning(f"행 인덱스 {row_idx}에 링크가 없음")
            return detail

        # 클릭 후 네비게이션 대기
        link.click()
        page.wait_for_load_state("networkidle", timeout=20000)
        time.sleep(delay)

        # 상세 정보 테이블에서 항목 추출 (여러 선택자 시도)
        detail_rows = []
        for selector in [
            "table.boardView tr",
            "table.viewTbl tr",
            ".view_table tr",
            "table.tbl_view tr",
            "#contents table tr",
        ]:
            detail_rows = page.query_selector_all(selector)
            if detail_rows:
                break

        for dr in detail_rows:
            th = dr.query_selector("th")
            td = dr.query_selector("td")
            if th and td:
                key = th.inner_text().strip()
                value = td.inner_text().strip()
                if key and value:
                    detail[key] = value

        # 본문 영역 추출 (여러 선택자 시도)
        for selector in [
            ".view_cont",
            ".boardViewCont",
            ".detailContent",
            ".cont_area",
            "#contents .cont",
            ".board_view",
        ]:
            body_el = page.query_selector(selector)
            if body_el:
                text = body_el.inner_text().strip()
                if len(text) > 50:
                    detail["본문"] = text
                    break

        # 본문을 못 찾았으면 전체 컨텐츠에서 추출
        if not detail.get("본문"):
            content_el = page.query_selector("#contents")
            if content_el:
                detail["본문_전체"] = content_el.inner_text().strip()

    except Exception as e:
        logger.warning(f"상세 페이지 파싱 실패: {e}")

    return detail


def navigate_to_page(page: Any, page_num: int) -> bool:
    """목록에서 특정 페이지 번호로 이동합니다."""
    if page_num <= 1:
        return True

    try:
        # 페이지 번호 링크 클릭 시도
        paging_link = page.query_selector(
            f"a[onclick*='goPage({page_num})'], "
            f"a[onclick*=\"goPage('{page_num}')\"], "
            f".paging a:text-is('{page_num}')"
        )
        if paging_link:
            paging_link.click()
            page.wait_for_load_state("networkidle", timeout=15000)
            time.sleep(1)
            return True
    except Exception:
        pass

    try:
        # goPage 함수 직접 호출 시도
        page.evaluate(f"goPage({page_num})")
        page.wait_for_load_state("networkidle", timeout=15000)
        time.sleep(1)
        return True
    except Exception as e:
        logger.warning(f"페이지 {page_num} 이동 실패: {e}")
        return False


def has_next_page(page: Any, current_page: int) -> bool:
    """다음 페이지가 존재하는지 확인합니다."""
    next_num = current_page + 1

    # 다음 페이지 번호 링크 확인
    next_link = page.query_selector(
        f"a[onclick*='goPage({next_num})'], "
        f"a[onclick*=\"goPage('{next_num}')\"], "
        f".paging a:text-is('{next_num}')"
    )
    if next_link:
        return True

    # '다음' 버튼 확인
    next_btn = page.query_selector(
        "a.next, .paging .next, a[title='다음'], "
        "a:text-is('다음'), a:text-is('>')"
    )
    return next_btn is not None


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

    all_cases: list[dict[str, Any]] = []
    page_num = 1

    while True:
        logger.info(f"{label} - 페이지 {page_num} 크롤링 중")

        # 페이지 이동 (1페이지는 이미 로드됨)
        if page_num > 1:
            # 목록 페이지를 다시 로드하고 해당 페이지로 이동
            page.goto(url, wait_until="networkidle", timeout=30000)
            time.sleep(delay)
            if not navigate_to_page(page, page_num):
                break

        rows = parse_rows(page)
        if not rows:
            logger.info(f"{label} - 페이지 {page_num}에 데이터 없음, 크롤링 종료")
            break

        for i, row in enumerate(rows):
            case: dict[str, Any] = {
                "제목": row["제목"],
                "셀_텍스트": row["셀_텍스트"],
                "검색_키워드": keyword or "",
                "출처": "공정거래위원회 심결례",
                "카테고리": "불공정약관",
            }

            # 상세 페이지 크롤링: 클릭 → 파싱 → 뒤로가기
            detail = fetch_detail_by_click(page, i, delay)
            if detail:
                case["상세정보"] = detail

            # 목록으로 복귀
            try:
                page.go_back(wait_until="networkidle", timeout=20000)
                time.sleep(delay)
            except Exception:
                # go_back 실패 시 URL로 직접 재이동
                page.goto(url, wait_until="networkidle", timeout=30000)
                time.sleep(delay)
                if page_num > 1:
                    navigate_to_page(page, page_num)

            all_cases.append(case)

        logger.info(f"{label} - 페이지 {page_num} 완료: {len(rows)}건 (누적: {len(all_cases)}건)")

        # 다음 페이지 확인 (목록 페이지 다시 로드하여 확인)
        if not has_next_page(page, page_num):
            break

        page_num += 1

    logger.info(f"{label} 크롤링 완료 - 총 {len(all_cases)}건 수집")
    return all_cases


def deduplicate(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """제목 기준으로 중복을 제거합니다. 상세정보가 있는 건을 우선 유지합니다."""
    seen: dict[str, dict[str, Any]] = {}
    for case in cases:
        title = case.get("제목", "")
        if not title:
            continue

        if title not in seen:
            seen[title] = case
        else:
            existing = seen[title]
            existing_detail_len = len(json.dumps(existing.get("상세정보", {}), ensure_ascii=False))
            new_detail_len = len(json.dumps(case.get("상세정보", {}), ensure_ascii=False))
            # 키워드 병합
            existing_kw = existing.get("검색_키워드", "")
            new_kw = case.get("검색_키워드", "")
            merged_kw = ", ".join(sorted(filter(None, {existing_kw, new_kw})))
            if new_detail_len > existing_detail_len:
                case["검색_키워드"] = merged_kw
                seen[title] = case
            else:
                seen[title]["검색_키워드"] = merged_kw

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
        description="공정거래위원회 심결례 - 불공정약관 시정조치 사례 수집"
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
