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
"""

import json
import logging
import os
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


def parse_rows(page: Any) -> list[dict[str, str]]:
    """현재 페이지의 테이블 행을 파싱하여 사건 목록을 반환합니다."""
    rows: list[dict[str, str]] = []

    # 테이블 행 선택 (목록 테이블의 tbody tr)
    tr_elements = page.query_selector_all("table.boardList tbody tr")
    if not tr_elements:
        # 대체 선택자 시도
        tr_elements = page.query_selector_all("#contents table tbody tr")
    if not tr_elements:
        tr_elements = page.query_selector_all("table tbody tr")

    for tr in tr_elements:
        tds = tr.query_selector_all("td")
        if len(tds) < 3:
            continue

        # 링크에서 상세 페이지 정보 추출
        link = tr.query_selector("a")
        onclick = link.get_attribute("onclick") if link else ""
        title = link.inner_text().strip() if link else ""

        # 각 td에서 텍스트 추출
        td_texts = [td.inner_text().strip() for td in tds]

        row: dict[str, str] = {
            "제목": title,
            "onclick": onclick or "",
            "셀_텍스트": td_texts,
        }
        rows.append(row)

    return rows


def fetch_detail(page: Any, onclick: str, delay: float) -> dict[str, str]:
    """심결례 상세 페이지에서 본문 정보를 추출합니다."""
    detail: dict[str, str] = {}
    try:
        # onclick 함수 실행으로 상세 페이지 이동
        page.evaluate(onclick.rstrip(";"))
        page.wait_for_load_state("networkidle", timeout=15000)
        time.sleep(delay)

        # 상세 정보 테이블에서 항목 추출
        detail_rows = page.query_selector_all("table.boardView tr, table.viewTbl tr, .view_table tr")
        for dr in detail_rows:
            th = dr.query_selector("th")
            td = dr.query_selector("td")
            if th and td:
                key = th.inner_text().strip()
                value = td.inner_text().strip()
                if key:
                    detail[key] = value

        # 본문 영역 추출 (여러 선택자 시도)
        for selector in [".view_cont", ".boardViewCont", "#contents .cont", ".detailContent"]:
            body_el = page.query_selector(selector)
            if body_el:
                detail["본문"] = body_el.inner_text().strip()
                break

        if not detail.get("본문"):
            # 본문이 별도 영역에 없으면 전체 컨텐츠에서 추출
            content_el = page.query_selector("#contents")
            if content_el:
                detail["본문_전체"] = content_el.inner_text().strip()

    except Exception as e:
        logger.warning(f"상세 페이지 파싱 실패: {e}")

    return detail


def crawl_list_page(page: Any, page_num: int) -> list[dict[str, str]]:
    """목록에서 특정 페이지로 이동하여 데이터를 파싱합니다."""
    if page_num > 1:
        try:
            # 페이지네이션 클릭
            page.evaluate(f"goPage({page_num})")
            page.wait_for_load_state("networkidle", timeout=15000)
            time.sleep(1)
        except Exception:
            try:
                # 대체: 페이지 링크 직접 클릭
                paging_link = page.query_selector(f"a[onclick*='goPage({page_num})']")
                if paging_link:
                    paging_link.click()
                    page.wait_for_load_state("networkidle", timeout=15000)
                    time.sleep(1)
            except Exception as e:
                logger.warning(f"페이지 {page_num} 이동 실패: {e}")
                return []

    return parse_rows(page)


def get_total_count(page: Any) -> int:
    """검색 결과 총 건수를 추출합니다."""
    # 총 건수 텍스트에서 숫자 추출 (예: "총 123건")
    for selector in [".total_count", ".result_count", ".board_count", ".pageInfo"]:
        el = page.query_selector(selector)
        if el:
            text = el.inner_text()
            import re
            nums = re.findall(r"[\d,]+", text)
            if nums:
                return int(nums[0].replace(",", ""))

    # 대체: 페이지 전체 텍스트에서 "총 N건" 패턴 탐색
    import re
    body_text = page.query_selector("body").inner_text()
    match = re.search(r"총\s*([\d,]+)\s*건", body_text)
    if match:
        return int(match.group(1).replace(",", ""))

    return 0


def crawl_category(
    page: Any,
    keyword: str | None = None,
    delay: float = 1.5,
) -> list[dict[str, Any]]:
    """불공정약관 카테고리(+ 키워드 검색) 전체 페이지를 크롤링합니다."""
    # 기본 URL에 불공정약관 유형 파라미터 추가
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
        rows = crawl_list_page(page, page_num)

        if not rows:
            logger.info(f"{label} - 페이지 {page_num}에 데이터 없음, 크롤링 종료")
            break

        for row in rows:
            case: dict[str, Any] = {
                "제목": row["제목"],
                "셀_텍스트": row["셀_텍스트"],
                "검색_키워드": keyword or "",
                "출처": "공정거래위원회 심결례",
                "카테고리": "불공정약관",
            }

            # 상세 페이지 크롤링
            if row.get("onclick"):
                # 현재 URL 저장 (목록으로 복귀용)
                list_url = page.url
                detail = fetch_detail(page, row["onclick"], delay)
                if detail:
                    case["상세정보"] = detail

                # 목록 페이지로 복귀
                page.goto(list_url, wait_until="networkidle", timeout=30000)
                time.sleep(delay)

                # 원래 페이지로 다시 이동
                if page_num > 1:
                    try:
                        page.evaluate(f"goPage({page_num})")
                        page.wait_for_load_state("networkidle", timeout=15000)
                        time.sleep(1)
                    except Exception:
                        pass

            all_cases.append(case)

        logger.info(f"{label} - 페이지 {page_num} 완료: {len(rows)}건 (누적: {len(all_cases)}건)")

        # 다음 페이지 존재 여부 확인
        next_link = page.query_selector(f"a[onclick*='goPage({page_num + 1})']")
        if not next_link:
            # 대체: '다음' 버튼 확인
            next_btn = page.query_selector("a.next, .paging .next, a[title='다음']")
            if not next_btn:
                break

        page_num += 1
        time.sleep(delay)

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
            # 상세정보가 더 풍부한 쪽을 유지
            existing = seen[title]
            existing_detail_len = len(json.dumps(existing.get("상세정보", {}), ensure_ascii=False))
            new_detail_len = len(json.dumps(case.get("상세정보", {}), ensure_ascii=False))
            if new_detail_len > existing_detail_len:
                # 키워드 목록 병합
                existing_kw = existing.get("검색_키워드", "")
                new_kw = case.get("검색_키워드", "")
                merged_kw = ", ".join(filter(None, set(f"{existing_kw}, {new_kw}".split(", "))))
                case["검색_키워드"] = merged_kw
                seen[title] = case
            else:
                # 키워드만 병합
                existing_kw = existing.get("검색_키워드", "")
                new_kw = case.get("검색_키워드", "")
                merged_kw = ", ".join(filter(None, set(f"{existing_kw}, {new_kw}".split(", "))))
                seen[title]["검색_키워드"] = merged_kw

    return list(seen.values())


def crawl_all(delay: float = 1.5) -> None:
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
        browser = p.chromium.launch(headless=True)
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
    args = parser.parse_args()

    SEED_DIR.mkdir(parents=True, exist_ok=True)
    crawl_all(delay=args.delay)


if __name__ == "__main__":
    main()
