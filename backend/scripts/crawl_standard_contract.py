# backend/scripts/crawl_standard_contract.py
"""
공정거래위원회 표준계약서 HWP 자동 다운로드 및 텍스트 추출 스크립트

6개 카테고리(표준약관, 표준하도급, 표준가맹, 표준유통거래, 표준대리점거래,
표준비밀유지)의 게시판을 순회하며 HWP 파일을 다운로드하고
olefile + zlib으로 텍스트를 추출하여 JSON으로 저장합니다.

사용법:
    python backend/scripts/crawl_standard_contract.py
    python backend/scripts/crawl_standard_contract.py --skip-download   # HWP 재다운로드 없이 파싱만
    python backend/scripts/crawl_standard_contract.py --category 표준약관  # 특정 카테고리만
    python backend/scripts/crawl_standard_contract.py --no-headless      # 브라우저 화면 표시

샘플 출력 데이터 (data/raw/contract/contracts_parsed.json):
    {
        "총_건수": 45,
        "카테고리별_건수": {"표준약관": 20, "표준하도급계약서": 10},
        "추출_필드": ["제목", "카테고리", "담당부서", "등록일", "원본_파일명", "추출_텍스트"],
        "사례": [
            {
                "제목": "표준약관 제1호 (용역계약)",
                "카테고리": "표준약관",
                "담당부서": "약관심사과",
                "등록일": "2024-01-15",
                "파일ID": "12345",
                "원본_파일명": "표준약관_표준약관_제1호_용역계약_12345.hwp",
                "텍스트_길이": 8500,
                "추출_텍스트": "제1조(목적) 이 약관은 용역계약에 관한..."
            }
        ]
    }
"""

import json
import os
import re
import struct
import sys
import time
import zlib
import argparse
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import save_json, setup_logger, PROJECT_ROOT

logger = setup_logger("crawl_standard_contract.log")

CONTRACT_DIR = Path(os.environ.get("CONTRACT_RAW_DIR", str(PROJECT_ROOT / "data/raw/contract")))
HWP_DIR = CONTRACT_DIR / "hwp"

BASE_URL = os.environ.get("CONTRACT_BASE_URL", "https://www.ftc.go.kr/www/selectBbsNttList.do")
DOWNLOAD_URL = os.environ.get("CONTRACT_DOWNLOAD_URL", "https://www.ftc.go.kr/www/downloadBbsFile.do")

CATEGORIES: dict[str, dict[str, str]] = {
    "표준약관": {"bordCd": "201", "key": "202"},
    "표준하도급계약서": {"bordCd": "202", "key": "203"},
    "표준가맹계약서": {"bordCd": "203", "key": "204"},
    "표준유통거래계약서": {"bordCd": "204", "key": "205"},
    "표준대리점거래계약서": {"bordCd": "205", "key": "206"},
    "표준비밀유지계약서": {"bordCd": "206", "key": "207"},
}


"""HWP 파싱 결과의 surrogate 문자를 재귀적으로 제거합니다.

JSON 직렬화 왕복(roundtrip)을 통해 중첩된 딕셔너리·리스트 전체에서
유효하지 않은 surrogate를 대체 문자(U+FFFD)로 치환합니다.
HWP 텍스트 추출 후 저장 직전에 호출합니다.

Args:
    data (Any): 정리할 데이터 (딕셔너리, 리스트 등).

Returns:
    Any: surrogate가 제거된 동일 구조의 데이터.
"""

def clean_surrogates_deep(data: Any) -> Any:
    return json.loads(
        json.dumps(data, ensure_ascii=False, default=str).encode("utf-8", errors="replace")
    )


"""안전한 파일명을 생성합니다.

파일 시스템에서 사용할 수 없는 문자를 언더스코어로 치환합니다.

Args:
    text (str): 파일명으로 변환할 원본 문자열.

Returns:
    str: 특수문자와 공백이 제거된 안전한 파일명 문자열 (최대 140자).
         빈 문자열이면 "unknown" 반환.
"""

def build_safe_filename(text: str) -> str:
    safe = re.sub(r'[\\/:*?"<>|]', "_", text)
    safe = re.sub(r"\s+", "_", safe).strip("._")
    return safe[:140] or "unknown"


"""게시판 목록 페이지 URL을 생성합니다.

Args:
    bord_cd (str): 게시판 코드 (예: "201" = 표준약관).
    key (str): 게시판 키 값 (예: "202").
    page_index (int): 조회할 페이지 번호 (1부터 시작).

Returns:
    str: pageUnit=10 고정, 해당 게시판의 목록 페이지 URL 문자열.
"""

def build_list_url(bord_cd: str, key: str, page_index: int) -> str:
    return (
        f"{BASE_URL}?pageUnit=10&searchCnd=all"
        f"&key={key}&bordCd={bord_cd}&pageIndex={page_index}"
    )


"""olefile + zlib을 이용하여 HWP 파일에서 텍스트를 추출합니다.

OLE 형식이 아닌 경우(HWPX 등) extract_text_from_hwpx로 폴백합니다.

Args:
    filepath (Path): 텍스트를 추출할 HWP 파일 경로.

Returns:
    str: 추출된 텍스트 문자열. 실패 시 빈 문자열.
"""

def extract_text_from_hwp(filepath: Path) -> str:
    try:
        import olefile
    except ImportError:
        logger.error("olefile이 설치되어 있지 않습니다: pip install olefile")
        return ""

    try:
        if not olefile.isOleFile(str(filepath)):
            logger.warning(f"OLE 형식이 아닌 파일 (HWPX 등): {filepath.name}")
            return extract_text_from_hwpx(filepath)

        ole = olefile.OleFileIO(str(filepath))
        text_parts: list[str] = []

        # HWP 파일 헤더에서 압축 여부 확인
        header = ole.openstream("FileHeader").read()
        is_compressed = (header[36] & 1) == 1

        for stream_name in ole.listdir():
            if stream_name[0] == "BodyText":
                raw_data = ole.openstream(stream_name).read()

                if is_compressed:
                    try:
                        raw_data = zlib.decompress(raw_data, -15)
                    except zlib.error:
                        try:
                            raw_data = zlib.decompress(raw_data)
                        except zlib.error:
                            logger.warning(
                                f"압축 해제 실패 - {filepath.name}/{'/'.join(stream_name)}"
                            )
                            continue

                text = extract_text_from_bodytext(raw_data)
                if text:
                    text_parts.append(text)

        ole.close()
        result = "\n".join(text_parts)
        return result.encode("utf-8", errors="replace").decode("utf-8")

    except Exception as e:
        logger.error(f"HWP 텍스트 추출 실패 - {filepath.name}: {e}")
        return ""


"""HWP BodyText 바이너리 데이터에서 텍스트를 추출합니다.

HWP 레코드 헤더(4바이트)를 순회하며 HWPTAG_PARA_TEXT(type=67) 레코드를 파싱합니다.

Args:
    data (bytes): 압축 해제된 BodyText 스트림 바이너리 데이터.

Returns:
    str: 파싱된 텍스트 문단을 줄바꿈으로 연결한 문자열.
"""

def extract_text_from_bodytext(data: bytes) -> str:
    text_parts: list[str] = []
    i = 0

    while i < len(data) - 4:
        # HWP 레코드 헤더 파싱 (4바이트)
        try:
            header = struct.unpack_from("<I", data, i)[0]
        except struct.error:
            break

        rec_type = header & 0x3FF
        rec_size = (header >> 20) & 0xFFF

        # 확장 크기 처리
        if rec_size == 0xFFF:
            if i + 8 > len(data):
                break
            rec_size = struct.unpack_from("<I", data, i + 4)[0]
            i += 8
        else:
            i += 4

        if i + rec_size > len(data):
            break

        # HWPTAG_PARA_TEXT (type=67)
        if rec_type == 67:
            rec_data = data[i : i + rec_size]
            text = decode_para_text(rec_data)
            if text.strip():
                text_parts.append(text.strip())

        i += rec_size

    return "\n".join(text_parts)


"""HWP 문단 텍스트 레코드를 디코딩합니다.

2바이트씩 읽어 유니코드 코드 포인트로 변환하며, 제어 문자는 별도 처리합니다.

Args:
    data (bytes): HWPTAG_PARA_TEXT 레코드의 바이너리 데이터.

Returns:
    str: 디코딩된 문단 텍스트 문자열.
"""

def decode_para_text(data: bytes) -> str:
    chars: list[str] = []
    j = 0

    while j < len(data) - 1:
        code = struct.unpack_from("<H", data, j)[0]
        j += 2

        # 제어 문자 처리
        if code < 32:
            if code == 13:  # 줄바꿈
                chars.append("\n")
            elif code in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23):
                # 인라인 제어 문자: 추가 바이트 건너뛰기
                if code in (9, 10):
                    j += 12  # 탭/필드 시작 등
                elif code in (1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23):
                    j += 12
            elif code == 24:
                pass  # 하이픈
            elif code == 30:
                chars.append("\u00A0")  # 묶음 빈칸
            elif code == 31:
                chars.append("\u00AD")  # 고정폭 빈칸
        else:
            chars.append(chr(code))

    return "".join(chars)


"""HWPX(ZIP 기반) 파일에서 텍스트를 추출합니다.

ZIP 내 section*.xml 파일을 파싱하여 XML 태그를 제거하고 텍스트만 추출합니다.

Args:
    filepath (Path): 텍스트를 추출할 HWPX 파일 경로.

Returns:
    str: 섹션별 텍스트를 줄바꿈으로 연결한 문자열. 실패 시 빈 문자열.
"""

def extract_text_from_hwpx(filepath: Path) -> str:
    import zipfile

    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            text_parts: list[str] = []
            for name in sorted(zf.namelist()):
                if "section" in name.lower() and name.endswith(".xml"):
                    xml_data = zf.read(name).decode("utf-8", errors="replace")
                    # XML 태그 제거하고 텍스트만 추출
                    clean_text = re.sub(r"<[^>]+>", "", xml_data)
                    clean_text = re.sub(r"\s+", " ", clean_text).strip()
                    if clean_text:
                        text_parts.append(clean_text)
            return "\n".join(text_parts)
    except Exception as e:
        logger.warning(f"HWPX 텍스트 추출 실패 - {filepath.name}: {e}")
        return ""


"""단일 카테고리의 전체 페이지를 순회하며 게시글 정보를 수집합니다.

연속된 두 페이지가 동일하면 수집을 중단합니다.

Args:
    page (Any): Playwright 페이지 객체.
    category_name (str): 카테고리 이름 (예: "표준약관").
    bord_cd (str): 게시판 코드 (예: "201").
    key (str): 게시판 키 값 (예: "202").
    delay (float): 페이지 요청 간 대기 시간(초).
    max_pages (int): 순회할 최대 페이지 수.

Returns:
    list[dict[str, Any]]: 게시글 정보 딕셔너리 리스트.
                          각 항목 키: "번호", "제목", "담당부서", "등록일", "카테고리", "첨부파일"
"""

def crawl_category(
    page: Any,
    category_name: str,
    bord_cd: str,
    key: str,
    delay: float,
    max_pages: int,
) -> list[dict[str, Any]]:
    logger.info(f"[{category_name}] 크롤링 시작 (bordCd={bord_cd}, key={key})")
    items: list[dict[str, Any]] = []
    previous_signature = ""

    for page_index in range(1, max_pages + 1):
        url = build_list_url(bord_cd, key, page_index)
        logger.info(f"[{category_name}] 페이지 {page_index} 크롤링: {url}")

        try:
            page.goto(url, wait_until="networkidle", timeout=30000)
        except Exception as e:
            logger.warning(f"[{category_name}] 페이지 {page_index} 로드 실패: {e}")
            break
        time.sleep(delay)

        table = page.query_selector("table.p-table")
        if not table:
            table = page.query_selector("table")
        if not table:
            logger.info(f"[{category_name}] 페이지 {page_index}: 테이블 없음, 종료")
            break

        rows = table.query_selector_all("tbody tr")
        if not rows:
            logger.info(f"[{category_name}] 페이지 {page_index}: 행 없음, 종료")
            break

        first_text = rows[0].inner_text().strip()[:50]
        last_text = rows[-1].inner_text().strip()[:50]
        page_signature = f"{first_text}||{last_text}||{len(rows)}"
        if page_signature == previous_signature:
            logger.info(f"[{category_name}] 페이지 {page_index}: 이전과 동일, 종료")
            break
        previous_signature = page_signature

        for tr in rows:
            tds = tr.query_selector_all("td")
            if len(tds) < 4:
                continue

            first_td_text = tds[0].inner_text().strip()
            if "없습니다" in first_td_text or "데이터" in first_td_text:
                break

            number = tds[0].inner_text().strip()
            title_el = tds[1].query_selector("a")
            title = title_el.inner_text().strip() if title_el else tds[1].inner_text().strip()
            department = tds[2].inner_text().strip() if len(tds) > 2 else ""
            reg_date = tds[3].inner_text().strip() if len(tds) > 3 else ""

            file_links: list[dict[str, str]] = []
            attach_td = tds[4] if len(tds) > 4 else None
            if attach_td:
                for a in attach_td.query_selector_all("a[href*='downloadBbsFile']"):
                    href = a.get_attribute("href") or ""
                    file_id_match = re.search(r"atchmnflNo=(\d+)", href)
                    if file_id_match:
                        file_links.append({
                            "file_id": file_id_match.group(1),
                            "download_url": f"{DOWNLOAD_URL}?atchmnflNo={file_id_match.group(1)}",
                        })

            item: dict[str, Any] = {
                "번호": number,
                "제목": title,
                "담당부서": department,
                "등록일": reg_date,
                "카테고리": category_name,
                "첨부파일": file_links,
            }
            items.append(item)

        logger.info(
            f"[{category_name}] 페이지 {page_index} 완료: {len(rows)}건 (누적: {len(items)}건)"
        )

    logger.info(f"[{category_name}] 크롤링 완료 - 총 {len(items)}건")
    return items


"""Playwright로 HWP 파일을 일괄 다운로드합니다.

이미 존재하는 파일은 건너뜁니다. 다운로드 후 파일 크기가 100 bytes 미만이면 삭제합니다.

Args:
    items (list[dict[str, Any]]): crawl_category 반환값의 게시글 리스트.
    delay (float): 파일 다운로드 간 대기 시간(초).
    headless (bool): True이면 브라우저를 백그라운드에서 실행.

Returns:
    dict[str, Path]: file_id → 로컬 HWP 파일 경로 매핑 딕셔너리.
                     예: {"12345": Path("data/raw/contract/hwp/표준약관_제1호_12345.hwp")}
"""

def download_hwp_files(
    items: list[dict[str, Any]], delay: float, headless: bool = True
) -> dict[str, Path]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error("Playwright가 설치되어 있지 않습니다: pip install playwright && playwright install chromium")
        return {}

    HWP_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}
    success, skipped, failed = 0, 0, 0

    targets: list[tuple[dict[str, Any], dict[str, str]]] = []
    for item in items:
        for file_info in item.get("첨부파일", []):
            file_id = file_info.get("file_id", "")
            if not file_id:
                continue
            safe_name = build_safe_filename(f"{item['카테고리']}_{item['제목']}_{file_id}")
            filepath = HWP_DIR / f"{safe_name}.hwp"
            if filepath.exists():
                skipped += 1
                downloaded[file_id] = filepath
                continue
            targets.append((item, file_info))

    if not targets:
        logger.info(
            f"HWP 다운로드 완료 - 성공: {success}, 건너뜀: {skipped}, 실패: {failed}"
        )
        return downloaded

    logger.info(f"다운로드 대상: {len(targets)}건 (이미 존재: {skipped}건)")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="ko-KR",
            accept_downloads=True,
        )
        bpage = context.new_page()

        # 세션 확보를 위해 목록 페이지 접속
        try:
            bpage.goto(
                f"{BASE_URL}?bordCd=201&key=202",
                wait_until="networkidle",
                timeout=30000,
            )
            time.sleep(1)
        except Exception as e:
            logger.warning(f"초기 페이지 접속 실패: {e}")

        for idx, (item, file_info) in enumerate(targets, 1):
            file_id = file_info["file_id"]
            url = file_info["download_url"]
            safe_name = build_safe_filename(
                f"{item['카테고리']}_{item['제목']}_{file_id}"
            )
            filepath = HWP_DIR / f"{safe_name}.hwp"

            logger.info(
                f"[{idx}/{len(targets)}] 다운로드 중: {item['제목'][:40]} (ID: {file_id})"
            )

            try:
                with bpage.expect_download(timeout=30000) as download_info:
                    bpage.evaluate(
                        """(url) => {
                            const a = document.createElement('a');
                            a.href = url;
                            document.body.appendChild(a);
                            a.click();
                            a.remove();
                        }""",
                        url,
                    )
                download = download_info.value
                suggested = download.suggested_filename

                if suggested:
                    ext = Path(suggested).suffix.lower()
                    if ext and ext != ".hwp":
                        filepath = filepath.with_suffix(ext)

                download.save_as(str(filepath))
                file_size = filepath.stat().st_size

                if file_size < 100:
                    logger.warning(f"파일이 너무 작음 ({file_size} bytes): {filepath.name}")
                    filepath.unlink(missing_ok=True)
                    failed += 1
                    continue

                logger.info(f"다운로드 완료: {filepath.name} ({file_size:,} bytes)")
                downloaded[file_id] = filepath
                success += 1
                time.sleep(delay)

            except Exception as e:
                logger.warning(f"다운로드 실패 - {item['제목'][:40]}: {e}")
                failed += 1

        browser.close()

    logger.info(
        f"HWP 다운로드 완료 - 성공: {success}, 건너뜀: {skipped}, 실패: {failed}"
    )
    return downloaded


"""다운로드된 HWP 파일에서 텍스트를 추출합니다.

Args:
    items (list[dict[str, Any]]): crawl_category 반환값의 게시글 리스트.
    downloaded (dict[str, Path]): download_hwp_files 반환값 (file_id → 파일 경로 매핑).

Returns:
    list[dict[str, Any]]: 파싱 결과 딕셔너리 리스트. 텍스트 추출 실패 항목은 제외됨.
                          각 항목 키: "제목", "카테고리", "담당부서", "등록일", "파일ID",
                                      "원본_파일명", "텍스트_길이", "추출_텍스트"
"""

def parse_hwp_files(
    items: list[dict[str, Any]], downloaded: dict[str, Path]
) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    success, failed = 0, 0

    for item in items:
        for file_info in item.get("첨부파일", []):
            file_id = file_info.get("file_id", "")
            filepath = downloaded.get(file_id)

            if not filepath or not filepath.exists():
                continue

            logger.info(f"파싱 중: {item['제목'][:40]} ({filepath.name})")
            text = extract_text_from_hwp(filepath)

            if not text:
                logger.warning(f"텍스트 추출 실패: {filepath.name}")
                failed += 1
                continue

            result: dict[str, Any] = clean_surrogates_deep({
                "제목": item["제목"],
                "카테고리": item["카테고리"],
                "담당부서": item["담당부서"],
                "등록일": item["등록일"],
                "파일ID": file_id,
                "원본_파일명": filepath.name,
                "텍스트_길이": len(text),
                "추출_텍스트": text,
            })
            parsed.append(result)
            success += 1
            logger.info(
                f"파싱 완료: {item['제목'][:40]} (텍스트 {len(text):,}자)"
            )

    logger.info(f"HWP 파싱 완료 - 성공: {success}, 실패: {failed}")
    return parsed


"""전체 카테고리를 크롤링하고 HWP 텍스트를 추출합니다.

1단계: 카테고리별 게시글 목록 수집 → 2단계: HWP 다운로드
→ 3단계: 텍스트 추출 → 4단계: 카테고리별 + 통합 JSON 저장.

Args:
    delay (float): 요청 간 대기 시간(초).
    headless (bool): True이면 브라우저를 백그라운드에서 실행.
    max_pages (int): 카테고리별 최대 순회 페이지 수.
    target_category (str | None): 지정 시 해당 카테고리만 수집. None이면 전체 수집.
    skip_download (bool): True이면 HWP 다운로드를 건너뛰고 기존 파일만 파싱.
"""

def crawl_all(
    delay: float = 1.0,
    headless: bool = True,
    max_pages: int = 50,
    target_category: str | None = None,
    skip_download: bool = False,
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error("Playwright가 설치되어 있지 않습니다: pip install playwright && playwright install chromium")
        return

    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = CONTRACT_DIR / "contract_list_raw.json"

    if skip_download and raw_path.exists():
        logger.info("기존 목록 데이터 로드")
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        all_items = raw_data.get("사례", [])
    else:
        categories = CATEGORIES
        if target_category:
            if target_category not in CATEGORIES:
                logger.error(f"알 수 없는 카테고리: {target_category}")
                logger.error(f"사용 가능: {', '.join(CATEGORIES.keys())}")
                return
            categories = {target_category: CATEGORIES[target_category]}

        logger.info(f"=== 표준계약서 수집 시작 ({len(categories)}개 카테고리) ===")
        all_items: list[dict[str, Any]] = []

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

            for cat_name, cat_params in categories.items():
                try:
                    items = crawl_category(
                        page,
                        cat_name,
                        cat_params["bordCd"],
                        cat_params["key"],
                        delay,
                        max_pages,
                    )
                    all_items.extend(items)
                except Exception as e:
                    logger.error(f"[{cat_name}] 크롤링 실패: {e}")

            browser.close()

        raw_data = {
            "총_건수": len(all_items),
            "카테고리": list(categories.keys()),
            "사례": all_items,
        }
        save_json(raw_data, raw_path)
        logger.info(f"목록 저장 완료: {raw_path} ({len(all_items)}건)")

    if skip_download:
        logger.info("HWP 다운로드 건너뜀 (--skip-download)")
        downloaded: dict[str, Path] = {}
        for item in all_items:
            for file_info in item.get("첨부파일", []):
                file_id = file_info.get("file_id", "")
                if not file_id:
                    continue
                safe_name = build_safe_filename(
                    f"{item['카테고리']}_{item['제목']}_{file_id}"
                )
                # HWP 또는 다른 확장자 파일 검색
                for ext in [".hwp", ".hwpx", ".pdf", ".doc", ".docx", ".zip"]:
                    filepath = HWP_DIR / f"{safe_name}{ext}"
                    if filepath.exists():
                        downloaded[file_id] = filepath
                        break
        logger.info(f"기존 파일 {len(downloaded)}건 발견")
    else:
        downloaded = download_hwp_files(all_items, delay, headless)

    parsed = parse_hwp_files(all_items, downloaded)

    cat_groups: dict[str, list[dict[str, Any]]] = {}
    for item in parsed:
        cat = item.get("카테고리", "기타")
        cat_groups.setdefault(cat, []).append(item)

    for cat_name, cat_items in cat_groups.items():
        safe_cat = build_safe_filename(cat_name)
        cat_path = CONTRACT_DIR / f"contracts_{safe_cat}.json"
        cat_result = {
            "총_건수": len(cat_items),
            "카테고리": cat_name,
            "추출_필드": ["제목", "카테고리", "담당부서", "등록일", "원본_파일명", "추출_텍스트"],
            "사례": cat_items,
        }
        save_json(cat_result, cat_path)
        logger.info(f"  {cat_name}: {len(cat_items)}건 → {cat_path}")

    output_path = CONTRACT_DIR / "contracts_parsed.json"
    result = {
        "총_건수": len(parsed),
        "카테고리별_건수": {cat: len(items) for cat, items in cat_groups.items()},
        "추출_필드": ["제목", "카테고리", "담당부서", "등록일", "원본_파일명", "추출_텍스트"],
        "사례": parsed,
    }
    save_json(result, output_path)
    logger.info(f"=== 저장 완료: 전체 {output_path} ({len(parsed)}건) ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="공정거래위원회 표준계약서 HWP 다운로드 및 텍스트 추출"
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="HWP 다운로드 건너뛰고 파싱만 수행"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="요청 간격(초, 기본값: 1.0)"
    )
    parser.add_argument(
        "--no-headless", action="store_true", help="브라우저 화면 표시 (디버깅용)"
    )
    parser.add_argument(
        "--max-pages", type=int, default=50, help="카테고리별 최대 순회 페이지 수(기본값: 50)"
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help=f"특정 카테고리만 수집 ({', '.join(CATEGORIES.keys())})"
    )
    args = parser.parse_args()

    crawl_all(
        delay=args.delay,
        headless=not args.no_headless,
        max_pages=args.max_pages,
        target_category=args.category,
        skip_download=args.skip_download,
    )


if __name__ == "__main__":
    main()
