# scripts/parse_ftc_pdf.py
"""
공정거래위원회 시정조치 PDF 다운로드 및 텍스트 파싱 스크립트

ftc_cases_raw.json의 pdf_url에서 PDF를 다운로드하고
pdfplumber로 텍스트를 추출하여 구조화된 데이터로 변환합니다.

사전 설치:
    pip install pdfplumber requests

사용법:
    python scripts/parse_ftc_pdf.py
    python scripts/parse_ftc_pdf.py --skip-download   # PDF 재다운로드 없이 파싱만
"""

import json
import logging
import os
import re
import time
import argparse
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

LOG_DIR = Path(os.environ["LOG_DIR"])
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "parse_ftc_pdf.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

SEED_DIR = Path(os.environ.get("FTC_SEED_DIR", "data/seed"))
PDF_DIR = SEED_DIR / "pdfs"


def load_raw_cases(filepath: Path) -> list[dict[str, Any]]:
    """ftc_cases_raw.json에서 사례 목록을 로드합니다."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("사례", [])


def save_json(data: Any, filepath: Path) -> None:
    """JSON 데이터를 파일로 저장합니다."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


DOWNLOAD_URL = "https://case.ftc.go.kr/ocp/co/getFileList.do"

HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://case.ftc.go.kr/",
}


def build_case_identifier(case: dict[str, Any], fallback_index: int | None = None) -> str:
    """사건번호/의결번호/사건명을 조합한 안전한 식별자를 생성합니다."""
    cell_data = case.get("셀_데이터", {})
    identifier_parts = [
        str(cell_data.get("사건번호", "")).strip(),
        str(cell_data.get("의결번호", "")).strip(),
        str(case.get("사건명", "")).strip(),
    ]
    raw_name = " | ".join(part for part in identifier_parts if part)
    if not raw_name and fallback_index is not None:
        raw_name = f"case_{fallback_index}"
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", raw_name)
    safe_name = re.sub(r"\s+", "_", safe_name).strip("._")
    return safe_name[:140] or f"case_{fallback_index or 'unknown'}"


def download_pdf(doc_id: str, doc_sn: str, filepath: Path, delay: float = 1.0) -> bool:
    """docId, docSn을 POST로 전송하여 PDF를 다운로드합니다."""
    if filepath.exists():
        logger.info(f"이미 존재 - 건너뜀: {filepath.name}")
        return True

    for attempt in range(1, 4):
        try:
            response = requests.post(
                DOWNLOAD_URL,
                data={"docId": doc_id, "docSn": doc_sn},
                headers=HEADERS,
                timeout=30,
            )
            response.raise_for_status()

            if len(response.content) < 100:
                logger.warning(f"응답이 너무 작음 ({len(response.content)} bytes): {filepath.name}")
                return False

            content_type = response.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower() and not response.content.startswith(b"%PDF"):
                logger.warning(
                    "PDF 응답 검증 실패: %s | Content-Type=%s",
                    filepath.name,
                    content_type,
                )
                return False

            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(response.content)

            file_size = filepath.stat().st_size
            logger.info(f"다운로드 완료: {filepath.name} ({file_size:,} bytes)")
            time.sleep(delay)
            return True

        except requests.RequestException as e:
            if attempt < 3:
                logger.warning(f"재시도 {attempt}/3 - {filepath.name} | {e}")
                time.sleep(delay * attempt)
            else:
                logger.error(f"최종 실패 - {filepath.name} | {e}")

    return False


def extract_text_from_pdf(filepath: Path) -> str:
    """PDF에서 전체 텍스트를 추출합니다."""
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber가 설치되어 있지 않습니다: pip install pdfplumber")
        return ""

    text_parts: list[str] = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        logger.error(f"PDF 텍스트 추출 실패 - {filepath.name}: {e}")
        return ""

    return "\n".join(text_parts)


def extract_clause_text(text: str) -> list[str]:
    """약관 조항 원문을 추출합니다."""
    clauses: list[str] = []

    # "피심인의 약관 제X조", "약관 제X조" 패턴
    patterns = [
        r"(피심인의\s*약관\s*제\d+조[^\n]*(?:\n(?![\d]+\.)[^\n]*)*)",
        r"(약관\s*제\d+조[가-힣\s]*\([^\)]*\)[^\n]*(?:\n(?![\d]+\.)[^\n]*)*)",
        r"(제\d+조\s*[\(（][^\)）]*[\)）][^\n]*(?:\n(?!\s*제\d+조)[^\n]*)*)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            clause = m.strip()
            if len(clause) > 20:
                clauses.append(clause)

    # 중복 제거 (순서 유지)
    seen: set[str] = set()
    unique: list[str] = []
    for c in clauses:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique


def extract_violation_type(text: str) -> list[str]:
    """위반 유형을 추출합니다."""
    types: list[str] = []

    patterns = [
        r"(고객에게\s*부당하게\s*불리한\s*조항)",
        r"(상당한\s*이유\s*없이[^\n.]*불리한[^\n.]*)",
        r"(부당하게[^\n.]*면책[^\n.]*)",
        r"(손해배상[^\n.]*제한[^\n.]*)",
        r"(해제[·,\s]*해지[^\n.]*제한[^\n.]*)",
        r"(의사표시의\s*의제[^\n.]*)",
        r"(대리인의\s*책임[^\n.]*가중[^\n.]*)",
        r"(소제기의\s*금지[^\n.]*)",
        r"(불공정약관[^\n.]*해당[^\n.]*)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        types.extend(m.strip() for m in matches)

    # 약관규제법 위반 조항 유형 추출
    law_patterns = re.findall(
        r"약관(?:의)?규제(?:에\s*관한)?법\s*제(\d+)조", text
    )
    type_map: dict[str, str] = {
        "6": "일반원칙 위반",
        "7": "면책조항 금지 위반",
        "8": "손해배상액 예정 위반",
        "9": "계약 해제·해지 제한 위반",
        "10": "채무이행 관련 위반",
        "11": "고객 권익 제한 위반",
        "12": "의사표시 의제 위반",
        "13": "대리인 책임 가중 위반",
        "14": "소제기 금지 위반",
    }
    for num in law_patterns:
        if num in type_map:
            types.append(f"약관규제법 제{num}조 - {type_map[num]}")

    return list(dict.fromkeys(types))


def extract_legal_basis(text: str) -> list[str]:
    """근거 법령을 추출합니다."""
    bases: list[str] = []

    patterns = [
        r"(약관(?:의)?규제(?:에\s*관한)?법\s*제\d+조(?:\s*제\d+항)?(?:\s*제\d+호)?)",
        r"(민법\s*제\d+조(?:\s*제\d+항)?)",
        r"(소비자기본법\s*제\d+조(?:\s*제\d+항)?)",
        r"(전자상거래(?:등에서의\s*소비자보호에\s*관한)?법?\s*제\d+조(?:\s*제\d+항)?)",
        r"(상법\s*제\d+조(?:\s*제\d+항)?)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        bases.extend(m.strip() for m in matches)

    return list(dict.fromkeys(bases))


def extract_corrective_action(text: str) -> list[str]:
    """시정 내용(수정된 조항)을 추출합니다."""
    actions: list[str] = []

    patterns = [
        r"시정[내명]용[^\n]*\n((?:[^\n]+\n?)*?)(?=\n\s*\d+\.|$)",
        r"(?:수정|개선|삭제|변경)\s*(?:전|후|내용)[^\n]*\n((?:[^\n]+\n?)*?)(?=\n\s*\d+\.|$)",
        r"(「[^」]+」\s*(?:으로|로)\s*(?:수정|변경|개정)[^\n]*)",
        r"(삭제\s*(?:한다|하여야|하도록)[^\n]*)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            action = m.strip()
            if len(action) > 10:
                actions.append(action)

    return list(dict.fromkeys(actions))


def parse_single_pdf(filepath: Path, case_info: dict[str, Any]) -> dict[str, Any]:
    """단일 PDF를 파싱하여 구조화된 데이터를 반환합니다."""
    text = extract_text_from_pdf(filepath)
    if not text:
        return {}

    clauses = extract_clause_text(text)
    violations = extract_violation_type(text)
    legal_bases = extract_legal_basis(text)
    corrections = extract_corrective_action(text)

    result: dict[str, Any] = {
        "사건명": case_info.get("사건명", ""),
        "셀_데이터": case_info.get("셀_데이터", {}),
        "pdf_파일": filepath.name,
        "전체_텍스트_길이": len(text),
        "조항_원문": clauses,
        "위반_유형": violations,
        "근거_법령": legal_bases,
        "시정_내용": corrections,
        "risk_level": "High",
        "risk_level_근거": "공정위 시정조치 대상 = 위반 확정",
    }

    return result


def run_download(cases: list[dict[str, Any]], delay: float) -> dict[str, Path]:
    """PDF 파일을 일괄 다운로드합니다."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}
    success, skipped, failed, no_url = 0, 0, 0, 0

    for i, case in enumerate(cases, 1):
        pdf_info = case.get("pdf_info", {})
        doc_id = pdf_info.get("docId", "")
        doc_sn = pdf_info.get("docSn", "")
        title = case.get("사건명", f"case_{i}")
        safe_name = build_case_identifier(case, fallback_index=i)

        if not doc_id:
            logger.warning(f"[{i}/{len(cases)}] PDF 정보 없음: {title}")
            no_url += 1
            continue

        filepath = PDF_DIR / f"{safe_name}.pdf"
        if filepath.exists():
            skipped += 1
            downloaded[title] = filepath
            continue

        if download_pdf(doc_id, doc_sn, filepath, delay):
            success += 1
            downloaded[title] = filepath
        else:
            failed += 1

    logger.info(
        f"PDF 다운로드 완료 - 성공: {success}, 건너뜀: {skipped}, "
        f"실패: {failed}, URL없음: {no_url}"
    )
    return downloaded


def run_parse(cases: list[dict[str, Any]], downloaded: dict[str, Path]) -> list[dict[str, Any]]:
    """다운로드된 PDF를 파싱합니다."""
    parsed: list[dict[str, Any]] = []
    success, failed = 0, 0

    for i, case in enumerate(cases, 1):
        title = case.get("사건명", "")
        filepath = downloaded.get(title)

        if not filepath or not filepath.exists():
            continue

        logger.info(f"[{i}/{len(cases)}] 파싱 중: {title}")
        result = parse_single_pdf(filepath, case)
        if result:
            parsed.append(result)
            clause_count = len(result.get("조항_원문", []))
            violation_count = len(result.get("위반_유형", []))
            logger.info(
                f"[{i}/{len(cases)}] 파싱 완료: 조항 {clause_count}건, "
                f"위반유형 {violation_count}건"
            )
            success += 1
        else:
            logger.warning(f"[{i}/{len(cases)}] 파싱 결과 없음: {title}")
            failed += 1

    logger.info(f"PDF 파싱 완료 - 성공: {success}, 실패: {failed}")
    return parsed


def main() -> None:
    """메인 실행 함수입니다."""
    parser = argparse.ArgumentParser(
        description="공정거래위원회 시정조치 PDF 다운로드 및 텍스트 파싱"
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="PDF 다운로드 건너뛰고 파싱만 수행"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="다운로드 요청 간격(초, 기본값: 1.0)"
    )
    args = parser.parse_args()

    raw_path = SEED_DIR / "ftc_cases_raw.json"
    if not raw_path.exists():
        logger.error(f"원본 데이터 파일이 없습니다: {raw_path}")
        logger.error("먼저 crawl_ftc_seed.py를 실행하세요.")
        return

    cases = load_raw_cases(raw_path)
    logger.info(f"=== PDF 처리 시작 - 총 {len(cases)}건 ===")

    # 2단계: PDF 다운로드
    if args.skip_download:
        logger.info("PDF 다운로드 건너뜀 (--skip-download)")
        # 기존 다운로드 파일 매핑
        downloaded: dict[str, Path] = {}
        for i, case in enumerate(cases, 1):
            title = case.get("사건명", "")
            safe_name = build_case_identifier(case, fallback_index=i)
            filepath = PDF_DIR / f"{safe_name}.pdf"
            if filepath.exists():
                downloaded[title] = filepath
        logger.info(f"기존 PDF 파일 {len(downloaded)}건 발견")
    else:
        downloaded = run_download(cases, args.delay)

    # 3단계: PDF 텍스트 추출 및 파싱
    parsed = run_parse(cases, downloaded)

    # 4단계: 저장
    output_path = SEED_DIR / "ftc_cases_parsed.json"
    result = {
        "총_건수": len(parsed),
        "파싱_소스": "ftc_cases_raw.json",
        "추출_필드": ["조항_원문", "위반_유형", "근거_법령", "시정_내용", "risk_level"],
        "사례": parsed,
    }
    save_json(result, output_path)
    logger.info(f"=== 저장 완료: {output_path} ({len(parsed)}건) ===")


if __name__ == "__main__":
    main()
