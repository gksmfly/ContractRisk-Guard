# backend/scripts/parse_ftc_case_pdf.py
"""
공정거래위원회 시정조치 PDF 다운로드 및 텍스트 파싱 스크립트

ftc_cases_raw.json의 pdf_info(docId, docSn)를 이용하여
Playwright로 PDF를 다운로드하고 pdfplumber로 텍스트를 추출합니다.

사용법:
    python backend/scripts/parse_ftc_case_pdf.py
    python backend/scripts/parse_ftc_case_pdf.py --skip-download   # PDF 재다운로드 없이 파싱만

샘플 출력 데이터 (data/raw/ftc_cases/ftc_cases_parsed.json):
    {
        "총_건수": 300,
        "파싱_소스": "ftc_cases_raw.json",
        "추출_필드": ["조항_원문", "위반_유형", "근거_법령", "시정_내용", "risk_level"],
        "사례": [
            {
                "사건명": "㈜OO 불공정약관 시정",
                "셀_데이터": {"사건번호": "2024약관1234", "의결번호": "제2024-XXX호"},
                "pdf_파일": "2024약관1234_제2024-XXX호_㈜OO_불공정약관_시정.pdf",
                "전체_텍스트_길이": 12345,
                "조항_원문": ["약관 제5조(환불) 소비자는 구매일로부터 7일 이내에..."],
                "위반_유형": ["약관규제법 제9조 - 계약 해제·해지 제한 위반"],
                "근거_법령": ["약관의규제에관한법 제9조 제1항"],
                "시정_내용": ["환불 기간을 14일로 수정한다"],
                "risk_level": "High",
                "risk_level_근거": "공정위 시정조치 대상 = 위반 확정"
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.scripts.utils import save_json, setup_logger, PROJECT_ROOT

logger = setup_logger("parse_ftc_case_pdf.log")

FTC_DIR  = Path(os.environ.get("FTC_DIR",  str(PROJECT_ROOT / "data/raw/ftc_cases")))
PDF_DIR  = FTC_DIR / "pdfs"
# crawl_ftc_cases.py와 동일한 대표위반유형 필터. 기본값은 불공정약관(10*)이지만,
# 인접 카테고리(전자상거래 11*, 방문판매 12*, 가맹사업 13*, 할부거래 14* 등)를
# 크롤링할 때는 이 값을 맞춰줘야 다운로드 목록 순회가 해당 사건을 찾아낸다.
FTC_VIOLATION_TYPE = os.environ.get("FTC_VIOLATION_TYPE", "10*")


"""ftc_cases_raw.json에서 사례 목록을 로드합니다.

Args:
    filepath (Path): ftc_cases_raw.json 파일 경로.

Returns:
    list[dict[str, Any]]: 사건 레코드 딕셔너리 리스트 (파일 내 "사례" 키의 값).
"""

def load_raw_cases(filepath: Path) -> list[dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("사례", [])


BASE_URL = os.environ.get("FTC_CASE_URL", "https://case.ftc.go.kr/ocp/co/ltfr.do")


"""사건번호/의결번호/사건명을 조합한 안전한 식별자를 생성합니다.

파일 시스템에 사용할 수 없는 문자를 언더스코어로 치환합니다.

Args:
    case (dict[str, Any]): 사건 레코드 딕셔너리. "셀_데이터"와 "사건명" 키를 사용.
    fallback_index (int | None): 식별자를 생성할 수 없을 때 사용할 fallback 번호.

Returns:
    str: 파일명으로 사용 가능한 안전한 식별자 문자열 (최대 140자).
"""

def build_case_identifier(case: dict[str, Any], fallback_index: int | None = None) -> str:
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
    # 파일 시스템 한계(255 bytes) 대비 UTF-8 바이트 기준 180bytes 이하로 제한
    encoded = safe_name.encode("utf-8")
    if len(encoded) > 180:
        safe_name = encoded[:180].decode("utf-8", errors="ignore").strip("._")
    return safe_name or f"case_{fallback_index or 'unknown'}"


"""불공정약관 목록 페이지 URL을 생성합니다.

Args:
    page_index (int): 조회할 페이지 번호 (1부터 시작).

Returns:
    str: reprsntVioltTy=10*(대표위반유형=불공정약관) 필터가 적용된 목록 페이지 URL 문자열.
"""

def build_list_url(page_index: int) -> str:
    # crawl_ftc_cases.py와 동일한 필터를 써야 한다 — 파라미터명이 reprsntViolTy(오타)로
    # 다르면 필터가 전혀 적용되지 않아, caseNo=약관 목록만 순회하게 되고 사건번호에
    # "약관"이 없는(그러나 위반유형은 불공정약관인) 사건은 PDF를 못 찾는다.
    # FTC_VIOLATION_TYPE으로 카테고리를 바꿀 수 있다 — raw.json에 다른 카테고리
    # 사건이 섞여 있는데 이 값이 안 맞으면, 다운로드 목록 순회에서 그 사건들을
    # 영영 못 찾고(테이블은 매번 정상 로드되니 "테이블 없음"류 에러도 없이) 조용히
    # 남은 건수만 계속 쌓인다 — 반드시 raw.json을 만든 크롤링과 값을 맞춰야 한다.
    query = urlencode(
        {
            "pageIndex": page_index,
            "caseNo": "",
            "caseNm": "",
            "decsnNo": "",
            "startRceptDt": "",
            "endRceptDt": "",
            "reprsntManagtTyCd": "",
            "reprsntVioltTy": FTC_VIOLATION_TYPE,
            "searchKrwd": "",
        }
    )
    return f"{BASE_URL}?{query}"


"""Playwright 다운로드 이벤트를 이용하여 PDF를 저장합니다.

Args:
    page (Any): Playwright 페이지 객체.
    btn (Any): 클릭할 다운로드 버튼 엘리먼트.
    filepath (Path): PDF를 저장할 파일 경로.
    delay (float): 다운로드 완료 후 대기 시간(초).

Returns:
    bool: 다운로드 성공 시 True, 실패(100 bytes 미만 또는 예외) 시 False.
"""

def download_pdf_via_playwright(
    page: Any, btn: Any, filepath: Path, delay: float = 1.0
) -> bool:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        with page.expect_download(timeout=30000) as download_info:
            btn.click()
        download = download_info.value
        download.save_as(str(filepath))

        file_size = filepath.stat().st_size
        if file_size < 100:
            logger.warning(f"파일이 너무 작음 ({file_size} bytes): {filepath.name}")
            filepath.unlink(missing_ok=True)
            return False

        logger.info(f"다운로드 완료: {filepath.name} ({file_size:,} bytes)")
        time.sleep(delay)
        return True
    except Exception as e:
        logger.warning(f"다운로드 실패 - {filepath.name} | {e}")
        return False


"""PDF에서 전체 텍스트를 추출합니다.

Args:
    filepath (Path): 텍스트를 추출할 PDF 파일 경로.

Returns:
    str: 페이지별 텍스트를 줄바꿈으로 연결한 문자열.
         pdfplumber 미설치 또는 추출 실패 시 빈 문자열.
"""

TEXT_CACHE_DIR = FTC_DIR / "text_cache"      # FTC_DIR을 따라간다(경로 규칙)


def extract_text_from_pdf(filepath: Path, use_cache: bool = True) -> str:
    """PDF에서 텍스트를 뽑고 **캐시에 남긴다.**

    파싱 결과 JSON은 `전체_텍스트_길이`만 남기고 본문을 버린다. 그래서 원문이 필요할
    때마다 PDF 3,359개를 다시 연다 — 08-21 파서 수정, 08-30 조항 텍스트 확인,
    08-31 대조표 조사까지 **세 번째**다. 한 번 연 김에 남긴다.

    캐시는 PDF 파일명 기준이라 재파싱해도 재사용된다. 지우면 자동으로 다시 쌓인다.
    """
    if use_cache:
        cached = TEXT_CACHE_DIR / (filepath.stem + ".txt")
        if cached.exists():
            return cached.read_text(encoding="utf-8")

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

    text = "\n".join(text_parts)
    if use_cache and text:
        TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (TEXT_CACHE_DIR / (filepath.stem + ".txt")).write_text(text, encoding="utf-8")
    return text


"""약관 조항 원문을 추출합니다.

"피심인의 약관 제X조", "약관 제X조", "제X조 (제목)" 패턴을 탐색합니다.

Args:
    text (str): PDF에서 추출한 전체 텍스트.

Returns:
    list[str]: 20자 이상의 조항 원문 문자열 리스트 (중복 제거, 순서 유지).
"""

_MIN_CLAUSE_LEN = 20   # 이보다 짧으면 조항으로 보지 않는다(extract_clause_text의 하한과 동일)
_MAX_CLAUSE_LEN = 600  # 실제 조항 하나가 이보다 길면 뒤에 다른 내용(법조문 인용, 심결 서술)이
                        # 딸려왔을 가능성이 높음 — 뒷부분을 잘라 오염을 줄인다

# 조항 원문 뒤에는 보통 "이러한 사실은 ~증거로 인정된다", "나. 검토의견 : 무효",
# "(2) 심사의견 : 무 효", "나. 적용 법조 '약관의 규제에 관한 법률'..." 같은 증거 인용·
# 심결 의견·법조문 설명이 이어진다 — 이 지점부터는 잘라낸다.
# "검토의견"과 "심사의견"은 같은 의미로 문서마다 혼용되는데, "심사의견" 누락으로 인해
# 해당 섹션의 일반론적 위반기준 서술("사업자가 고객에게 '부당하게 과중'한 손해배상을
# 부담시키는...")이 evidence_span으로 잘못 추출되는 오염이 확인되어 추가한다.
_BOUNDARY_MARKERS = [
    "이러한 사실은", "검토의견", "심사의견", "불공정성 판단", "적용 법조", "약관법",
    "약관의 규제", "민법", "동법", "법률 제", "시행령", "판례는", "대법원",
    "정하고 있는 조항은", "무효로 한다",
    "공정성을 잃은 것으로 추정", "다음 각 호의 어느 하나에 해당하는",
    # --- 2026-08-21 추가 ---
    # 라벨링 파일럿에서 조항 2,093건 중 767건(36.6%)이 약관 조항이 아니라 의결서 본문·
    # 주문·PDF 잔해였다. LLM에게 "이 조항이 어느 조 위반인가"를 물으면서 공정위의
    # 판단문을 입력으로 준 셈이라, 아래 표지를 만나면 그 앞까지만 조항으로 본다.
    "해당됨", "해당된다", "해당한다", "해당되므로", "해당하므로",
    "시정권고", "시정명령", "시정조치", "공정거래위원회", "피심인",
    "위 약관조항은", "이 사건 약관", "심결", "의결한다", "권고한다", "명한다",
]

# 약관규제법 제6~17조, 민법 해지(543~553조)·손해배상(750~766조) 관련 조문은 그 자체가
# "제N조(제목)" 형태라 일반 패턴에 그대로 걸린다 — 헤더가 이 실제 조문명과 정확히
# 일치하면 피심인의 약관이 아니라 법 조문 그 자체다. 공백 유무가 PDF 추출 결과와
# 다를 수 있어 비교 시점에 공백을 전부 제거해서 맞춘다.
_STATUTE_ARTICLE_TITLES = {
    re.sub(r"\s+", "", t) for t in [
        # 약관의 규제에 관한 법률 제6~17조
        "제6조(일반원칙)", "제7조(면책조항의 금지)", "제8조(손해배상액의 예정)",
        "제9조(계약의 해제ㆍ해지)", "제9조(계약의 해제·해지)", "제10조(채무의 이행)",
        "제11조(고객의 권익 보호)", "제12조(의사표시의 의제)", "제13조(대리인의 책임 가중)",
        "제14조(소송 제기의 금지 등)", "제15조(적용의 제한)", "제16조(일부 무효의 특칙)",
        "제17조(불공정약관조항의 사용금지)", "제17조(시정 조치)",
        # 민법 제543~553조 (해지·해제)
        "제543조(해지, 해제권)", "제544조(이행지체와 해제)", "제545조(정기행위와 해제)",
        "제546조(이행불능과 해제)", "제547조(해지, 해제권의 불가분성)",
        "제548조(해제의 효과, 원상회복의무)", "제549조(원상회복의무와 동시이행)",
        "제550조(해지의 효과)", "제551조(해지, 해제와 손해배상)",
        "제552조(해제권행사여부의 최고권)", "제553조(훼손 등으로 인한 해제권의 소멸)",
        # 민법 제750~766조 (불법행위·손해배상)
        "제750조(불법행위의 내용)", "제751조(재산 이외의 손해의 배상)",
        "제752조(생명침해로 인한 위자료)", "제753조(미성년자의 책임능력)",
        "제754조(심신상실자의 책임능력)", "제755조(감독자의 책임)",
        "제756조(사용자의 배상책임)", "제757조(도급인의 책임)",
        "제758조(공작물등의 점유자, 소유자의 책임)", "제759조(동물의 점유자의 책임)",
        "제760조(공동불법행위자의 책임)", "제761조(정당방위, 긴급피난)",
        "제762조(손해배상청구권에 있어서의 태아의 지위)", "제763조(준용규정)",
        "제764조(명예훼손의 경우의 특칙)", "제765조(배상액의 경감청구)",
        "제766조(손해배상청구권의 소멸시효)",
    ]
}


# 1990년대 PDF는 소제목을 "심 사 의 견"처럼 글자 사이에 공백을 넣어 조판하는 경우가
# 있어 단순 substring 매칭(find)으로는 못 잡는다 — 마커의 각 글자 사이에 공백을
# 허용하는 정규식으로 컴파일해서 매칭한다.
_BOUNDARY_PATTERNS = [
    re.compile(r"\s*".join(re.escape(ch) for ch in marker))
    for marker in _BOUNDARY_MARKERS
]


def _truncate_at_boundary(clause: str) -> str:
    cut = len(clause)
    for pattern in _BOUNDARY_PATTERNS:
        m = pattern.search(clause)
        if m:
            cut = min(cut, m.start())
    return clause[:cut].strip()


# 시정권고 사건에서는 "제N조(제목)를 이 시정권고를 받은 날부터 60일 이내에 삭제 또는
# 수정할 것을 권고한다. / 시정권고 이유 / ... / 가. 약관조항 / [진짜 조항 원문]" 구조가
# 흔하다 — 앞부분(제N조 헤더 + 권고 명령문 + 위반유형 요약)이 "제N조(제목)" 패턴에 걸려
# 조항 원문인 것처럼 통째로 캡처된다. 이 경우 진짜 조항은 "가. 약관조항"류 마커 뒤에서
# 시작하므로, 그 지점으로 시작 위치를 다시 앵커링한다(뒷부분을 자르는 _truncate_at_boundary
# 와 반대로, 앞부분을 잘라낸다).
_RECOMMENDATION_PREAMBLE = re.compile(r"[을를]\s*이\s*시정권고를?\s*받은\s*날")
_CLAUSE_START_MARKER = re.compile(r"[<〈]\s*약관\s*조항\s*[>〉]|[가-힣]\s*\.\s*약관\s*조항")

# PDF 조판 잔해. pdfplumber가 페이지 바닥글("- 1 -")과 표/도형 마커를 본문에 섞어 넣는다.
# 조항 한복판에 끼어들기 때문에 경계 마커로는 못 자르고 따로 지워야 한다.
_PDF_ARTIFACTS = [
    re.compile(r"[-–—]\s*\d{1,3}\s*[-–—]"),      # 페이지 번호 "- 1 -"
    re.compile(r"[<〈]\s*약관\s*조항\s*[>〉]?"),   # 남은 섹션 마커
    re.compile(r"[<〈]\s*(?=$|\s)"),                # 끝에 매달린 여는 괄호
    re.compile(r"\u3000+"),                          # 전각 공백
]


# 조항 끝에 붙는 범례. 의결서는 조항을 인용한 뒤 "* 갑 : 위임인(청구인), 을 : 수임인(피심인)"
# 처럼 당사자 대응표를 각주로 단다 — 조항 본문이 아니므로 떼어낸다.
_TRAILING_LEGEND = re.compile(r"\s*[*※]\s*[“\"']?[갑을병][”\"']?\s*[:：].*$", re.S)


def _strip_pdf_artifacts(clause: str) -> str:
    clause = _TRAILING_LEGEND.sub("", clause)
    for pat in _PDF_ARTIFACTS:
        clause = pat.sub(" ", clause)
    return re.sub(r"\s+", " ", clause).strip(" <>〈〉·.,")


def _skip_recommendation_preamble(clause: str) -> str:
    """조항 앞에 붙은 주문·권고문을 잘라낸다.

    예전에는 "~을 이 시정권고를 받은 날" 문구가 앞 100자에 있을 때만 앵커링했는데,
    의결서 문형이 다양해 상당수를 놓쳤다(깨진 마크업 274건). `<약관조항>` 마커는
    그 자체로 "여기부터가 진짜 조항"을 뜻하므로, 마커가 조항 앞부분에 있으면
    권고문 여부와 무관하게 그 뒤부터 취한다.
    """
    m = _CLAUSE_START_MARKER.search(clause)
    if not m:
        return clause
    # 위치 비율로 판단하지 않는다 — 조각이 짧으면 마커가 뒤쪽에 오는 게 정상이라
    # 비율 조건이 오히려 진짜 조항을 버렸다. 기준은 "마커 뒤에 실질 내용이 남는가"다.
    rest = clause[m.end():].strip()
    if len(rest) < _MIN_CLAUSE_LEN:
        return clause
    return rest


def _is_statute_article(clause: str) -> bool:
    header = re.match(r"제\d+조\s*[\(（][^\)）]*[\)）]", clause)
    if not header:
        return False
    normalized = re.sub(r"\s+", "", header.group(0))
    return normalized in _STATUTE_ARTICLE_TITLES


# 경계 마커로 앞부분을 잘라내도, 조항이 아예 없이 의결서 서술만 캡처된 경우가 남는다
# (예: "제102조(과실의 취득)가 정하는 법정 과실은 ... 부당하게 불리한 조항으로 ...").
# 약관 조항은 사업자·고객의 권리의무를 규정하는 문장이지 제3자 시점의 판단문이 아니다.
_DECISION_PROSE = re.compile(
    r"부당하게\s*불리한\s*조항|불공정약관조항에?\s*해당|무효라고\s*할\s*것|"
    r"위반된다고?\s*(볼|할|판단)|것으로\s*판단된다|라고\s*할\s*것이다|"
    r"규정에\s*반할|살피건대|따라서\s*이\s*사건"
)


def _is_decision_prose(clause: str) -> bool:
    return bool(_DECISION_PROSE.search(clause))


# 의결서 주문은 "제3조(월 임대료)제2호, 제4조(관리비)제3호, 제16조(합의관할)를 이 시정권고를
# 받은 날부터 60일 이내에 삭제 또는 수정하고..." 처럼 **고칠 조문을 나열**한다. 경계 마커로
# "시정권고" 앞까지 자르면 "제3조(월 임대료)제2호, ... 제16조(합의관할)를 이"라는 조문 번호
# 나열만 남는데, 길이 필터(20자)를 통과해 조항인 척 들어온다 — 실제로 이 필터를 넣기 전
# 표본 검수에서 12건 중 3건이 이런 조각이었다.
#
# 조항 원문은 그 자체로 규범 문장이어야 한다. 조문 참조(제N조·제N항·제N호)와 구분자를
# 걷어냈을 때 실질 내용이 남지 않으면 인용 목록이지 조항이 아니다.
_ARTICLE_REF = re.compile(r"제\s*\d+\s*조(?:\s*[\(（][^\)）]*[\)）])?|제\s*\d+\s*[항호]|약관")
_REF_SEPARATORS = re.compile(r"[,、·ㆍ및\s\(\)（）]|이상|등|를|을|는|은|와|과|의|이")


# 주문 문장은 "…제16조(합의관할)를 이 시정권고를 받은 날부터…"로 이어진다. "시정권고"
# 앞에서 자르면 조사만 남아 "…를 이"로 끝나는데, 나열된 조문 제목에 일반 명사가 섞여
# 있으면(예: "입주계약서", "부대사항") 아래 잔여 한글 수 기준을 통과해버린다.
# 규범 문장은 "~한다/없다/있다"로 끝나지 조사로 끝나지 않으므로 별도 표지로 쓴다.
_DANGLING_PARTICLE = re.compile(r"[을를는은]\s*이$")


def _is_article_reference_list(clause: str) -> bool:
    if _DANGLING_PARTICLE.search(clause):
        return True
    residue = _ARTICLE_REF.sub(" ", clause)
    residue = _REF_SEPARATORS.sub("", residue)
    korean = sum(1 for ch in residue if "가" <= ch <= "힣")
    return korean < 10


def _is_mostly_non_korean(clause: str) -> bool:
    korean = sum(1 for ch in clause if "가" <= ch <= "힣")
    return korean < len(clause) * 0.3


# ── 섹션 기반 추출 ──────────────────────────────────────────────────────────
# 위 `extract_clause_text()`는 "제N조(제목)" 형식만 인식한다. 그런데 의결서 1,092건 중
# 410건(37.5%)이 조항 0건으로 나왔고, 원인은 조항이 없어서가 아니라 **조 번호가 없어서**였다:
#
#   [아고다]     "고객이 숙박예약을 취소하는 경우 ... 일률적으로 숙박대금 전액을
#                환불하지 않는다는 내용의 '환불불가' 조건"        ← 서술형
#   [루프트한자] "<특가 항공권 환불수수료(PENALTY)기준> ㅇ 출발전 : 환불불가"  ← 표/항목
#
# 플랫폼·항공·렌터카처럼 최근 사건일수록 의결서가 약관을 서술형·표로 인용한다.
# 조사해보니 **"약관조항" 섹션은 표본 80건 전부(100%)에 존재**하므로, 그 섹션을
# 추출 앵커로 쓴다. 앵커는 둘:
#
#   A. 주문   "…약관조항을 다시 사용하여서는 아니 된다." 뒤가 문제된 조항 내용
#   B. 본문   줄머리의 "N. 약관 조항" 섹션 헤더 뒤
#
# 종료 표지는 **줄머리 목록 번호**와 섹션명만 쓴다 — 문장 안의 "다."를 종료로 보면
# 조항이 중간에 잘린다(시제품에서 실제로 그랬다).
_SEC_ORDER = re.compile(
    r"약\s*관\s*조\s*항\s*을\s*다시\s*사용하여서는\s*아니\s*된다\s*[.。]?"
)
_SEC_HEADER = re.compile(r"(?:^|\n)\s*(?:\d+|[가-하]|[IVX]+)\s*[.、]\s*약관\s*조항\s*[:：]?")
_SEC_END = re.compile("|".join([
    r"\n\s*\d+\s*[.、]\s*피심인",
    r"\n\s*\d+\s*[.、]\s*[가-힣]",
    r"\n\s*[가-하]\s*[.、]\s*[가-힣]",
    r"\s*".join("적용법조"), r"\s*".join("심사의견"), r"\s*".join("검토의견"),
    r"\s*".join("위법성판단"), r"\s*".join("약관인지여부"), r"\s*".join("기초사실"),
    r"\s*".join("시정권고이유"), r"\s*".join("시정명령"),
    r"\n\s*이\s*유\s*\n",
]))
# 앵커 A는 "…아니 된다."로 끝나는데 PDF 줄바꿈 때문에 그 꼬리가 조항 앞에 남는 경우가 있다.
_SEC_LEAD_JUNK = re.compile(r"^\s*(?:하여서는|여서는|서는)?\s*아니\s*된다\s*[.。]?\s*|^[\s.。,、:：]+")

_SEC_MIN_LEN = 40   # 섹션 추출은 조 번호라는 신호가 없어 더 보수적으로 자른다


def extract_clause_sections(text: str) -> list[str]:
    """"약관조항" 섹션에서 조 번호 없는 조항 본문을 뽑는다."""
    found: list[str] = []
    for pattern, tail in ((_SEC_ORDER, 1500), (_SEC_HEADER, 1200)):
        for m in pattern.finditer(text):
            rest = text[m.end(): m.end() + tail]
            end = _SEC_END.search(rest, 20)
            seg = rest[: end.start()] if end else rest
            seg = _strip_pdf_artifacts(_SEC_LEAD_JUNK.sub("", seg))
            if len(seg) < _SEC_MIN_LEN:
                continue
            if _is_article_reference_list(seg) or _is_decision_prose(seg) or _is_mostly_non_korean(seg):
                continue
            found.append(seg[:_MAX_CLAUSE_LEN])

    # 긴 것부터 남기고, 다른 조각에 포함되는 것은 버린다(같은 섹션이 앵커 둘에 다 걸린다).
    found.sort(key=len, reverse=True)
    unique: list[str] = []
    for s in found:
        if not any(s in kept for kept in unique):
            unique.append(s)
    return unique


def extract_clause_text(text: str) -> list[str]:
    # "피심인의 약관 제X조"가 가장 구체적이고, 뒤로 갈수록 일반적이다 (일반 패턴은
    # 심결문 안의 법조문 인용까지 "제N조(...)"로 오매칭한다). 세 패턴 다 돌리되,
    # 아래 필터로 오매칭을 걸러낸다 — 특정 패턴에서만 잡히는 진짜 조항도 있어서
    # 첫 패턴에서 결과가 나와도 나머지를 건너뛰지 않는다.
    patterns = [
        r"(피심인의\s*약관\s*제\d+조[^\n]*(?:\n(?![\d]+\.)[^\n]*)*)",
        r"(약관\s*제\d+조[가-힣\s]*\([^\)]*\)[^\n]*(?:\n(?![\d]+\.)[^\n]*)*)",
        r"(제\d+조\s*[\(（][^\)）]*[\)）][^\n]*(?:\n(?!\s*제\d+조)[^\n]*)*)",
    ]

    clauses: list[str] = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            clause = _skip_recommendation_preamble(m.strip()[:_MAX_CLAUSE_LEN])
            clause = _truncate_at_boundary(clause)
            clause = _strip_pdf_artifacts(clause)
            if len(clause) <= _MIN_CLAUSE_LEN:
                continue
            if _is_statute_article(clause) or _is_mostly_non_korean(clause):
                continue
            if _is_decision_prose(clause) or _is_article_reference_list(clause):
                continue
            clauses.append(clause)

    seen: set[str] = set()
    unique: list[str] = []
    for c in clauses:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    # 조 번호 패턴이 아무것도 못 찾았을 때만 섹션 추출로 넘어간다 — 조 번호가 있는
    # 문서에서는 그쪽이 더 정확하고, 섹션 추출은 경계가 느슨해 오염 위험이 크다.
    if not unique:
        unique = extract_clause_sections(text)

    return unique


"""위반 유형을 추출합니다.

불공정약관 관련 상용 패턴과 약관규제법 조항 번호를 탐색합니다.

Args:
    text (str): PDF에서 추출한 전체 텍스트.

Returns:
    list[str]: 위반 유형 문자열 리스트 (중복 제거, 순서 유지).
               예: ["약관규제법 제9조 - 계약 해제·해지 제한 위반"]
"""

def extract_violation_type(text: str) -> list[str]:
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

    law_patterns = re.findall(
        r"약관(?:의\s*규제에\s*관한\s*법률?|규제에\s*관한\s*법률?|규제법|법)\s*제(\d+)조", text
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


"""근거 법령을 추출합니다.

약관규제법, 민법, 소비자기본법 등의 조항 참조를 탐색합니다.

Args:
    text (str): PDF에서 추출한 전체 텍스트.

Returns:
    list[str]: 근거 법령 문자열 리스트 (중복 제거, 순서 유지).
               예: ["약관의규제에관한법 제9조 제1항", "민법 제103조"]
"""

DOMAIN_CIVIL_ARTICLES: set[int] = set(range(103, 107)) | set(range(543, 554))


def extract_legal_basis(text: str) -> list[str]:
    bases: list[str] = []

    # 약관규제법 (표현 방식 전부 포함)
    yakgwan_matches = re.findall(
        r"약관(?:의\s*규제에\s*관한\s*법률?|규제에\s*관한\s*법률?|규제법|법)\s*(제\d+조(?:\s*제\d+항)?(?:\s*제\d+호)?)",
        text,
    )
    for m in yakgwan_matches:
        bases.append(f"약관의 규제에 관한 법률 {m.strip()}")

    # 민법 - 계약 해지(543~553), 불공정(103~106) 조문만
    civil_matches = re.findall(r"민법\s*제(\d+)조(?:\s*(제\d+항))?", text)
    for num_str, hang in civil_matches:
        try:
            if int(num_str) in DOMAIN_CIVIL_ARTICLES:
                entry = f"민법 제{num_str}조"
                if hang:
                    entry += f" {hang}"
                bases.append(entry)
        except ValueError:
            pass

    # 소비자기본법
    for m in re.findall(r"(소비자기본법\s*제\d+조(?:\s*제\d+항)?)", text):
        bases.append(m.strip())

    # 전자상거래법
    for m in re.findall(r"(전자상거래\s*등에서의\s*소비자보호에\s*관한\s*법률\s*제\d+조(?:\s*제\d+항)?)", text):
        bases.append(m.strip())

    return list(dict.fromkeys(bases))


"""시정 내용(수정된 조항)을 추출합니다.

시정명용, 수정/개선/삭제 내용, 「」 인용구 수정 패턴을 탐색합니다.

Args:
    text (str): PDF에서 추출한 전체 텍스트.

Returns:
    list[str]: 시정 내용 문자열 리스트 (10자 이상, 중복 제거, 순서 유지).
"""

def extract_corrective_action(text: str) -> list[str]:
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


"""단일 PDF를 파싱하여 구조화된 데이터를 반환합니다.

Args:
    filepath (Path): 파싱할 PDF 파일 경로.
    case_info (dict[str, Any]): 해당 사건의 메타데이터 딕셔너리 (사건명, 셀_데이터 등).

Returns:
    dict[str, Any]: 파싱 결과 딕셔너리. 텍스트 추출 실패 시 빈 딕셔너리.
                    키: "사건명", "셀_데이터", "pdf_파일", "전체_텍스트_길이",
                        "조항_원문", "위반_유형", "근거_법령", "시정_내용",
                        "risk_level", "risk_level_근거"
"""

def parse_single_pdf(filepath: Path, case_info: dict[str, Any]) -> dict[str, Any]:
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


"""Playwright로 목록 페이지를 순회하며 PDF를 일괄 다운로드합니다.

이미 존재하는 파일은 건너뜁니다. pdf_info가 없는 사건은 제외합니다.

Args:
    cases (list[dict[str, Any]]): 사건 레코드 리스트 (load_raw_cases 반환값).
    delay (float): 다운로드 요청 간 대기 시간(초).
    headless (bool): True이면 브라우저를 백그라운드에서 실행.

Returns:
    dict[str, Path]: 사건명 → PDF 파일 경로 매핑 딕셔너리.
                     예: {"㈜OO 불공정약관 시정": Path("data/raw/ftc_cases/pdfs/...pdf")}
"""

def run_download(
    cases: list[dict[str, Any]], delay: float, headless: bool = True
) -> dict[str, Path]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error("Playwright가 설치되어 있지 않습니다: pip install playwright && playwright install chromium")
        return {}

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}
    success, skipped, failed, no_pdf = 0, 0, 0, 0

    case_map: dict[str, tuple[dict[str, Any], str, int]] = {}
    for i, case in enumerate(cases, 1):
        pdf_info = case.get("pdf_info", {})
        # 다운로드 목록 순회 중엔 각 행의 첫 번째 열(사건번호, 아래 row_case_no)로
        # 대상을 찾는다 — case_map도 반드시 사건번호로 키를 잡아야 매칭된다.
        # pdf_info의 docId는 fileId(파일 식별자)라 사건번호와 전혀 다른 값이고,
        # 다운로드 버튼은 매칭된 행에서 CSS 선택자로 직접 찾아 클릭하므로 docId
        # 자체는 여기서 안 쓰인다 — "PDF 정보가 있는지"만 확인하는 용도.
        if not pdf_info.get("docId", ""):
            no_pdf += 1
            continue
        case_no = str(case.get("셀_데이터", {}).get("사건번호", "")).strip()
        if not case_no:
            no_pdf += 1
            continue
        safe_name = build_case_identifier(case, fallback_index=i)
        filepath = PDF_DIR / f"{safe_name}.pdf"
        if filepath.exists():
            skipped += 1
            downloaded[case.get("사건명", "")] = filepath
            continue
        case_map[case_no] = (case, safe_name, i)

    if not case_map:
        logger.info(
            f"PDF 다운로드 완료 - 성공: {success}, 건너뜀: {skipped}, "
            f"실패: {failed}, PDF없음: {no_pdf}"
        )
        return downloaded

    logger.info(f"다운로드 대상: {len(case_map)}건 (이미 존재: {skipped}건)")

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
        page = context.new_page()
        page.on("dialog", lambda d: d.accept())

        remaining = set(case_map.keys())
        page_index = 0

        while remaining:
            page_index += 1
            url = build_list_url(page_index)

            page_has_targets = True
            found_on_page = 0
            page_exhausted = False

            while page_has_targets and remaining:
                try:
                    page.goto(url, wait_until="networkidle", timeout=30000)
                except Exception as e:
                    logger.warning(f"페이지 {page_index} 로드 실패: {e}")
                    page_exhausted = True
                    break
                time.sleep(delay)

                table = page.query_selector("#contents table, table")
                if not table:
                    logger.info(f"페이지 {page_index}: 테이블 없음, 순회 종료")
                    page_exhausted = True
                    break

                rows = table.query_selector_all("tbody tr")
                if not rows:
                    logger.info(f"페이지 {page_index}: 행 없음, 순회 종료")
                    page_exhausted = True
                    break

                target_found = False
                for tr in rows:
                    tds = tr.query_selector_all("td")
                    if len(tds) < 2:
                        continue
                    row_case_no = tds[0].inner_text().strip()

                    if row_case_no not in remaining:
                        continue

                    case_info, safe_name, idx = case_map[row_case_no]
                    title = case_info.get("사건명", "")
                    filepath = PDF_DIR / f"{safe_name}.pdf"

                    pdf_btn = tr.query_selector("a.down_files.pdf")
                    if not pdf_btn:
                        logger.warning(f"[{idx}/{len(cases)}] PDF 버튼 없음: {title}")
                        remaining.discard(row_case_no)
                        failed += 1
                        continue

                    if download_pdf_via_playwright(page, pdf_btn, filepath, delay):
                        success += 1
                        downloaded[title] = filepath
                    else:
                        failed += 1

                    remaining.discard(row_case_no)
                    found_on_page += 1
                    target_found = True
                    break  # 페이지를 다시 로드해야 하므로 루프 탈출

                if not target_found:
                    page_has_targets = False

            logger.info(
                f"페이지 {page_index}: {found_on_page}건 다운로드 "
                f"(누적 성공: {success}, 남은: {len(remaining)})"
            )

            if page_exhausted or not remaining:
                break

        browser.close()

    logger.info(
        f"PDF 다운로드 완료 - 성공: {success}, 건너뜀: {skipped}, "
        f"실패: {failed}, PDF없음: {no_pdf}"
    )
    return downloaded


"""다운로드된 PDF를 파싱합니다.

Args:
    cases (list[dict[str, Any]]): 사건 레코드 리스트 (load_raw_cases 반환값).
    downloaded (dict[str, Path]): 사건명 → PDF 파일 경로 매핑 딕셔너리 (run_download 반환값).

Returns:
    list[dict[str, Any]]: 파싱 결과 딕셔너리 리스트. 빈 결과는 포함하지 않음.
"""

def run_parse(cases: list[dict[str, Any]], downloaded: dict[str, Path]) -> list[dict[str, Any]]:
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
    parser = argparse.ArgumentParser(
        description="공정거래위원회 시정조치 PDF 다운로드 및 텍스트 파싱"
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="PDF 다운로드 건너뛰고 파싱만 수행"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="다운로드 요청 간격(초, 기본값: 1.0)"
    )
    parser.add_argument(
        "--no-headless", action="store_true", help="브라우저 화면 표시 (디버깅용)"
    )
    parser.add_argument(
        "--raw-file", default=None,
        help="사건 목록 JSON 경로 (기본: data/raw/ftc_cases/ftc_cases_raw.json). "
             "ftc_cases_raw_FULL_1674.json을 주면 사건 1,674건 전체를 대상으로 한다."
    )
    parser.add_argument(
        "--only-missing-pdf", action="store_true",
        help="PDF가 아직 없는 사건만 다운로드 대상으로 삼는다(이미 받은 건 건너뜀)"
    )
    args = parser.parse_args()

    # 어느 사건 목록을 쓸지 고를 수 있어야 한다. `ftc_cases_raw.json`(1,092건)은
    # `ftc_cases_raw_FULL_1674.json`(1,674건)의 부분집합이라, 기본값만 쓰면 불공정약관
    # 사건 529건이 파이프라인에 아예 들어오지 않는다.
    raw_path = Path(args.raw_file) if args.raw_file else FTC_DIR / os.environ.get("FTC_RAW_FILE", "ftc_cases_raw.json")
    if not raw_path.exists():
        logger.error(f"원본 데이터 파일이 없습니다: {raw_path}")
        logger.error("먼저 crawl_ftc_cases.py를 실행하세요.")
        return

    cases = load_raw_cases(raw_path)
    logger.info(f"=== PDF 처리 시작 - 총 {len(cases)}건 ===")

    if args.skip_download:
        logger.info("PDF 다운로드 건너뜀 (--skip-download)")
        downloaded: dict[str, Path] = {}
        for i, case in enumerate(cases, 1):
            title = case.get("사건명", "")
            safe_name = build_case_identifier(case, fallback_index=i)
            filepath = PDF_DIR / f"{safe_name}.pdf"
            if filepath.exists():
                downloaded[title] = filepath
        logger.info(f"기존 PDF 파일 {len(downloaded)}건 발견")
    else:
        targets = cases
        if args.only_missing_pdf:
            targets = [
                c for i, c in enumerate(cases, 1)
                if not (PDF_DIR / f"{build_case_identifier(c, fallback_index=i)}.pdf").exists()
            ]
            logger.info(f"PDF 미보유 사건만 대상: {len(targets)}건 / 전체 {len(cases)}건")
        downloaded = run_download(targets, args.delay, headless=not args.no_headless)
        # 이미 받아둔 PDF도 파싱 대상에 포함시킨다.
        for i, c in enumerate(cases, 1):
            title = c.get("사건명", "")
            fp = PDF_DIR / f"{build_case_identifier(c, fallback_index=i)}.pdf"
            if title not in downloaded and fp.exists():
                downloaded[title] = fp

    parsed = run_parse(cases, downloaded)

    output_path = FTC_DIR / "ftc_cases_parsed.json"
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
