# -*- coding: utf-8 -*-
"""시연용 샘플 계약서 PDF 생성기.

/api/analyze-pdf 업로드 시연에 쓸 가상 계약서를 만든다. 본문은 약관규제법
제6~14조(backend/labeling/articles.py의 taxonomy)에 걸리도록 의도적으로 구성한
독소조항과, 대비용 정상 조항을 섞었다.

주의 — 본문에 `제N조` 형태의 상호참조를 쓰지 말 것. backend/api/services/analyze.py의
split_clauses()가 `(?=제\\s*\\d+\\s*조|...)`로 조항을 나누기 때문에, 조문 안에서
다른 조를 인용하면 그 지점에서 조항이 쪼개진다. 그래서 전부 "본 계약"으로 썼다.

폰트 준비(최초 1회) — 결과 PDF에 한글을 임베드하기 위해 필요하다:

    pip install reportlab fonttools
    curl -sL -o /tmp/NotoKR.ttf \\
      'https://github.com/google/fonts/raw/main/ofl/notosanskr/NotoSansKR%5Bwght%5D.ttf'
    python - <<'EOF'
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont
    import os
    os.makedirs('demo/_fonts', exist_ok=True)
    for w, out in [(400, 'KR-Regular.ttf'), (700, 'KR-Bold.ttf')]:
        f = TTFont('/tmp/NotoKR.ttf')
        instantiateVariableFont(f, {'wght': w}, inplace=True, updateFontNames=True)
        f.save('demo/_fonts/' + out)
    EOF

주의 — Google Fonts CSS API(`fonts.googleapis.com/css?family=Noto+Sans+KR`)가 주는 TTF는
**라틴 subset이라 한글 글리프가 0개다**(36KB). 그걸 임베드하면 본문이 통째로 빈칸이 된다.
반드시 위 github.com/google/fonts의 가변폰트 원본(약 10MB)에서 인스턴스를 떠야 한다.

또 하나 — 조항 분리는 사실상 `제N조` 패턴에만 의존한다. pypdf는 페이지 **안**에서는
여백을 아무리 키워도 빈 줄을 만들지 않기 때문이다(Spacer 60mm까지 확인). `\n{2,}` 경계가
생기는 곳은 페이지 이음매뿐이다 — 각 페이지 텍스트가 `\n`으로 끝나는데 추출부가 페이지를
다시 `\n`으로 잇는다(backend/api/routers/analyze.py `_extract_pdf_text`).

그래서 각 조항을 KeepTogether로 묶는 것이 **레이아웃 취향이 아니라 정확성 요건**이다.
묶지 않으면 페이지 중간에서 끊긴 조항이 이음매의 `\n\n` 때문에 두 조각으로 갈라져
서로 다른 조항으로 분석된다.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether,
)
from reportlab.lib import colors
import os
import sys

# 한글 폰트를 PDF에 임베드한다. 비임베드 CID 폰트(HYSMyeongJo 등)는 뷰어가 한국어
# 폰트팩을 갖고 있어야 제대로 보이는데, 시연은 어느 노트북에서 열릴지 모른다.
# Noto Sans KR 가변폰트에서 wght 400/700 인스턴스를 떠서 쓴다(OFL, 임베딩 허용).
# reportlab이 사용 글리프만 subset 하므로 원본이 6MB여도 결과 PDF는 작다.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 한글 TTF는 저장소에 커밋하지 않는다(6MB × 2, OFL 재배포 부담). 아래 준비 절차 참고.
_FONT_DIR = os.environ.get("DEMO_FONT_DIR", os.path.join(PROJECT_ROOT, "demo", "_fonts"))
pdfmetrics.registerFont(TTFont("KR", _FONT_DIR + "/KR-Regular.ttf"))
pdfmetrics.registerFont(TTFont("KR-Bold", _FONT_DIR + "/KR-Bold.ttf"))
# <b> 태그가 Bold 페이스로 매핑되도록 패밀리를 등록한다(안 하면 굵게가 무시된다).
registerFontFamily("KR", normal="KR", bold="KR-Bold", italic="KR", boldItalic="KR-Bold")
SERIF, SANS = "KR", "KR-Bold"

TITLE = "온라인 서비스 이용계약서"

# (제목, 본문) — 위험 조항과 정상 조항을 섞는다. 조 번호는 목록 순서에서 자동으로
# 매긴다(수기로 달면 조항을 끼워 넣을 때 번호가 어긋난다).
CLAUSES = [
    ("목적",
     "본 계약은 주식회사 가나다소프트(이하 \"회사\"라 한다)가 제공하는 온라인 문서관리 "
     "서비스(이하 \"서비스\"라 한다)의 이용에 관하여 회사와 이용자 사이의 권리, 의무 및 "
     "책임사항을 정함을 목적으로 한다."),
    ("용어의 정의",
     "\"이용자\"란 본 계약에 동의하고 회사가 제공하는 서비스를 이용하는 자를 말한다. "
     "\"유료서비스\"란 회사가 이용자에게 유상으로 제공하는 서비스를 말하며, \"콘텐츠\"란 "
     "이용자가 서비스를 통하여 등록하거나 저장한 일체의 자료를 말한다."),
    ("계약기간 및 갱신",
     "본 계약의 기간은 가입일로부터 1년으로 한다. 이용자가 기간 만료일 30일 전까지 서면으로 "
     "별도의 의사표시를 하지 아니하는 경우 본 계약은 동일한 조건으로 자동으로 갱신된 것으로 "
     "보며, 갱신된 계약기간 중에는 이용자의 중도 해지를 일절 허용하지 아니한다."),
    ("이용요금 및 납부",
     "서비스의 이용요금은 회사가 정하여 홈페이지에 게시한 요금표에 따른다. 이용자는 매월 "
     "말일까지 해당 월분의 이용요금을 회사가 지정한 계좌로 납부하여야 하며, 납부가 지연되는 "
     "경우 회사는 지연일수에 대하여 상법이 정한 이율의 지연손해금을 청구할 수 있다."),
    ("상계 및 기한의 이익",
     "이용자는 회사에 대하여 가지는 어떠한 채권으로도 이용요금 채무와 상계할 수 없다. "
     "이용요금의 납부가 1회라도 지연되는 경우 이용자는 별도의 최고 없이 기한의 이익을 "
     "상실하며, 잔여 계약기간에 해당하는 이용요금 전액을 즉시 지급하여야 한다."),
    ("요금 및 서비스 내용의 변경",
     "회사는 서비스의 이용요금, 제공 범위 및 내용을 이용자의 동의를 받지 아니하고 회사의 "
     "판단에 따라 임의로 변경할 수 있으며, 변경된 내용은 이용자에 대한 별도의 통지 없이 "
     "홈페이지에 게시한 날부터 즉시 효력이 발생한다."),
    ("서비스의 중단",
     "회사는 운영상 또는 기술상의 필요가 있다고 판단하는 경우 사전 통지 없이 서비스의 전부 "
     "또는 일부의 제공을 중단할 수 있으며, 이로 인하여 이용자에게 발생한 손해에 대하여는 그 "
     "사유를 불문하고 배상하지 아니한다."),
    ("회사의 면책",
     "회사는 서비스의 이용과 관련하여 이용자에게 발생한 일체의 손해에 대하여 회사의 고의 "
     "또는 중대한 과실이 있는 경우를 포함하여 어떠한 책임도 부담하지 아니한다."),
    ("콘텐츠의 보관",
     "회사는 이용자가 등록한 콘텐츠를 보관할 의무를 지지 아니한다. 콘텐츠의 멸실, 훼손 또는 "
     "유출에 대비한 백업과 보안 조치는 전적으로 이용자의 책임과 비용으로 하며, 회사는 이에 "
     "관하여 어떠한 담보책임도 지지 아니한다."),
    ("위약금",
     "이용자가 본 계약을 위반하여 회사에 손해가 발생한 경우, 이용자는 회사의 실제 손해액에 "
     "대한 입증 여부와 관계없이 손해배상과 별도로 연간 이용요금의 3배에 해당하는 금액을 "
     "위약금으로 회사에 지급하여야 한다."),
    ("계약의 해지",
     "회사는 사유를 불문하고 이용자에 대한 통보만으로 언제든지 본 계약을 즉시 해지할 수 있다. "
     "이에 반하여 이용자는 회사가 서면으로 승낙한 경우에 한하여 본 계약을 해지할 수 있으며, "
     "이미 납부한 이용요금은 어떠한 경우에도 반환하지 아니한다."),
    ("계약 종료 시의 조치",
     "본 계약이 종료된 경우 이용자는 회사로부터 제공받은 일체의 자료를 자신의 비용으로 즉시 "
     "폐기하고 그 폐기 사실을 회사가 지정하는 방식으로 입증하여야 하며, 이에 소요되는 모든 "
     "비용과 위험은 이용자가 부담한다."),
    ("개인정보 처리방침의 변경",
     "회사가 개인정보 처리방침의 변경 내용을 통지한 후 이용자가 7일 이내에 서면으로 이의를 "
     "제기하지 아니한 경우, 이용자는 변경된 처리방침의 전부에 대하여 동의한 것으로 본다."),
    ("권리와 의무의 양도",
     "이용자는 회사의 사전 서면 동의 없이 본 계약상의 권리와 의무의 전부 또는 일부를 제3자에게 "
     "양도하거나 담보로 제공할 수 없다. 회사가 영업을 양도하는 경우에도 같다."),
    ("대리인의 책임",
     "이용자를 대리하여 본 계약을 체결한 자는 이용자가 본 계약상의 의무를 이행하지 아니하는 "
     "경우 이용자와 연대하여 그 이행 및 손해배상의 책임을 부담한다."),
    ("분쟁의 해결",
     "본 계약과 관련하여 회사와 이용자 사이에 분쟁이 발생한 경우 이용자는 법원에 소송을 제기할 "
     "수 없으며, 회사가 단독으로 지정한 기관의 중재 판정에 따라 최종적으로 해결한다."),
    ("관할법원",
     "본 계약에 관하여 소송상의 다툼이 있는 경우 그 관할법원은 회사의 본점 소재지를 관할하는 "
     "법원을 전속적 관할법원으로 한다."),
    # 마지막 조항은 일부러 무해한 부칙으로 둔다. 말미의 작성일·서명란은 페이지 이음매가
    # 어디에 걸리느냐에 따라 마지막 조항 덩어리에 흡수될 수 있는데(페이지 안에서는 빈 줄이
    # 안 생기므로), 그 자리에 핵심 독소조항이 오면 시연 화면의 해당 카드에 서명란 텍스트가
    # 딸려 붙는다. 관할법원 조항을 끝에 두지 않는 이유다.
    ("기타",
     "본 계약에서 정하지 아니한 사항과 본 계약의 해석에 관하여 다툼이 있는 경우에는 관계 "
     "법령 및 일반 상관례에 따라 회사와 이용자가 협의하여 정한다."),
]

body = ParagraphStyle("body", fontName=SERIF, fontSize=10, leading=16.5,
                      alignment=TA_JUSTIFY, spaceAfter=0)
head = ParagraphStyle("head", fontName=SANS, fontSize=10.5, leading=15,
                      textColor=colors.HexColor("#111827"), spaceAfter=3)
title = ParagraphStyle("title", fontName=SANS, fontSize=17, leading=23,
                       alignment=TA_CENTER, spaceAfter=2)
notice = ParagraphStyle("notice", fontName=SANS, fontSize=8.2, leading=12.5,
                        alignment=TA_CENTER, textColor=colors.HexColor("#9A3412"))
meta = ParagraphStyle("meta", fontName=SERIF, fontSize=9, leading=14)
sign = ParagraphStyle("sign", fontName=SERIF, fontSize=9.5, leading=17)


def build(path):
    doc = SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=20 * mm, bottomMargin=18 * mm,
        title=TITLE, author="Verilex 시연용 샘플",
        subject="약관규제법 분석 시연을 위한 가상 계약서",
    )
    story = [Paragraph(TITLE, title), Spacer(1, 3 * mm)]

    box = Table([[Paragraph(
        "본 문서는 Verilex 계약서 분석 기능 시연을 위해 작성된 <b>가상의 예시 계약서</b>입니다.<br/>"
        "실재하는 기업, 인물 또는 실제 체결된 계약과 무관하며 법적 효력이 없습니다.", notice)]],
        colWidths=[doc.width])
    box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#FDBA74")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF7ED")),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [box, Spacer(1, 6 * mm)]

    story += [Paragraph(
        "주식회사 가나다소프트(이하 \"회사\")와 이용자 홍길동(이하 \"이용자\")은 회사가 제공하는 "
        "온라인 문서관리 서비스의 이용에 관하여 다음과 같이 계약을 체결한다.", meta),
        Spacer(1, 5 * mm)]

    for num, (subject, text) in enumerate(CLAUSES, start=1):
        story.append(KeepTogether([
            Paragraph(f"제{num}조 ({subject})", head),
            Paragraph(text, body),
            Spacer(1, 4.2 * mm),
        ]))

    story += [Spacer(1, 6 * mm), Paragraph(
        "위 계약의 성립을 증명하기 위하여 계약서 2통을 작성하고 회사와 이용자가 각각 "
        "기명날인한 후 1통씩 보관한다.", meta), Spacer(1, 7 * mm)]

    story.append(Paragraph("2026년 9월 2일", meta))
    story.append(Spacer(1, 5 * mm))
    t = Table([["회사   주식회사 가나다소프트   대표이사 김가나   (인)"],
               ["이용자   홍  길  동   (인)"]], colWidths=[doc.width])
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), SERIF, 9.5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.append(t)

    doc.build(story)


if __name__ == "__main__":
    build(sys.argv[1])
    print("written:", sys.argv[1])
