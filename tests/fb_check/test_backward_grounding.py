# tests/fb_check/test_backward_grounding.py
"""check_snippet_exists(E ⊂ C) 회귀 테스트.

이 검사가 무르면 발췌가 아닌 문장이 학습 라벨로 들어오고, 빡빡하면 멀쩡한 라벨을 버린다.
실제로 빡빡한 쪽으로 틀려 있었다 — 라벨링 914건 시점의 `snippet_not_found` 66건 중
54건(81.8%)이 PDF 줄바꿈이 단어를 쪼갠 것(`영업정 지`)이었다. 양쪽을 다 고정한다.
"""

import pytest

from backend.fb_check.backward_grounding import check_snippet_exists

# --- 통과해야 하는 것: PDF 레이아웃 잡음뿐이고 문구는 실존한다 (실제 라벨링 데이터에서 발췌) ---
SHOULD_PASS = [
    pytest.param(
        "제20조(점포주에 대한 사항) 2. 乙이 전항 및 제9조와 제13조에 위반할 경우에 甲은 "
        "단전을 하거나 영업정 지 및 점포폐쇄 조치를 시킬 수 있다. 2)",
        "甲은 단전을 하거나 영업정지 및 점포폐쇄 조치를 시킬 수 있다",
        id="줄바꿈이 단어를 쪼갬(영업정 지)",
    ),
    pytest.param(
        "제26조(자동이체출금) 6. 본 자동이체(CMS)신청과 관련하여 분쟁이 발생하는 경우 "
        "회사의 귀책사유가 없 는 한 회원의 책임으로 간주합니다. (2)",
        "회사의 귀책사유가 없는 한 회원의 책임으로 간주합니다",
        id="줄바꿈이 조사를 쪼갬(없 는)",
    ),
    pytest.param(
        "제4조(업종유지) │ │ ① 갑은 을의 허락없이 을의 임의대 │ │ 로 업종을 변경할 때, "
        "자동 해약함과 아울러 기불입한 분양대금은 │ │ 갑 에게 귀속한다. │",
        "을의 임의대로 업종을 변경할 때, 자동 해약함과 아울러 기불입한 분양대금은 갑에게 귀속한다",
        id="표 괘선이 문장 사이에 낌",
    ),
    pytest.param(
        "제3조 위탁자는 수탁자의 승인 없이 계약을 해지할 수 없다. - 12 - 제4조 …",
        "위탁자는 수탁자의 승인 없이 계약을 해지할 수 없다",
        id="페이지 번호 마커",
    ),
]

# --- 거절해야 하는 것: 모델이 원문을 바꾸거나 이어붙였다 ---
SHOULD_FAIL = [
    pytest.param(
        "제47조 ① 가맹점사업자는 계약 및 가맹점 운영상 알게 된 가맹본부의 매뉴얼 기타 "
        "영업 비밀을 계약기간은 물론 계약종료 후에도 제3자에게 누설해서는 아니 된다.",
        "가맹점사업자는 계약기간은 물론 계약종료 후에도 제3자에게 누설해서는 아니 된다",
        id="중간을 건너뛰고 앞뒤를 이어붙임",
    ),
    pytest.param(
        "제20조(위임 및 동의) ② “을”은 건물전체의 공용부분의 용도결정 및 이용계획의 결정에 "
        "관 련한 모든 권한을 “갑”에게 위임한다",
        "건물전체의 공용부분의 용도결정 및 이용계획의 결정에 관한 모든 권한을 “갑”에게 위임한다",
        id="단어 변형(관련한→관한)",
    ),
    pytest.param("제5조 사업자는 언제든지 계약을 변경할 수 있다.", "완전히 다른 문장을 지어냈습니다", id="생성"),
    pytest.param("제5조 사업자는 언제든지 계약을 변경할 수 있다.", "짧다", id="10자 미만"),
    pytest.param("제5조 사업자는 언제든지 계약을 변경할 수 있다.", "", id="빈 문자열"),
]


@pytest.mark.parametrize("clause,span", SHOULD_PASS)
def test_레이아웃_잡음은_통과시킨다(clause: str, span: str) -> None:
    assert check_snippet_exists(clause, span) is True


@pytest.mark.parametrize("clause,span", SHOULD_FAIL)
def test_원문을_바꾼_발췌는_거절한다(clause: str, span: str) -> None:
    assert check_snippet_exists(clause, span) is False
