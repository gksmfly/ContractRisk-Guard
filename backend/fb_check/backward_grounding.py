# backend/fb_check/backward_grounding.py
"""
Backward Grounding: evidence_span ⊂ C 인덱스 검증

**검증하는 것은 하나뿐이다.**
  - `check_snippet_exists`: GPT가 추출한 evidence_span이 원문에 실제로 존재하는지 확인.
    완전일치 → 레이아웃 제거 → 퍼지(0.85) 순. **순수 문자열 검사이고 모델을 쓰지 않는다.**

`predict`(KoELECTRA domain·risk_level)는 **판정에 쓰지 않는다.** `--record-backward`를
줬을 때 레코드에 기록만 하고, CLEAN/NOISE 결정은 `L == L'`(논문 정의)만 본다.

## 왜 모델이 투표하지 않는가 (Data Flywheel 철회)

원래 설계는 "CLEAN 데이터로 학습한 2세대 모델이 Backward Grounding 에이전트를 대체한다"
였고, 한때 3-way 다수결(forward/verify/backward 중 2개 일치)로 판정했다. **파이프라인의
산출물이 그 파이프라인의 검증자가 될 수 없다** — 2세대 모델은 자신이 검출해야 할 오류를
그대로 물려받는다. 논문이 정의한 2-way(`L == L'`)로 되돌렸다.
"""

import difflib
import re
from pathlib import Path

import torch
from transformers import ElectraTokenizerFast

from backend.model.electra import INV_DOMAIN_MAP, INV_RISK_MAP, DualHeadElectra

_PAGE_MARKER = re.compile(r'\s*-\s*\d+\s*-\s*')
_FUZZY_MATCH_THRESHOLD = 0.85  # 완전 일치 실패 시, 최장 공통 부분열이 근거 문구의 이 비율 이상이면 인정
# 매칭 규칙이 바뀌면 **반드시 올린다.** 레코드에 함께 저장돼, 규칙이 바뀐 뒤에만
# 재처리되도록 하는 근거가 된다(`__main__._load_checkpoint`의 redo 분기).
# forward_model / forward_prompt 가드와 같은 패턴이다 — "판정에 쓰지 않는 신호도
# 레코드에 남긴다"는 원칙의 연장이고, 그 원칙이 08-23에 모델 불일치를 잡아냈다.
#
#   snip-v1  완전일치 → 퍼지(공백 압축만)          ← PDF 줄바꿈이 쪼갠 단어를 전부 놓쳤다
#   snip-v2  완전일치 → 레이아웃 제거 → 퍼지@0.85  ← E⊂C 실패율 12.5% → 2.3%
MATCHER_VERSION = "snip-v2-striplayout-fuzzy085"

_BOX_DRAWING = re.compile(r'[\u2500-\u257f]')   # 의결서 PDF의 표 괘선(│ ─ ┌ …)
_WHITESPACE = re.compile(r'\s+')


def load_model(model_dir: Path, device: torch.device) -> tuple[DualHeadElectra, ElectraTokenizerFast]:
    model = DualHeadElectra(str(model_dir))
    heads = torch.load(model_dir / "heads.pt", map_location=device, weights_only=True)
    model.domain_head.load_state_dict(heads["domain_head"])
    model.risk_head.load_state_dict(heads["risk_head"])
    model.to(device).eval()
    tokenizer = ElectraTokenizerFast.from_pretrained(str(model_dir))
    return model, tokenizer


def _strip_layout(text: str) -> str:
    """페이지 번호·표 괘선·**모든 공백**을 없앤 비교용 문자열.

    공백을 압축(`" ".join(split())`)하는 것으로는 부족하다. 의결서 PDF는 줄바꿈 지점에서
    **단어 중간에 공백을 끼워 넣는다** — `영업정 지`, `비회원과 의 교제`, `을의 부 담`.
    GPT는 이걸 정상 표기로 되돌려 인용하므로 압축만 해서는 영영 안 맞는다.

    실측(라벨링 914건 시점, `snippet_not_found` 66건 전량 분해):

        공백/괘선을 제거하면 완전일치          54건  81.8%   ← 매칭 버그였다
        모델이 '...'로 생략                     5건   7.6%
        진짜 불일치(중간 건너뛰기·단어 변형)     6건   9.1%   ← 거절이 맞다
        span 없음/10자 미만                     1건   1.5%

    공백 제거가 무르지 않은가? 근거 문구는 10자 이상이라 한국어에서 공백만 지운 10자
    이상 문자열이 우연히 일치할 확률은 사실상 0이다. 실제로 위 6건은 그대로 걸러진다
    — `관련한`→`관한`(단어 변형), 문장 중간을 건너뛰고 앞뒤를 이어붙인 경우.
    """
    return _WHITESPACE.sub("", _BOX_DRAWING.sub("", _PAGE_MARKER.sub(" ", text)))


def check_snippet_exists(clause_text: str, evidence_span: str) -> bool:
    """evidence_span이 clause_text 안에 있는지 확인한다.

    완전 일치 → 레이아웃 제거 후 완전 일치 → 퍼지 매칭 순으로 확인한다. PDF에서 텍스트를
    추출할 때 표 셀이 뒤섞이거나 줄바꿈이 단어를 쪼개는 경우가 있는데, 이때는 근거 문구
    자체는 실존해도 완전 일치가 깨진다. 레이아웃 제거는 `_strip_layout` 참조.
    """
    if not evidence_span or len(evidence_span) < 10:
        return False
    norm_text = " ".join(_PAGE_MARKER.sub(" ", clause_text).split())
    norm_span = " ".join(evidence_span.split())

    if norm_span in norm_text:
        return True

    bare_text, bare_span = _strip_layout(clause_text), _strip_layout(evidence_span)
    if bare_span and bare_span in bare_text:
        return True

    matcher = difflib.SequenceMatcher(None, bare_span, bare_text, autojunk=False)
    match = matcher.find_longest_match(0, len(bare_span), 0, len(bare_text))
    return (match.size / len(bare_span)) >= _FUZZY_MATCH_THRESHOLD if bare_span else False


def predict(
    text: str, model: DualHeadElectra, tokenizer: ElectraTokenizerFast, device: torch.device,
) -> tuple[str, str]:
    enc = tokenizer(text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        d_logits, r_logits = model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
            enc.get("token_type_ids", torch.zeros(1, 256, dtype=torch.long)).to(device),
        )
    return INV_DOMAIN_MAP[d_logits.argmax().item()], INV_RISK_MAP[r_logits.argmax().item()]
