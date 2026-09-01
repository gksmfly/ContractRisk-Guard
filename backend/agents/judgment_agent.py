# backend/agents/judgment_agent.py
"""Judgment Agent — 파인튜닝된 KoELECTRA로 **위반 소지 조**를 판단한다.

GPT-4o(Analysis Agent)는 조항 유형 1차 분석·근거 문구 추출·설명 생성에만 쓰고,
최종 판단은 이 모듈(분류 모델)이 맡는다.

## 위험도 3단계를 더 이상 내지 않는다 (2026-08-31)

이전에는 `models/v4`(dual-head)가 domain 2종 + risk_level 3단계를 냈고, 화면이
High/Medium/Low를 표시했다. 두 가지가 그 설계를 무너뜨렸다:

1. **risk_level에 외부 준거가 없다.** 조 multi-label로 taxonomy를 바꾸면서 risk 헤드를
   일부러 뺐다 — gold를 정의할 방법이 없었다. 검증되지 않은 3단계를 화면에 띄우는 것은
   `out_of_scope`에서 막으려던 것과 같은 종류의 거짓 확신이다.
2. **`confidence_band`의 실측 정확도(57.5% / 46.8% / 38.2%)는 v4 전용이다.**
   n=453, 옛 라벨 정의 위에서 잰 값이라 새 모델로 옮길 수 없다. 옮기면 숫자만 남고
   근거는 사라진다.

## 지금 내는 것과, 그것이 서는 근거

    출력      위반 소지 조 목록(빈 리스트 가능)
    주장      "이 조항을 확인해 보세요 — 제N조 관련으로 보입니다"
    근거      조항 단위 재현 78.0%  (배포 임계값, clean gold n=255, `article_gold_eval`)
              ← r(계약서 내 위반 비율)과 무관한 모델 속성이라 r 없이도 보고할 수 있다

**오경보율은 아직 없다.** 예전에 여기 "오경보 2.6%"라고 적혀 있었는데, 그 값은 음성 풀의
정답이 GPT 라벨(forward ∩ verify) 그 자체여서 순환이었다 — 모델이 GPT보다 잘 찾아 짚은
것도 오경보로 세어진다. 이름을 `disagree_with_gpt`로 바꿨고, 독립 준거로 잰 오경보율은
표준계약서 조항을 사람이 판단해야 나온다.

조 단위 정밀도는 그보다 낮다(44%대). 그래서 **조 이름은 단정하지 않고 참고로 붙인다** —
"제9조 위반입니다"는 44% 위에 서지만 "이 조항을 확인해 보세요"는 78% 위에 선다.
같은 모델·같은 임계값인데 무엇을 주장하느냐가 어느 지표로 채점되는지를 정한다.

임계값은 학습 때 dev에서 확정한 조별 값(`thresholds.npy`)을 그대로 쓴다. 배포 유병률 r로
재최적화하면 최대 10% 비용 개선이 있으나, r을 신뢰성 있게 측정할 방법이 없어 하지 않는다
(`backend/eval/threshold_r_sweep.py`, `prevalence_worksheet.py` 참고).
"""

import os
from pathlib import Path
from typing import Any

import torch
from transformers import ElectraTokenizerFast

from backend.agents.state import ClauseState
from backend.model.electra import ArticleMultiLabelElectra
from backend.utils import PROJECT_ROOT

MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(PROJECT_ROOT / "models/article_v2")))


def model_version() -> str:
    """판단에 쓰인 체크포인트 이름. 저장된 분석 결과에 함께 남긴다.

    기본값은 `models/article_v2`(조 multi-label, max_len 512)다. 옛 `models/v4`(domain
    2종 + risk 3단계)는 **서빙에서 빠졌다** — 누수 있는 분할이었고, 학습 라벨의 16.4%를
    정확도 45%짜리 이전 세대가 결정했다.

    **`article_v1`(max_len 256)에서 이름을 올린 이유가 이 필드다.** 두 세대는 taxonomy가
    같아서 결과 모양이 구분되지 않는다 — 이름을 안 바꿨으면 저장된 판정이 256에서 잘린
    입력으로 난 것인지 아닌지 나중에 알 수 없다.

    **이 필드가 있어야 taxonomy가 다른 두 세대의 저장분을 나중에 구분할 수 있다.**
    v4 시절 결과에는 `risk_level`·`confidence_band`가 있고 `articles`가 없다 —
    `analyses.result`가 JSONB라 마이그레이션 없이 함께 저장되므로
    `WHERE result->>'model_version' = 'v4'`로 옛 판정을 골라낼 수 있다.
    """
    return MODEL_DIR.name
# 프로젝트 규칙상 GPU는 cuda:1 고정(`Claude.md`). 이전에는 `torch.device("cuda")`라
# 인덱스 없이 잡아 항상 cuda:0으로 갔다 — EXAONE(16GB)과 같은 GPU에 몰리는 원인이었다.
JUDGMENT_DEVICE = os.environ.get("JUDGMENT_DEVICE", "cuda:1")

_electra_model: ArticleMultiLabelElectra | None = None
_thresholds: Any = None
_electra_tokenizer: ElectraTokenizerFast | None = None
_electra_device: Any = None
_max_len: int = 256


def _get_electra() -> tuple[ArticleMultiLabelElectra, ElectraTokenizerFast, Any]:
    """`ArticleMultiLabelElectra.load()`로 읽는다 — 직접 조립하면 지문 검사를 건너뛴다.

    임계값은 학습 때 dev에서 확정한 조별 값을 그대로 쓴다(`thresholds.npy`).
    **여기서 다시 고르지 않는다** — 그게 곧 평가셋 오염이다.
    """
    global _electra_model, _electra_tokenizer, _electra_device, _thresholds, _max_len
    if _electra_model is None:
        import json

        import numpy as np
        _electra_device = torch.device(JUDGMENT_DEVICE if torch.cuda.is_available() else "cpu")
        _electra_model = ArticleMultiLabelElectra.load(MODEL_DIR).to(_electra_device).eval()
        _electra_tokenizer = ElectraTokenizerFast.from_pretrained(str(MODEL_DIR))
        _thresholds = np.load(MODEL_DIR / "thresholds.npy")
        # **max_len을 체크포인트에서 읽는다 — 여기 상수로 박지 않는다.**
        # 예전에 256이 박혀 있었다. 학습이 512로 가도 서빙만 256에서 잘랐을 것이고,
        # evidence_span 때와 **똑같은 형태의 사고**가 된다(모델은 바뀌는데 입력만 남는다).
        _max_len = int((json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
                        .get("train_config") or {}).get("max_len", 256))
    return _electra_model, _electra_tokenizer, _electra_device


def electra_predict(text: str) -> list[str]:
    """조항이 위반할 가능성이 있는 **약관규제법 조 목록**을 반환한다. 빈 리스트면 미지목.

    ## 위험도 3단계를 더 이상 내지 않는다 (2026-08-31)

    조 multi-label 모델에는 risk 헤드가 없다 — gold가 정의되지 않아 일부러 뺐다.
    `models/v4`의 3단계 위험도는 외부 준거로 검증된 적이 없고, 그 위에 얹혀 있던
    `confidence_band`의 실측 정확도(57.5% / 46.8% / 38.2%)도 v4 전용이라 여기 옮길 수 없다.
    **없는 근거를 화면에 표시하지 않는다.**

    대신 제품이 주장하는 것은 **조항 지목**이다. 그 주장이 서는 지표는 측정돼 있다
    (배포 임계값, `article_gold_eval`):

        조항 단위 재현 78.0%                   ← r(계약서 내 위반 비율)과 무관한 모델 속성
        조 단위 정밀도는 그보다 낮으므로(44%대) **조 이름은 참고로만 붙인다**

    화면 문구가 "이 조항을 확인해 보세요 — 제N조 관련으로 보입니다"인 이유가 이것이다.

    ## 입력은 **조항 원문**이다 — evidence_span을 넣지 말 것 (2026-09-01)

    2026-08-31까지 `judgment_node`가 `evidence_span or clause`를 넣었다. `models/v4`는
    span 증강으로 학습했으니 그때는 맞았지만, `article_v1`은 원문 전용으로 학습했고
    (`train_article` 규칙 6) 채점도 원문으로 했다. **모델과 평가만 바뀌고 운영 입력이
    남아 있었다** — 78.0%가 운영이 받는 입력을 설명하지 않았다.

    측정(`backend/eval/input_parity_eval.py`, span 보유 136건 페어드, 같은 조항 두 입력):

        조항 재현   원문 81.6%  →  조각 72.1%   -9.6%p [-16.9,-2.9]  유의하게 나빠짐
        조 F1       원문 41.8%  →  조각 37.0%    -4.8%p [-11.0,+1.2]  미판정

    게다가 **입력이 정답과 상관한다.** span은 GPT가 위반이라 본 부분이라 위반 조항의
    53%에만 있고 비위반 조항에는 2%뿐이다 — 양성은 조각으로, 음성은 원문으로 들어가서
    재현은 78.0→72.9%로 내려가고 GPT불일치는 2.6→4.0%로 올라간다. **두 축이 같은 방향으로
    나빠지는 게 아니라 서로 반대로 어긋난다.**

    ## 길이도 체크포인트를 따른다 — 256을 상수로 박지 말 것 (2026-09-01)

    `max_len`은 학습 하이퍼파라미터인데 **운영에서는 조용한 입력 절단**으로 작동했다.
    실제 약관은 한 조에 항이 여럿 붙어 학습 텍스트보다 2.4배 길고, 256토큰에서
    **35.4%가 뒷부분을 잃었다**(gold는 5.5%뿐이라 78.0%에는 안 보였다).

    512로 재학습해 절단을 없앴고, 판정은 **사전 등록한 무해 확인**으로 했다
    (`backend/eval/maxlen_harm_check.py`):

        gold 조항 재현  78.0% → 78.4%   +0.4%p [-3.9,+4.7]  ← 주 판정. 해가 없다
        조 F1          38.6% → 38.5%    -0.1%p             ← 참고
        실제 약관 절단   35.4% →  8.1%                      ← 개입이 표방한 일

    근거는 "512가 더 좋다"가 **아니다** — 그건 gold로 고르는 것이라 금지다. 근거는
    **사용자가 넣은 입력을 버리지 않는다**이고, gold는 그게 해롭지 않은지만 확인했다.
    """
    model, tokenizer, device = _get_electra()
    enc = tokenizer(text, max_length=_max_len, padding="max_length", truncation=True,
                    return_tensors="pt")
    with torch.no_grad():
        logits = model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
            enc.get("token_type_ids", torch.zeros(1, _max_len, dtype=torch.long)).to(device),
        )
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    return [n for n, p, t in zip(model.article_names, probs, _thresholds) if p >= t]


def judgment_node(state: ClauseState) -> dict:
    # **조항 원문을 넣는다. evidence_span으로 되돌리지 말 것** (2026-09-01, 실측 근거는
    # `electra_predict` 참고). 옛 주석은 "models/v4가 span 길이로 학습됐으므로 span을
    # 넣어야 분포가 맞는다"였는데, v4는 서빙에서 빠졌고 `article_v1`은 원문 전용으로
    # 학습됐다(`train_article` 규칙 6, augment=False). 모델만 바뀌고 입력만 남아 있었다.
    articles = electra_predict(state["clause"])
    return {
        "model_articles": articles,
        # 조항을 지목했는가 = 사용자에게 보여줄 것인가. 이진이다.
        "needs_review": bool(articles),
        # GPT와 모델이 **같은 조**를 짚었는지. 신뢰도가 아니라 두 판단의 일치 여부다.
        # 옛 구현은 `electra_domain == state["domain"]`으로 2-도메인 값을 비교했는데,
        # taxonomy가 조 단위로 바뀐 지금은 조 교집합이 맞는 비교다.
        "verified": bool(set(articles) & set(state.get("articles") or [])),
    }
