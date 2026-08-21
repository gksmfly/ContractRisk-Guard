# backend/model/electra.py
"""
DualHeadElectra 모델 정의 및 공유 레이블 상수

train.py (학습) 과 fb_check.py (추론) 양쪽에서 공유한다.
"""

from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel

DOMAIN_MAP     = {"해지_조항": 0, "책임제한_조항": 1}
INV_DOMAIN_MAP = {v: k for k, v in DOMAIN_MAP.items()}
DOMAIN_NAMES   = ["해지_조항", "책임제한_조항"]

# risk 라벨 체계 — 3class가 기본이고, 2class는 Medium을 제외한 재정의다.
#
# Medium을 뺀 근거(2026-08-16 실측, `backend/eval/measurement_findings_2026-08-16.md`):
#   1. 라벨 소스 간 편차 — 같은 clean.jsonl 694건에 Medium이 seed 60 / forward 159 /
#      verify 116 / final 86건으로 2.6배 차이. 어느 판단 주체도 일관되게 못 찍는다
#   2. 데이터 확장 6회 실패 — v5~v9 전부 v4를 못 넘음
#   3. 모델 조합 31가지 실패 — 어떤 앙상블을 써도 Medium 예측 정밀도가 12~13%에 고정
#      (Medium이라고 하면 87%가 틀림). 앙상블은 Medium을 더 많이 뱉게 할 뿐이었다
# → 학습 실패가 아니라 라벨이 학습 가능한 신호를 담고 있지 않다는 결론.
RISK_SCHEMES = {
    "3class": ["High", "Medium", "Low"],
    "2class": ["High", "Low"],
}
_DEFAULT_RISK_SCHEME = "3class"

RISK_NAMES     = RISK_SCHEMES[_DEFAULT_RISK_SCHEME]
RISK_MAP       = {name: i for i, name in enumerate(RISK_NAMES)}
INV_RISK_MAP   = {v: k for k, v in RISK_MAP.items()}


def risk_scheme(name: str = _DEFAULT_RISK_SCHEME) -> tuple[dict[str, int], dict[int, str], list[str]]:
    """(RISK_MAP, INV_RISK_MAP, RISK_NAMES)를 체계별로 만들어 준다.

    모듈 전역 상수는 3class로 고정돼 있다(judgment_agent·backward_grounding 등 기존
    호출부가 그대로 동작해야 하므로). 2class 학습·평가는 이 함수로 받아서 쓴다.
    """
    if name not in RISK_SCHEMES:
        raise ValueError(f"알 수 없는 risk 체계: {name} (가능: {list(RISK_SCHEMES)})")
    names = RISK_SCHEMES[name]
    mapping = {n: i for i, n in enumerate(names)}
    return mapping, {v: k for k, v in mapping.items()}, names


class DualHeadElectra(nn.Module):
    """domain + risk_level 동시 분류를 위한 듀얼 헤드 모델.

    num_risk_labels를 안 주면 3class(=len(RISK_MAP))로 만든다 — 2class 체크포인트를
    로드할 때는 반드시 2를 넘겨야 heads.pt의 형상과 맞는다.
    """

    def __init__(self, base_model_name: str, num_risk_labels: int | None = None) -> None:
        super().__init__()
        # AutoModel을 쓰는 이유: 백본을 KoELECTRA 외의 계열(KLUE-RoBERTa 등)로 바꿔
        # 비교할 수 있어야 한다. `ElectraModel.from_pretrained`로 고정돼 있으면
        # ELECTRA 계열만 실을 수 있어 백본 축 자체를 측정할 수 없다.
        # ELECTRA 체크포인트는 AutoModel로 읽어도 동일한 ElectraModel이 나온다.
        self.encoder     = AutoModel.from_pretrained(base_model_name)
        hidden           = self.encoder.config.hidden_size
        self.domain_head = nn.Linear(hidden, len(DOMAIN_MAP))
        self.risk_head   = nn.Linear(hidden, num_risk_labels or len(RISK_MAP))
        self.dropout     = nn.Dropout(0.1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # RoBERTa 계열은 token_type_ids를 안 받는다(forward 시그니처에 없음) —
        # 백본마다 받는 인자가 달라서 지원 여부를 보고 넘긴다.
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self.encoder.forward.__code__.co_varnames:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.domain_head(cls), self.risk_head(cls)

    def save(self, path: Path) -> None:
        self.encoder.save_pretrained(path)
        torch.save(
            {"domain_head": self.domain_head.state_dict(), "risk_head": self.risk_head.state_dict()},
            path / "heads.pt",
        )
