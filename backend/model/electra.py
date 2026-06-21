# backend/model/electra.py
"""
DualHeadElectra 모델 정의 및 공유 레이블 상수

train.py (학습) 과 fb_check.py (추론) 양쪽에서 공유한다.
"""

from pathlib import Path

import torch
from torch import nn
from transformers import ElectraModel

DOMAIN_MAP     = {"해지_조항": 0, "책임제한_조항": 1}
RISK_MAP       = {"High": 0, "Medium": 1, "Low": 2}
INV_DOMAIN_MAP = {v: k for k, v in DOMAIN_MAP.items()}
INV_RISK_MAP   = {v: k for k, v in RISK_MAP.items()}
DOMAIN_NAMES   = ["해지_조항", "책임제한_조항"]
RISK_NAMES     = ["High", "Medium", "Low"]


class DualHeadElectra(nn.Module):
    """domain + risk_level 동시 분류를 위한 듀얼 헤드 모델."""

    def __init__(self, base_model_name: str) -> None:
        super().__init__()
        self.encoder     = ElectraModel.from_pretrained(base_model_name)
        hidden           = self.encoder.config.hidden_size
        self.domain_head = nn.Linear(hidden, len(DOMAIN_MAP))
        self.risk_head   = nn.Linear(hidden, len(RISK_MAP))
        self.dropout     = nn.Dropout(0.1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.domain_head(cls), self.risk_head(cls)

    def save(self, path: Path) -> None:
        self.encoder.save_pretrained(path)
        torch.save(
            {"domain_head": self.domain_head.state_dict(), "risk_head": self.risk_head.state_dict()},
            path / "heads.pt",
        )
