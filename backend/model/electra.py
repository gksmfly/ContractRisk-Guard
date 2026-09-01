# backend/model/electra.py
"""
KoELECTRA 분류 헤드 정의 및 공유 레이블 상수

헤드가 둘 있고 **둘 다 살아 있다** — 쓰임이 다르고, 전환이 아직 안 끝났다:

    ArticleMultiLabelElectra   조항 → 약관규제법 조 multi-label. **프로덕션 판단 경로**
                               (`backend.agents.judgment_agent`, 기본 `models/article_v1`)
    DualHeadElectra            domain 2-class + risk 3-class. 전량 라벨링 경로
                               (`backend.fb_check`)와 과거 실험 스크립트가 아직 쓴다

학습도 각각이다 — `backend.training.train_article`(조 multi-label)과
`backend.training.train`(domain+risk).
"""

from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel

from backend.utils import load_logger

logger = load_logger("electra.log")

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


# ─────────────────────────────────────────────────────────────────────────────
# 약관규제법 조 multi-label 헤드
# ─────────────────────────────────────────────────────────────────────────────
#
# `DualHeadElectra`(domain 2-class + risk 3-class, 각각 single-label)는 **고치지 않는다.**
# 지금도 7개 파일이 import하고 있고 그중 둘(`fb_check/__main__.py`, `fb_check/backward_grounding.py`)이
# 전량 라벨링 경로다 — 제자리에서 갈아엎으면 라벨 생성이 먼저 깨진다. 전환기 동안 둘 다
# 살아 있어야 하므로 새 클래스로 추가하고, 학습 진입점도 `train_article.py`로 따로 뒀다.
#
# **어떤 조를 헤드에 둘지는 데이터가 정한다.** FTC 1,163건 기준 제13조(대리인 책임 가중)는
# gold 0건, 제12조도 희소하다. 학습 표본이 없는 조에 출력 뉴런을 두면 그 조는 절대 학습되지
# 않으면서 macro 평균만 끌어내린다. `article_labels(counts, min_support)`가 support 기준으로
# 골라내고, 접힌 조는 예측 대상에서 제외된다.

_MIN_ARTICLE_SUPPORT = 5


def article_labels(gold_counts: dict[str, int], min_support: int = _MIN_ARTICLE_SUPPORT) -> list[str]:
    """support가 충분한 조만 헤드 라벨로 고른다(조 번호 순 정렬 — 인덱스 안정성)."""
    kept = [a for a, n in gold_counts.items() if n >= min_support]
    return sorted(kept, key=lambda a: int(a.strip("제조")))


class ArticleMultiLabelElectra(nn.Module):
    """조항 → 약관규제법 조 multi-label.

    `DualHeadElectra`와의 차이는 헤드 하나와 손실뿐이다:

      - single-label(softmax + CrossEntropy) → **multi-label(sigmoid + BCEWithLogits)**
        한 조항이 여러 조에 걸리는 게 정상이다(공정위 의결서 기준 케이스당 평균 2.08개).
      - 출력 차원이 고정 상수가 아니라 **데이터에서 정해진 조 목록** 길이다.

    `risk_level` 헤드는 **일부러 두지 않았다.** 조 예측에서 risk를 유도하는 규칙을 지금
    정하면 또 근거 없는 상수가 되고, risk의 gold 자체가 아직 미정이다
    (Phase A 프록시 vs clean.jsonl). 조 multi-label을 먼저 세우고, 조→risk 매핑은
    그 뒤에 데이터에서 측정한다. 두 축을 섞으면 어느 쪽이 문제인지 또 못 가린다.
    """

    def __init__(self, base_model_name: str, article_names: list[str]) -> None:
        super().__init__()
        if not article_names:
            raise ValueError("article_names가 비었다 — support 기준을 낮추거나 라벨을 확인할 것")
        self.encoder       = AutoModel.from_pretrained(base_model_name)
        self.article_names = list(article_names)
        hidden             = self.encoder.config.hidden_size
        self.article_head  = nn.Linear(hidden, len(self.article_names))
        self.dropout       = nn.Dropout(0.1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """조별 **로짓**을 반환한다(sigmoid 전). 손실은 BCEWithLogitsLoss가 받는다."""
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self.encoder.forward.__code__.co_varnames:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.article_head(cls)

    def _fingerprint(self) -> float:
        """인코더+헤드 **전 파라미터**의 norm 합. 랜덤 초기화면 값이 확 달라진다.

        **float64로, 기기와 무관하게 계산한다.** float32로 하면 GPU와 CPU의 감산 순서가
        달라 값이 흔들린다 — 실측(KoELECTRA-base 197키):

            float32 GPU  5325.7930      학습 중 저장 시점
            float32 CPU  5325.0845      평가 시 로드 시점   → 0.71 차이 (상대 1.3e-4)
            float64      5325.7934      양쪽 일치 (차이 4e-10)

        기기에 따라 흔들리는 지문은 쓸모가 없다. 실제로 float32로 넣었더니 **멀쩡한
        체크포인트가 반려됐다** — 인코더는 197키 전부 완벽히 복원됐는데도(공통 키 차이
        norm 0.000000) 지문만 어긋났다. 검사가 진짜 사고 대신 자기 잡음을 잡으면
        다음엔 검사를 끄게 된다.
        """
        return float(sum(p.detach().cpu().double().norm() for p in self.parameters()))

    def save(self, path: Path) -> None:
        self.encoder.save_pretrained(path)
        torch.save(
            {"article_head": self.article_head.state_dict(),
             "article_names": self.article_names,
             "fingerprint_f64": self._fingerprint()},
            path / "heads.pt",
        )

    @classmethod
    def load(cls, path: Path) -> "ArticleMultiLabelElectra":
        """`save`가 쓴 것을 그대로 되읽는다. **저장 형식을 아는 곳을 한 군데로 묶는다.**

        이게 없어서 평가 스크립트가 `model.safetensors`를 모델 전체에 `strict=False`로
        밀어넣었고, 인코더 키(`encoder.` 접두사 없음)도 헤드도 하나도 안 맞아 **랜덤 가중치로
        채점**했다. 건별 F1 5.8%가 나왔는데 같은 체크포인트의 학습 로그는 macro 0.73이었다.
        `strict=False`가 조용히 삼켜서 예외도 안 났다.

        ## 헤드만 막으면 반쪽이다

        헤드는 `strict=True`로 막힌다. 그런데 **인코더는 여전히 열려 있다** —
        `from_pretrained`는 가중치가 없거나 config가 어긋나면 경고만 찍고 랜덤 초기화로
        진행한다. 방금 당한 것과 정확히 같은 실패 모드다.

        그래서 저장 시점의 **전 파라미터 norm 합**을 함께 적고 로드 후 대조한다. 인코더든
        헤드든 하나라도 안 실리면 값이 크게 벌어져 즉시 걸린다. "판정에 안 쓰는 신호도
        남긴다"가 라벨링에서 모델 불일치를 잡아낸 것과 같은 발상이다.
        """
        heads = torch.load(path / "heads.pt", map_location="cpu", weights_only=False)
        model = cls(str(path), heads["article_names"])      # 인코더는 저장된 것에서 읽는다
        model.article_head.load_state_dict(heads["article_head"], strict=True)
        if model.article_head.out_features != len(heads["article_names"]):
            raise ValueError("헤드 차원과 article_names 길이가 다르다 — 체크포인트가 깨졌다")

        # 키 이름에 정밀도를 박아둔다 — float32 시절 지문은 기기마다 달라 비교 자체가 무의미하다.
        want = heads.get("fingerprint_f64")
        if want is None:
            logger.warning(f"  {path.name}: float64 지문이 없는 옛 체크포인트다 — 가중치 검사를 건너뛴다. "
                           f"다시 학습하면 찍힌다")
        else:
            got = model._fingerprint()
            if abs(got - want) > max(1e-6, abs(want) * 1e-9):   # float64 왕복 오차보다 훨씬 큰 값만 잡는다
                raise ValueError(
                    f"가중치가 저장 시점과 다르다 — 인코더나 헤드가 안 실렸다 "
                    f"(지문 {got:.6f} ≠ {want:.6f}). `strict=False`로 조용히 넘어가던 그 상황이다")
        return model

    @staticmethod
    def load_article_names(path: Path) -> list[str]:
        """체크포인트에서 조 목록을 복원한다 — 출력 차원과 조 이름의 대응이 깨지면 안 된다."""
        return torch.load(path / "heads.pt", map_location="cpu", weights_only=True)["article_names"]


def article_pos_weight(gold_counts: dict[str, int], names: list[str], n_samples: int) -> torch.Tensor:
    """조별 양성 가중치 = (음성 수 / 양성 수). 불균형이 심한 조가 무시되지 않게 한다.

    제6조는 정답의 36%에 붙는 사실상 다수 클래스이고 제7·12조는 한 자릿수다 —
    가중치 없이 BCE를 돌리면 희소한 조는 "항상 0"이 최적해가 된다.
    """
    w = []
    for a in names:
        pos = max(gold_counts.get(a, 0), 1)
        w.append((n_samples - pos) / pos)
    return torch.tensor(w, dtype=torch.float)
