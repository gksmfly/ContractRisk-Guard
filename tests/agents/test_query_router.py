# tests/agents/test_query_router.py
"""backend/agents/query_router.py 단위 테스트.

실제 EXAONE 모델(7.8B, GPU) 없이도 검증 가능한 범위만 다룬다: JSON 추출 순수
함수, 그리고 route_law_names()의 각 실패 경로(법령 목록 없음/모델 로딩 실패/
추론 실패)가 예외를 밖으로 던지지 않고 전부 None으로 우아하게 degrade하는지.
모델 추론 자체는 최소한의 가짜 모델/토크나이저로 흉내내 정상 경로(JSON 파싱→
law_names 필터링)까지 확인한다.
"""

import pytest
import torch

from backend.agents import query_router


class _FakeBatch(dict):
    """tokenizer(...)가 반환하는 BatchEncoding 흉내 — .to(device)만 있으면 된다."""

    def to(self, device):
        return self


class _FakeTokenizer:
    eos_token_id = 0

    def __init__(self, decode_return: str):
        self._decode_return = decode_return

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "PROMPT"

    def __call__(self, prompt, return_tensors="pt"):
        return _FakeBatch({"input_ids": torch.zeros((1, 3), dtype=torch.long)})

    def decode(self, ids, skip_special_tokens=True):
        return self._decode_return


class _FakeModel:
    def generate(self, **kwargs):
        return torch.zeros((1, 6), dtype=torch.long)


class TestExtractJson:
    def test_extracts_valid_json(self):
        assert query_router._extract_json('앞뒤 텍스트 {"laws": ["민법"]} 더 텍스트') == {"laws": ["민법"]}

    def test_no_json_returns_none(self):
        assert query_router._extract_json("그냥 텍스트, JSON 없음") is None

    def test_malformed_json_returns_none(self):
        assert query_router._extract_json('{"laws": [민법]}') is None  # 따옴표 없는 값 — 문법 오류


class TestRouteLawNames:
    @pytest.fixture(autouse=True)
    def _routing_on(self, monkeypatch):
        """이 클래스의 테스트는 라우팅 **로직**을 검증하므로 기능 스위치를 켠 상태로 고정한다.

        `EXAONE_ENABLED`는 `.env`로도 설정되는 개발용 스위치라(GPU 메모리 확보 목적),
        고정하지 않으면 개발자의 로컬 `.env` 값에 따라 테스트가 통과했다 실패했다 한다.
        스위치 자체의 동작은 아래 `TestEnabledSwitch`가 따로 검증한다.
        """
        monkeypatch.setenv("EXAONE_ENABLED", "1")

    def test_no_law_names_returns_none(self, monkeypatch):
        monkeypatch.setattr(query_router, "get_law_names", lambda: [])
        assert query_router.route_law_names("아무 조항") is None

    def test_model_load_failure_returns_none(self, monkeypatch):
        monkeypatch.setattr(query_router, "get_law_names", lambda: ["민법"])

        def _raise():
            raise RuntimeError("GPU 없음")

        monkeypatch.setattr(query_router, "_get_local_model", _raise)
        assert query_router.route_law_names("조항") is None

    def test_inference_failure_returns_none(self, monkeypatch):
        """model.generate()가 던지는 예외(예: CUDA OOM)도 로딩 실패와 동일하게
        None으로 흡수돼야 한다 — 분석 요청 전체가 500으로 죽으면 안 되는 경로."""
        monkeypatch.setattr(query_router, "get_law_names", lambda: ["민법"])

        class _BrokenModel:
            def generate(self, **kwargs):
                raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(
            query_router, "_get_local_model", lambda: (_BrokenModel(), _FakeTokenizer(""))
        )
        assert query_router.route_law_names("조항") is None

    def test_happy_path_filters_to_known_law_names(self, monkeypatch):
        law_names = ["민법", "상법", "약관의 규제에 관한 법률"]
        monkeypatch.setattr(query_router, "get_law_names", lambda: law_names)
        monkeypatch.setattr(
            query_router,
            "_get_local_model",
            lambda: (_FakeModel(), _FakeTokenizer('{"laws": ["민법", "존재하지않는법"]}')),
        )
        # 모델이 목록에 없는 법령("존재하지않는법")을 예측해도 실제 law_names로 필터링된다.
        assert query_router.route_law_names("조항") == ["민법"]

    def test_all_predictions_unknown_returns_none(self, monkeypatch):
        monkeypatch.setattr(query_router, "get_law_names", lambda: ["민법"])
        monkeypatch.setattr(
            query_router,
            "_get_local_model",
            lambda: (_FakeModel(), _FakeTokenizer('{"laws": ["존재하지않는법"]}')),
        )
        assert query_router.route_law_names("조항") is None

    def test_no_json_in_output_returns_none(self, monkeypatch):
        monkeypatch.setattr(query_router, "get_law_names", lambda: ["민법"])
        monkeypatch.setattr(
            query_router, "_get_local_model", lambda: (_FakeModel(), _FakeTokenizer("JSON 아님"))
        )
        assert query_router.route_law_names("조항") is None


class TestEnabledSwitch:
    """`EXAONE_ENABLED` 스위치 — 꺼져 있으면 모델을 **로드조차 하지 않아야** 한다.

    목적이 GPU 메모리(약 15.6GB) 확보이므로, None만 반환하고 로드는 하는 구현이면
    의미가 없다. 그래서 "로드 함수가 호출되지 않았는가"까지 확인한다.
    """

    def _tripwire(self, monkeypatch):
        called = []
        monkeypatch.setattr(query_router, "get_law_names", lambda: ["민법"])
        monkeypatch.setattr(
            query_router, "_get_local_model",
            lambda: called.append(1) or (_FakeModel(), _FakeTokenizer('{"laws": ["민법"]}')),
        )
        return called

    @pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", " 0 "])
    def test_disabled_values_skip_model_load(self, monkeypatch, value):
        monkeypatch.setenv("EXAONE_ENABLED", value)
        called = self._tripwire(monkeypatch)
        assert query_router.route_law_names("조항") is None
        assert called == [], "꺼진 상태인데 모델을 로드했다 — GPU 메모리 절약 효과가 사라진다"

    @pytest.mark.parametrize("value", ["1", "true", "yes", "anything"])
    def test_enabled_values_run_routing(self, monkeypatch, value):
        monkeypatch.setenv("EXAONE_ENABLED", value)
        self._tripwire(monkeypatch)
        assert query_router.route_law_names("조항") == ["민법"]

    def test_default_is_enabled(self, monkeypatch):
        # 환경변수가 아예 없으면 켜진 상태여야 한다 — 끄는 건 명시적 선택이어야 함
        monkeypatch.delenv("EXAONE_ENABLED", raising=False)
        self._tripwire(monkeypatch)
        assert query_router.route_law_names("조항") == ["민법"]
