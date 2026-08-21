# backend/agents/query_router.py
"""Query Router — 조항이 어느 법령에 해당할지 로컬 LLM으로 먼저 판단한다(Retrieval
Strategy Agent가 이 결과로 법령 검색 범위를 좁힌다).

배경: 법령 코퍼스는 43청크(약관의 규제에 관한 법률)~1,305청크(민법)로 극단적으로
불균형한데, 필터 없이 전체를 한 풀에서 RRF로 경쟁시키면 소수 법령의 정답 조문이
대형 법령들에 밀려 top_k 밖으로 나간다(`backend/eval/retrieval_alternatives_survey.md`
"핵심 발견" 참고). backend.eval.raptor_lite_compare.py로 같은 100건 FTC 근거_법령
ground truth에서 실측: 필터 없는 RRF 8% → 이 방식(EXAONE 라우팅 후 검색) 33%,
McNemar p<0.0001.

로컬 EXAONE-3.5-7.8B-Instruct(한국어 특화)를 쓴다 — 같은 실험에서 Qwen2.5-14B(2배
가까이 큰 모델)로 바꿔도 26%로 EXAONE보다 낮았다(모델 크기보다 한국어 특화가
중요했다는 근거). OpenAI API 비용은 0원.

주의: 재구성(쿼리 텍스트 자체를 다시 쓰는 LegalMALR-lite)은 일부러 안 붙인다 —
같은 실험에서 "재구성+라우팅+재랭킹을 한 EXAONE 호출에 다 시키면"(종합 콤보) 28%로
라우팅 단독(33%)보다 낮았다. 한 번에 여러 일을 시키면 각각의 품질이 떨어지는 것으로
보여, 이 모듈은 라우팅 하나만 한다.
"""

import json
import os
import re
from typing import Any

import torch

from backend.api.services.retrieval import get_law_names
from backend.utils import load_logger

logger = load_logger("query_router.log")

_MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
# 프로젝트 규칙상 GPU는 cuda:1 고정(`Claude.md`). 다른 GPU로 분리 배치해야 할 때만
# EXAONE_DEVICE로 덮어쓴다 — EXAONE은 bf16 기준 약 16GB라 배치 변경 시 여유를 확인할 것.
_DEVICE = os.environ.get("EXAONE_DEVICE", "cuda:1" if torch.cuda.is_available() else "cpu")

_SYSTEM_TMPL = (
    "너는 한국 법률 전문가다. 계약 조항을 읽고, 이 조항의 불공정성을 다툴 때 가장 근거로 "
    "삼을 가능성이 높은 법령을 아래 목록에서 정확히 2개 골라라.\n"
    "목록: {law_list}\n"
    '반드시 JSON만 출력: {{"laws": ["법령명1", "법령명2"]}}'
)
_FEWSHOT = [
    {"role": "user", "content": "계약 조항:\n제5조 을은 계약기간 중 언제든지 갑에게 서면 통지만으로 계약을 해지할 수 있으며, 갑은 이에 대해 이의를 제기할 수 없다."},
    {"role": "assistant", "content": '{"laws": ["약관의 규제에 관한 법률", "민법"]}'},
    {"role": "user", "content": "계약 조항:\n제9조 회원이 이용약관을 위반한 경우 회사는 손해배상과 별도로 위약금으로 계약금의 3배를 청구할 수 있다."},
    {"role": "assistant", "content": '{"laws": ["약관의 규제에 관한 법률", "민법"]}'},
]

_model = None
_tokenizer = None


def _extract_json(text: str) -> dict | None:
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _get_local_model() -> tuple[Any, Any]:
    """EXAONE을 지연 로드해 모듈 레벨에 캐싱한다 — 첫 호출(첫 요청)이 느리다는 뜻,
    서버 기동 시 워밍업하려면 server.py lifespan에서 이 함수를 한 번 호출하면 된다."""
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID, torch_dtype=torch.bfloat16, device_map=_DEVICE, trust_remote_code=True,
        )
        _model.eval()
    return _model, _tokenizer


def is_enabled() -> bool:
    """EXAONE 라우팅 사용 여부(`EXAONE_ENABLED`, 기본 켜짐).

    끄는 용도: EXAONE은 bfloat16으로 약 15.6GB를 GPU에 상주시킨다. 학습·평가 실험처럼
    라우팅이 필요 없는 작업을 같은 GPU에서 돌릴 때 이 메모리가 그대로 낭비된다.
    끄면 `route_law_names()`가 모델을 **로드조차 하지 않고** None을 반환하고, 호출자는
    이미 있는 폴백대로 필터 없이 검색한다(장애 시 경로와 동일).

    대가: 법령 조문 적중률이 33% → 8% 수준으로 떨어진다
    (`backend/eval/measurement_findings_2026-08-16.md`). 분석 품질이 필요한 상황에서는
    반드시 되돌릴 것.
    """
    return os.environ.get("EXAONE_ENABLED", "1").strip().lower() not in ("0", "false", "no")


def route_law_names(clause: str, max_new_tokens: int = 100) -> list[str] | None:
    """조항 텍스트로 관련 법령 top-2를 예측한다. 실패 시 None(호출자는 필터 없이 검색해야 함)."""
    if not is_enabled():
        return None

    law_names = get_law_names()
    if not law_names:
        return None

    try:
        model, tokenizer = _get_local_model()
    except Exception as e:
        logger.warning(f"EXAONE 모델 로딩 실패, 필터 없이 검색: {e}")
        return None

    messages = [{"role": "system", "content": _SYSTEM_TMPL.format(law_list=", ".join(law_names))}]
    messages.extend(_FEWSHOT)
    messages.append({"role": "user", "content": f"계약 조항:\n{clause[:800]}"})

    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(_DEVICE)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    except Exception as e:
        # GPU OOM 등 추론 자체가 실패해도 분석 요청 전체가 500으로 죽으면 안 된다 —
        # 라우팅은 검색 정확도를 높이는 보조 수단이지 필수 경로가 아니다(로딩 실패와
        # 동일하게 필터 없이 검색으로 흡수한다).
        logger.warning(f"EXAONE 추론 실패, 필터 없이 검색: {e}")
        return None

    result = _extract_json(text)
    if not result:
        return None

    predicted = [law for law in result.get("laws", []) if law in law_names]
    return predicted or None
