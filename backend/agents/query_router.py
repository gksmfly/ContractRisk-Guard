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
import re
from typing import Any

import torch

from backend.api.services.retrieval import get_law_names

_MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

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


def route_law_names(clause: str, max_new_tokens: int = 100) -> list[str] | None:
    """조항 텍스트로 관련 법령 top-2를 예측한다. 실패 시 None(호출자는 필터 없이 검색해야 함)."""
    law_names = get_law_names()
    if not law_names:
        return None

    try:
        model, tokenizer = _get_local_model()
    except Exception:
        return None

    messages = [{"role": "system", "content": _SYSTEM_TMPL.format(law_list=", ".join(law_names))}]
    messages.extend(_FEWSHOT)
    messages.append({"role": "user", "content": f"계약 조항:\n{clause[:800]}"})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    result = _extract_json(text)
    if not result:
        return None

    predicted = [law for law in result.get("laws", []) if law in law_names]
    return predicted or None
