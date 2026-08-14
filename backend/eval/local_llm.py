# backend/eval/local_llm.py
"""
OpenAI API 대신 로컬 GPU에서 도는 오픈 모델로 짧은 분류·재구성 작업을 처리하는 공용
헬퍼. backend/fb_check/oss_experiment/local_labeling.py의 패턴(모델 로드·generate)을
재사용한다 — 단 그 실험은 "GPT-4o를 로컬 모델로 완전히 대체할 수 있는가"(원문 전체
라벨링, 결과: domain 40~60%로 부적합)를 테스트한 것이고, 여기서는 훨씬 짧고 단순한
태스크(쿼리 재구성/분류)라 그 결과가 그대로 적용되진 않는다 — 그 실험에서도 "근거만"
(evidence-only, 짧은 텍스트) 조건에서는 85~90%까지 개선됐던 것과 같은 종류의 태스크
(짧은 입력)라는 점에서 참고할 만하다.

기본 모델은 EXAONE-3.5-7.8B-Instruct(한국어 특화)지만, model_key로 다른 캐시된 모델
(Qwen2.5-14B-Instruct 등)로 바꿔 같은 태스크에 대한 모델 간 비교도 할 수 있다.

실행 비용: OpenAI API 토큰 0원 — GPU 로컬 추론만 사용.
"""

import json
import re
from typing import Any

import torch

MODELS = {
    "exaone-3.5-7.8b": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
}
_DEFAULT_MODEL_KEY = "exaone-3.5-7.8b"
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_models: dict[str, Any] = {}
_tokenizers: dict[str, Any] = {}
_device = "cuda:0" if torch.cuda.is_available() else "cpu"


def _extract_json(text: str) -> dict | None:
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def get_local_model(model_key: str = _DEFAULT_MODEL_KEY) -> tuple[Any, Any]:
    if model_key not in _models:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = MODELS[model_key]
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=_device, trust_remote_code=True,
        )
        model.eval()
        _models[model_key] = model
        _tokenizers[model_key] = tokenizer
    return _models[model_key], _tokenizers[model_key]


def generate_json(
    system: str, user_content: str, max_new_tokens: int = 200, fewshot: list[dict] | None = None,
    model_key: str = _DEFAULT_MODEL_KEY,
) -> dict | None:
    """fewshot: [{"role": "user"/"assistant", "content": ...}, ...] — user/assistant 쌍을
    system과 실제 질의 사이에 끼워 넣는다(local_labeling.py와 동일 패턴)."""
    model, tokenizer = get_local_model(model_key)
    messages = [{"role": "system", "content": system}]
    if fewshot:
        messages.extend(fewshot)
    messages.append({"role": "user", "content": user_content})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(_device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _extract_json(text)
