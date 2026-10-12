# backend/fb_check/local_llm.py
"""
FB-Check 단계를 OpenAI 대신 로컬 오픈 모델로 돌리는 백엔드 (2026-10-12)

`VERIFY_MODEL=local:qwen2.5-14b`처럼 `local:` 접두사를 주면 `run_verify`가 여기로 온다.
API 토큰 0원 — GPU 로컬 추론만 쓴다.

## 왜 Qwen2.5-14B(4비트)인가

- 같은 프롬프트·판정 규칙으로 GPT forward 위에 verify만 바꿔 본 실측(`eval/reported_numbers.md` §11 E_xv):
  Qwen verify는 GPT CLEAN 안에서도 문서 층화 +2.0%p(p<0.001)를 더 걸러냈다. Llama-3.1-8B는 미판정
- Apache 2.0이라 상업 사용 가능(EXAONE은 NC)

## 답변을 `{`로 시작시키는 이유

OpenAI 호출은 `response_format=json_object`로 JSON만 받는다. 로컬 모델에는 그 장치가 없어서,
Qwen이 앞뒤가 잘린 조각 조항에 JSON 대신 줄글로 답했다(AI Hub 첫 56건 중 26.8% 파싱 실패).
답변 첫 글자를 `{`로 고정하자 0.4%로 줄었다(`eval/paper_plan.md` §7-14).
"""

import json
import os
import re
from typing import Any

MODELS = {
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
}
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_cache: dict[str, tuple[Any, Any, str]] = {}


def is_local(model: str) -> bool:
    return isinstance(model, str) and model.startswith("local:")


def _load(key: str) -> tuple[Any, Any, str]:
    if key not in _cache:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if key not in MODELS:
            raise ValueError(f"알 수 없는 로컬 모델: {key} (가능: {sorted(MODELS)})")
        # 프로젝트 규칙: 기본 cuda:1, GPU 1이 다른 사용자로 차 있으면 LOCAL_LLM_DEVICE=cuda:0
        device = os.environ.get("LOCAL_LLM_DEVICE", "cuda:1" if torch.cuda.is_available() else "cpu")
        # 디스크가 모자라 모델을 RAM 디스크(/dev/shm)에 받아 둔 경우 그 캐시를 가리킨다(2026-10-12).
        # HF_HOME을 통째로 바꾸면 KoELECTRA 등 다른 캐시를 못 찾으므로 이 모델에만 적용한다
        cache = os.environ.get("LOCAL_LLM_CACHE") or None
        tok = AutoTokenizer.from_pretrained(MODELS[key], trust_remote_code=True, cache_dir=cache)
        model = AutoModelForCausalLM.from_pretrained(
            MODELS[key], device_map={"": device}, trust_remote_code=True, cache_dir=cache,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
        )
        model.eval()
        _cache[key] = (model, tok, device)
    return _cache[key]


def chat_json(model: str, messages: list[dict], max_new_tokens: int = 200) -> dict | None:
    """`local:<키>` 모델로 JSON 하나를 생성한다. 파싱 실패면 None."""
    import torch

    m, tok, device = _load(model.removeprefix("local:"))
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "{"
    enc = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    text = "{" + tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    hit = _JSON_RE.search(text)
    if not hit:
        return None
    try:
        return json.loads(hit.group(0))
    except json.JSONDecodeError:
        return None
