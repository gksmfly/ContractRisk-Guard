# backend/fb_check/oss_experiment/local_labeling.py
"""
로컬 오픈소스 LLM 기반 Forward Labeling / Consistency Verification

backend/fb_check/forward_labeling.py, consistency_verification.py가 쓰는 GPT-4o 호출을
로컬 GPU에서 도는 오픈소스 모델로 대체한다. 프롬프트(_SYSTEM/_FEW_SHOT_EXAMPLES)는
그대로 재사용해 GPT-4o 버전과 공정하게 비교되도록 한다.

backend/fb_check/llm_benchmark.py의 소규모 비교에서:
  - mode=full(원문 전체)  : 세 모델 다 domain 40~60%로 GPT-4o 대체 부적합
  - mode=evidence(근거만) : Qwen2.5-14B/Llama-3.1-8B는 domain 85~90%까지 개선
이 결과에 따라 Forward Labeling은 원문 전체를 다루므로 품질 리스크가 있고,
Consistency Verification은 evidence_span만 다루므로 상대적으로 안전하다 —
그래도 "전체 데이터로 실제 돌리면 어떻게 되는지"를 보는 대조군 실험이라
Forward/Verify 둘 다 로컬 모델로 대체해 전체 파이프라인을 실행한다.
"""

import json
import re
from typing import Any

import torch

from backend.fb_check.forward_labeling import _SYSTEM as FWD_SYSTEM, _FEW_SHOT_EXAMPLES as FWD_FEWSHOT
from backend.fb_check.consistency_verification import _SYSTEM as VERIFY_SYSTEM, _FEW_SHOT_EXAMPLES as VERIFY_FEWSHOT

MODELS = {
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def load_local_model(model_key: str, device: str) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _generate(
    model: Any, tokenizer: Any, device: str, system: str, fewshot: list[dict],
    user_content: str, max_new_tokens: int,
) -> dict | None:
    messages = [{"role": "system", "content": system}]
    messages.extend(fewshot)
    messages.append({"role": "user", "content": user_content})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _extract_json(text)


def run_forward_local(model: Any, tokenizer: Any, device: str, clause_text: str) -> dict | None:
    """forward_labeling.run_forward()의 로컬 모델 버전 (C → L + evidence_span)."""
    content = f"계약 조항:\n{clause_text[:3000]}"
    return _generate(model, tokenizer, device, FWD_SYSTEM, FWD_FEWSHOT, content, max_new_tokens=300)


def run_verify_local(model: Any, tokenizer: Any, device: str, evidence_span: str) -> dict | None:
    """consistency_verification.run_verify()의 로컬 모델 버전 (E → L')."""
    content = f"근거 문구:\n{evidence_span}"
    return _generate(model, tokenizer, device, VERIFY_SYSTEM, VERIFY_FEWSHOT, content, max_new_tokens=100)
