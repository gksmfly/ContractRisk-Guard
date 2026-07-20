# backend/fb_check/llm_benchmark.py
"""
FB-Check — 오픈소스 LLM 대체 가능성 비교

전체 1,835건 FB-Check 재실행은 GPT-4o API 비용이 드는데, 로컬 GPU로 돌릴 수 있는
오픈소스 모델로 대체 가능한지 소규모로 먼저 검증한다. data/fb_check/fb_check_results.jsonl
(이미 GPT-4o가 처리한 50건)을 기준값으로 삼아 후보 모델들의 일치율을 비교한다.

두 가지 모드 (--mode):
  full     - forward_labeling.py와 동일한 프롬프트로 원문 전체(text, 최대 3000자)를 보고
             domain/risk_level/evidence_span을 판단. forward_domain/forward_label과 비교.
             (원본 FTC 판례 발췌문은 각주·법령 인용이 섞인 긴 텍스트라, "긴 노이즈 텍스트"가
             오픈소스 모델 성능을 깎는지 확인하는 기준선)
  evidence - consistency_verification.py와 동일한 프롬프트로 evidence_span(평균 49자)만
             보고 domain/risk_level을 판단. verify_domain/verify_label(GPT-4o가 동일 조건에서
             내린 판단)과 비교. "짧은 텍스트면 오픈소스 모델도 나아지는가"를 검증하는 모드.

비교 대상:
  - Qwen/Qwen2.5-14B-Instruct
  - LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
  - meta-llama/Llama-3.1-8B-Instruct
    (Llama는 70B급이어야 한국어 품질이 좋다고 알려져 있으나, 로컬 단일 GPU(32GB)에서
     양자화 없이 돌릴 수 있는 8B로 우선 비교하고, 품질이 부족하면 별도로 양자화된
     70B 비교를 추가한다.)

실행:
  python -m backend.fb_check.llm_benchmark --mode full
  python -m backend.fb_check.llm_benchmark --mode evidence --sample 20 --gpu 1
"""

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from openai import OpenAI

from backend.fb_check.forward_labeling import _SYSTEM as FWD_SYSTEM, _FEW_SHOT_EXAMPLES as FWD_FEWSHOT, run_forward
from backend.fb_check.consistency_verification import _SYSTEM as VERIFY_SYSTEM, _FEW_SHOT_EXAMPLES as VERIFY_FEWSHOT, run_verify
from backend.utils import load_jsonl, load_logger, save_json, PROJECT_ROOT

logger = load_logger("llm_benchmark.log")

OUT_DIR      = Path(os.environ.get("FB_OUT_DIR", str(PROJECT_ROOT / "data/fb_check")))
RESULTS_PATH = OUT_DIR / "fb_check_results.jsonl"

# 로컬 GPU에서 transformers로 돌리는 후보
CANDIDATES = {
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "exaone-3.5-7.8b": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}

# OpenAI API로 돌리는 후보 (forward_labeling.run_forward / consistency_verification.run_verify 재사용)
OPENAI_CANDIDATES = {
    "gpt-4o-mini": "gpt-4o-mini",
}

# 모드별: 텍스트 필드, 사용자 메시지 프리픽스, 시스템/few-shot, 비교 기준 필드
MODES = {
    "full": {
        "text_field":   "text",
        "user_prefix":  "계약 조항:\n",
        "system":       FWD_SYSTEM,
        "fewshot":      FWD_FEWSHOT,
        "ref_domain":   "forward_domain",
        "ref_risk":     "forward_label",
        "max_chars":    3000,
    },
    "evidence": {
        "text_field":   "evidence_span",
        "user_prefix":  "근거 문구:\n",
        "system":       VERIFY_SYSTEM,
        "fewshot":      VERIFY_FEWSHOT,
        "ref_domain":   "verify_domain",
        "ref_risk":     "verify_label",
        "max_chars":    500,
    },
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


def _load_samples(mode: str, n: int) -> list[dict]:
    cfg = MODES[mode]
    records = load_jsonl(RESULTS_PATH)
    records = [
        r for r in records
        if r.get(cfg["ref_domain"]) and r.get(cfg["ref_risk"]) and r.get(cfg["text_field"])
    ]
    return records[:n]


def _load_model(model_id: str, device: str) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _generate(model: Any, tokenizer: Any, device: str, cfg: dict[str, Any], input_text: str) -> dict | None:
    messages = [{"role": "system", "content": cfg["system"]}]
    messages.extend(cfg["fewshot"])
    messages.append({"role": "user", "content": f"{cfg['user_prefix']}{input_text[:cfg['max_chars']]}"})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=300, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _extract_json(text)


def _generate_openai(client: OpenAI, model_id: str, mode: str, input_text: str) -> dict | None:
    if mode == "full":
        return run_forward(client, input_text, model=model_id)
    return run_verify(client, input_text, model=model_id)


def evaluate_candidate(name: str, model_id: str, mode: str, samples: list[dict], device: str) -> dict[str, Any]:
    cfg = MODES[mode]
    is_openai = name in OPENAI_CANDIDATES
    logger.info(f"[{name}/{mode}] {model_id} {'(OpenAI API)' if is_openai else '로드 중...'}")

    if is_openai:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        model, tokenizer = _load_model(model_id, device)

    domain_match = risk_match = parse_ok = 0
    per_sample: list[dict] = []

    t0 = time.time()
    for i, rec in enumerate(samples, 1):
        if is_openai:
            pred = _generate_openai(client, model_id, mode, rec[cfg["text_field"]][:cfg["max_chars"]])
        else:
            pred = _generate(model, tokenizer, device, cfg, rec[cfg["text_field"]])
        ok = pred is not None
        ref_domain = rec[cfg["ref_domain"]]
        ref_risk   = rec[cfg["ref_risk"]]
        d_match = ok and pred.get("domain") == ref_domain
        r_match = ok and pred.get("risk_level") == ref_risk
        parse_ok += int(ok)
        domain_match += int(d_match)
        risk_match += int(r_match)
        per_sample.append({
            "chunk_id":      rec.get("chunk_id"),
            "ref_domain":    ref_domain,
            "ref_risk":      ref_risk,
            "pred_domain":   pred.get("domain") if ok else None,
            "pred_risk":     pred.get("risk_level") if ok else None,
            "parse_ok":      ok,
        })
        if i % 5 == 0:
            logger.info(f"  [{name}/{mode}] {i}/{len(samples)}")
    elapsed = time.time() - t0

    if not is_openai:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    n = len(samples)
    summary = {
        "model": model_id,
        "mode": mode,
        "n_samples": n,
        "parse_success_rate": round(parse_ok / n, 4),
        "domain_agreement":   round(domain_match / n, 4),
        "risk_agreement":     round(risk_match / n, 4),
        "avg_latency_sec":    round(elapsed / n, 2),
    }
    logger.info(f"[{name}/{mode}] parse={summary['parse_success_rate']:.1%} "
                f"domain={summary['domain_agreement']:.1%} risk={summary['risk_agreement']:.1%} "
                f"avg={summary['avg_latency_sec']:.2f}s/건")
    return {"summary": summary, "samples": per_sample}


ALL_CANDIDATES = {**CANDIDATES, **OPENAI_CANDIDATES}


def main() -> None:
    parser = argparse.ArgumentParser(description="FB-Check 오픈소스/저비용 LLM 비교")
    parser.add_argument("--mode",   choices=list(MODES), default="full",
                        help="full=원문 전체 비교, evidence=근거 문구만 비교 (기본 full)")
    parser.add_argument("--sample", type=int, default=20, help="비교할 샘플 수 (기본 20)")
    parser.add_argument("--gpu",    type=int, default=1,  help="GPU 인덱스 (기본 1, OpenAI 후보엔 미사용)")
    parser.add_argument("--only",   type=str, default=None,
                        help=f"특정 후보만 실행 ({', '.join(ALL_CANDIDATES)})")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    samples = _load_samples(args.mode, args.sample)
    logger.info(f"========== LLM 비교 시작 | mode={args.mode} | 샘플 {len(samples)}건 | device={device} ==========")

    candidates = {args.only: ALL_CANDIDATES[args.only]} if args.only else ALL_CANDIDATES

    out_path = OUT_DIR / f"llm_benchmark_result_{args.mode}.json"
    report: dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for name, model_id in candidates.items():
        try:
            report[name] = evaluate_candidate(name, model_id, args.mode, samples, device)
        except Exception as e:
            logger.error(f"[{name}] 실패: {e}")
            report[name] = {"error": str(e)}

    save_json(report, out_path)
    logger.info("========== LLM 비교 완료 ==========")
    logger.info(f"결과 저장: {out_path}")


if __name__ == "__main__":
    main()
