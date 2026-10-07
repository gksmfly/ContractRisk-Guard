# backend/eval/serving_cost_compare.py
"""인코더와 sLLM의 **서빙 비용**을 같은 입력으로 잰다.

## 왜 재는가

sLLM이 같은 재현율에서 오경보 6.8%p를 낮췄다(`threshold_match_sweep`). 그런데 "그래도
배포는 안 바꾼다"고 적으면서 **비용은 재지 않고 '비싸다'고만 썼다.** 근거 없이 닫는 것은
근거 없이 여는 것과 같은 잘못이다.

    잰다   조항당 지연(warm) · 모델 로드 시간 · GPU 상주 메모리 · 20조항 계약서 총 시간
    안 잼   배치 처리 최적화(vLLM 등) · 양자화 후 값 — 둘 다 하면 sLLM 쪽이 좋아진다

**지금 코드 그대로의 값**이라 sLLM에 불리한 하한이 아니라 **현재 상태**다. 최적화 여지가
sLLM 쪽에 더 크다는 것은 같이 적는다.

실행:

    .venv/bin/python -m eval.serving_cost_compare --gpu 1
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from eval.channel.gpt_channel_fpr import intact_stratum, load_negative_pool
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("serving_cost_compare.log")
OUT_PATH = PROJECT_ROOT / "data/eval/analysis/serving_cost_compare.json"


def _gpu_mb(dev: object) -> float:
    import torch
    return torch.cuda.memory_allocated(dev) / 1024 ** 2


def measure_encoder(texts: list[str], model_dir: str, gpu: int) -> dict:
    import torch
    from transformers import AutoTokenizer

    from backend.model.electra import ArticleMultiLabelElectra

    d = PROJECT_ROOT / model_dir
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    max_len = int((m.get("train_config") or {}).get("max_len", 256))
    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.init()                        # 통계 API는 컨텍스트가 선 뒤에만 먹는다
    torch.cuda.reset_peak_memory_stats(dev)

    t0 = time.time()
    model = ArticleMultiLabelElectra.load(d).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(str(d))
    load_s = time.time() - t0
    mem = _gpu_mb(dev)

    def one(t: str) -> None:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(dev)
        with torch.no_grad():
            model(enc["input_ids"], enc["attention_mask"],
                  enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])))

    for t in texts[:5]:                      # 워밍업 — 첫 호출은 커널 컴파일이 섞인다
        one(t)
    torch.cuda.synchronize(dev)
    t0 = time.time()
    for t in texts:
        one(t)
    torch.cuda.synchronize(dev)
    per = (time.time() - t0) / len(texts)
    del model
    torch.cuda.empty_cache()
    return {"load_s": load_s, "gpu_mb": mem, "per_clause_s": per}


def measure_sllm(texts: list[str], base: str, adapter: str, gpu: int) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from eval.model.sllm_lora_eval import _INSTR, _SYS, _patch_input_embeddings

    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, trust_remote_code=True).to(dev)
    _patch_input_embeddings(model)
    model = PeftModel.from_pretrained(model, str(PROJECT_ROOT / adapter)).eval()
    load_s = time.time() - t0
    mem = _gpu_mb(dev)

    def one(t: str) -> None:
        enc = tok.apply_chat_template(
            [{"role": "system", "content": _SYS},
             {"role": "user", "content": _INSTR.format(text=t)}],
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt").to(dev)
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=32, do_sample=False,
                           pad_token_id=tok.pad_token_id)

    for t in texts[:3]:
        one(t)
    torch.cuda.synchronize(dev)
    t0 = time.time()
    for t in texts:
        one(t)
    torch.cuda.synchronize(dev)
    per = (time.time() - t0) / len(texts)
    del model
    torch.cuda.empty_cache()
    return {"load_s": load_s, "gpu_mb": mem, "per_clause_s": per}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--n", type=int, default=40, help="측정에 쓸 조항 수")
    ap.add_argument("--base", default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
    ap.add_argument("--adapter", default="models/sllm_lora_exaone")
    args = ap.parse_args(argv)

    pool = intact_stratum(load_negative_pool())
    texts = [r["text"] for r in pool[:args.n]]
    logger.info(f"★ 서빙 비용 — 실제 표준계약서 조항 {len(texts)}건, 순차 1건씩(운영 경로와 동일)")

    enc = measure_encoder(texts, "models/article_v2", args.gpu)
    logger.info(f"  article_v2 (110M)   로드 {enc['load_s']:.1f}s · 상주 {enc['gpu_mb']:.0f}MB · "
                f"조항당 {enc['per_clause_s']*1000:.0f}ms")
    sll = measure_sllm(texts, args.base, args.adapter, args.gpu)
    logger.info(f"  EXAONE LoRA (7.8B)  로드 {sll['load_s']:.1f}s · 상주 {sll['gpu_mb']:.0f}MB · "
                f"조항당 {sll['per_clause_s']*1000:.0f}ms")

    ratio_t = sll["per_clause_s"] / enc["per_clause_s"]
    ratio_m = sll["gpu_mb"] / enc["gpu_mb"]
    logger.info("")
    logger.info(f"  ★ 지연 {ratio_t:.0f}배 · 메모리 {ratio_m:.0f}배")
    logger.info(f"  20조항 계약서 기준  인코더 {enc['per_clause_s']*20:.1f}s → "
                f"sLLM {sll['per_clause_s']*20:.1f}s")
    logger.info("  ⚠ 운영 Analysis의 병목은 이게 아니라 **GPT 호출**이다(20조항 36초 실측). "
                "조 분류를 sLLM으로 바꿔도 총 시간은 그만큼 안 늘어난다")
    logger.info("  ⚠ 배치·vLLM·양자화 미적용. 셋 다 sLLM 쪽 값을 크게 낮춘다 — "
                "**현재 코드 그대로의 값**이지 sLLM의 한계가 아니다")

    save_json({"n_texts": len(texts), "encoder": enc, "sllm": sll,
               "base": args.base, "adapter": args.adapter,
               "latency_ratio": ratio_t, "memory_ratio": ratio_m,
               "caveat": ("순차 1건 처리, 배치·vLLM·양자화 미적용. 운영 총 지연의 병목은 "
                          "GPT 호출(20조항 36초)이라 조 분류 교체의 체감 영향은 이 비보다 작다")},
              OUT_PATH)
    logger.info(f"저장: data/eval/analysis/serving_cost_compare.json")


if __name__ == "__main__":
    main()
