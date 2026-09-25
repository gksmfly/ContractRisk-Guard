# backend/eval/sllm_lora_eval.py
"""디코더 sLLM을 LoRA로 붙여 조 multi-label을 재고, `article_v2`와 **같은 잣대**로 채점한다.

## 왜 이걸 만드는가

백본 축은 **인코더 8종**으로 닫혔다(110M~600M / 2020~2025, 전부 상수 아래).
그런데 "디코더 계열을 파인튜닝하면 다른가"는 한 번도 안 쟀다. 중간발표 Q&A에서
"안 해봤습니다"로 답한 자리가 여기다. 안 해본 것과 재봤더니 아닌 것은 다르다.

## 잣대를 새로 만들지 않는다

이 파일이 지키는 규칙 하나: **채점 경로를 새로 짜지 않는다.**
`load_ftc_gold` / `exclude_gold_documents` / `constant_baseline_f1`을 `train_article`에서
그대로 가져다 쓰고, 조 목록은 `models/article_v2/metrics.json`의 것을 읽는다.
새로 짜면 점수가 올라도 **모델이 좋아진 건지 채점이 후해진 건지** 구분할 수 없다 —
45.7%가 85.3%로 바뀐 사고가 정확히 그것이었다.

    학습   data/fb_check/clean.jsonl  ·  label_source=agreed (article_v2 기본값과 동일)
    평가   load_ftc_gold("clean")  n=255  ·  per-sample 조 F1  ·  상수 기준선 병기
    배제   exclude_gold_documents — gold와 같은 문서에서 나온 학습 레코드 제거

## 미리 적어두는 한계 (규칙 8)

gold 255건에서 3%p를 보려면 n≈3,400이 필요하다. **차이가 작으면 미판정으로 끝난다.**
그것도 답이다 — "이 정답셋 크기로는 판정 불가"라고 적는다. 효과 없음이 아니다.

실행:

    .venv/bin/python -m backend.eval.sllm_lora_eval --gpu 0
    .venv/bin/python -m backend.eval.sllm_lora_eval --gpu 0 \
        --base meta-llama/Llama-3.1-8B-Instruct --tag llama31
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from backend.labeling.articles import ARTICLE_IDS
from backend.training.train_article import (
    constant_baseline_f1,
    exclude_gold_documents,
    load_article_records,
    load_ftc_gold,
)
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("sllm_lora_eval.log")

CLEAN_PATH = PROJECT_ROOT / "data/fb_check/clean.jsonl"
OUT_DIR = PROJECT_ROOT / "data/eval"
DEFAULT_BASE = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

_SYS = "너는 한국 약관규제법 전문가다. 계약 조항을 읽고 위반 소지가 있는 조를 고른다."
_INSTR = (
    "다음 계약 조항이 약관의 규제에 관한 법률 제6조~제14조 중 어느 조를 위반할 소지가 있는지 "
    "판단하라. 해당하는 조만 쉼표로 나열하고, 없으면 '없음'이라고만 답하라.\n\n조항:\n{text}"
)
_ART_RE = re.compile(r"제\s*(\d{1,2})\s*조")


def _patch_input_embeddings(model: Any) -> None:
    """`get_input_embeddings`가 없는 원격 코드 모델을 살린다.

    EXAONE-3.5의 `modeling_exaone.py`는 `ExaoneModel`에 이 메서드를 두지 않는데,
    transformers 5.x 기본 클래스는 NotImplementedError를 던지고 peft는 어댑터를 꽂기
    전에 이걸 호출한다. 임베딩 모듈을 직접 찾아 클래스에 매단다 —
    **모델 가중치는 건드리지 않는다.**
    """
    import torch.nn as nn
    try:
        model.get_input_embeddings()
        return
    except NotImplementedError:
        pass
    inner = getattr(model, "model", None) or getattr(model, "transformer", None)
    if inner is None:
        raise
    name = next((n for n, m in inner.named_children() if isinstance(m, nn.Embedding)), None)
    if name is None:
        raise SystemExit(f"{type(inner).__name__}에서 임베딩 모듈을 못 찾았다")
    cls = type(inner)
    cls.get_input_embeddings = lambda self, _n=name: getattr(self, _n)
    cls.set_input_embeddings = lambda self, v, _n=name: setattr(self, _n, v)
    logger.info(f"  패치: {cls.__name__}.get_input_embeddings → {name}")


def _names_from_checkpoint(model_dir: Path) -> list[str]:
    """조 목록을 `article_v2`에서 읽는다 — 여기서 다시 고르면 잣대가 어긋난다."""
    m = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    names = m.get("article_names")
    if not names:
        raise SystemExit(f"{model_dir}/metrics.json에 article_names가 없다")
    return list(names)


def _target(arts: list[str], names: list[str]) -> str:
    kept = [a for a in arts if a in names]
    return ", ".join(sorted(kept, key=lambda a: int(a.strip("제조")))) if kept else "없음"


def _parse(out: str, names: list[str]) -> list[str]:
    """생성 결과에서 조 번호만 뽑는다. 헤드 목록 밖은 버린다(인코더와 같은 조건)."""
    found = {f"제{int(n)}조" for n in _ART_RE.findall(out)}
    return sorted(found & set(names), key=lambda a: int(a.strip("제조")))


def _f1(pred: list[str], gold: list[str]) -> float:
    p, g = set(pred), set(gold)
    inter = p & g
    pr = len(inter) / len(p) if p else 0.0
    rc = len(inter) / len(g) if g else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def _split_by_document(records: list[dict], dev_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """문서 단위로 가른다 — 조항 단위로 가르면 형제 조항이 양쪽에 걸려 누수가 된다."""
    groups = sorted({r["group"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_dev = max(1, int(len(groups) * dev_frac))
    dev_groups = set(groups[:n_dev])
    tr = [r for r in records if r["group"] not in dev_groups]
    dv = [r for r in records if r["group"] in dev_groups]
    return tr, dv


def _build_example(tok: Any, text: str, target: str, max_len: int) -> dict | None:
    """프롬프트는 손실에서 가린다 — 답변 토큰만 학습한다."""
    msgs = [{"role": "system", "content": _SYS},
            {"role": "user", "content": _INSTR.format(text=text)}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(target + tok.eos_token, add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(a_ids) > max_len:
        return None
    ids = p_ids + a_ids
    labels = [-100] * len(p_ids) + a_ids
    return {"input_ids": ids, "labels": labels}


def _collate(batch: list[dict], pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    import torch
    ids, lab, att = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * k)
        lab.append(b["labels"] + [-100] * k)
        att.append([1] * len(b["input_ids"]) + [0] * k)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
            "attention_mask": torch.tensor(att)}


def _generate(model: Any, tok: Any, dev: Any, texts: list[str], bs: int = 16) -> list[str]:
    """배치 생성. 디코더는 **왼쪽 패딩**이어야 한다 — 오른쪽이면 생성이 패드에서 시작한다."""
    import torch
    prev, tok.padding_side = tok.padding_side, "left"
    outs: list[str] = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": _SYS},
             {"role": "user", "content": _INSTR.format(text=t)}],
            tokenize=False, add_generation_prompt=True) for t in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024, add_special_tokens=False).to(dev)
        n_in = enc["input_ids"].shape[-1]
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=32, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        outs += [tok.decode(g[n_in:], skip_special_tokens=True).strip() for g in gen]
    tok.padding_side = prev
    return outs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--tag", default="exaone")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ref-model", default="models/article_v2",
                    help="조 목록의 출처. 여기서 다시 고르지 않는다")
    ap.add_argument("--limit", type=int, default=0, help="연기 테스트용. 0이면 전부")
    ap.add_argument("--load-4bit", action="store_true",
                    help="NF4 양자화로 로드. 14B급은 bf16이 32GB에 안 들어간다")
    args = ap.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    names = _names_from_checkpoint(PROJECT_ROOT / args.ref_model)
    folded = sorted(set(ARTICLE_IDS) - set(names), key=lambda a: int(a.strip("제조")))
    logger.info(f"★ 조 목록({len(names)}개, 출처 {args.ref_model}): {names} · 접음 {folded}")

    gold = load_ftc_gold("clean")
    records = load_article_records(CLEAN_PATH, label_source="agreed")
    records = exclude_gold_documents(records, gold)
    tr, dv = _split_by_document(records, 0.05, args.seed)
    if args.limit:
        tr, dv, gold = tr[:args.limit], dv[:16], gold[:32]
    logger.info(f"★ 학습 {len(tr)} · dev {len(dv)} · gold(clean) {len(gold)}")

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.load_4bit:
        # 14B는 bf16 가중치만 28GB라 LoRA를 얹으면 32GB를 넘는다. NF4로 약 9GB.
        # ⚠ 양자화는 **비교 조건을 바꾼다** — 7.8B(bf16)와 14B(4bit)를 나란히 놓을 때
        #    "크기 차이"와 "양자화 차이"가 섞인다. 결과에 플래그를 남긴다.
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            args.base, trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.bfloat16, trust_remote_code=True).to(dev)
    _patch_input_embeddings(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"★ LoRA 학습 파라미터 {trainable/1e6:.1f}M / 전체 {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    def prep(rs: list[dict]) -> list[dict]:
        out = [e for r in rs
               if (e := _build_example(tok, r["text"], _target(r["articles"], names), args.max_len))]
        return out

    tr_ds, dv_ds = prep(tr), prep(dv)
    logger.info(f"★ 토큰화 후 학습 {len(tr_ds)} (길이 초과 {len(tr)-len(tr_ds)}건 제외)")
    loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                        collate_fn=lambda b: _collate(b, tok.pad_token_id))
    dv_loader = DataLoader(dv_ds, batch_size=args.batch, shuffle=False,
                           collate_fn=lambda b: _collate(b, tok.pad_token_id))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total = (len(loader) // args.accum + 1) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(total, 2),
                                                pct_start=0.05)
    t0, step = time.time(), 0
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for i, b in enumerate(loader):
            b = {k: v.to(dev) for k, v in b.items()}
            loss = model(**b).loss / args.accum
            loss.backward()
            run += loss.item() * args.accum
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 20 == 0:
                    logger.info(f"  ep{ep+1} step{step}/{total} loss {run/(i+1):.4f} "
                                f"({time.time()-t0:.0f}s)")
        model.eval()
        with torch.no_grad():
            dl = [model(**{k: v.to(dev) for k, v in b.items()}).loss.item() for b in dv_loader]
        logger.info(f"★ epoch {ep+1}  train {run/max(len(loader),1):.4f}  dev {np.mean(dl):.4f}")

    # ── 채점: gold 255건. article_gold_eval과 같은 per-sample F1 ──────────────
    model.config.use_cache = True
    model.eval()
    logger.info(f"  gold {len(gold)}건 생성 중…")
    raws = _generate(model, tok, dev, [g["text"] for g in gold])
    preds = [_parse(r, names) for r in raws]
    rows = [{"text": g["text"][:120], "gold": g["articles"], "pred": p, "raw": r}
            for g, p, r in zip(gold, preds, raws)]

    # ── 오경보: 표준계약서 온전 지층. 재현만 보고 끝내면 "더 많이 짚어서 더 맞혔다"와
    #    구분이 안 된다. 풀·규칙은 `gpt_channel_fpr`에서 그대로 가져온다(분모 일치).
    from backend.eval.gpt_channel_fpr import intact_stratum, load_negative_pool
    neg = intact_stratum(load_negative_pool())
    logger.info(f"  음성 풀(온전 지층) {len(neg)}건 생성 중…")
    neg_raw = _generate(model, tok, dev, [r["text"] for r in neg])
    neg_flag = [bool(_parse(r, names)) for r in neg_raw]
    fpr = float(np.mean(neg_flag))
    logger.info(f"  ★ 오경보율(온전 {len(neg)}) {fpr*100:.1f}%  "
                f"— 비교: article_v2 12.1% · GPT(fwd) 22.8%")

    gold_in = [{"articles": [a for a in g["articles"] if a in names]} for g in gold]
    f1s = [_f1(p, g["articles"]) for p, g in zip(preds, gold_in)]
    picked = [(p, g) for p, g in zip(preds, gold_in) if p]
    recall_any = float(np.mean([bool(p) for p, g in zip(preds, gold_in) if g["articles"]]))

    base = {k: constant_baseline_f1(gold_in, names, k)["f1"] for k in (1, 2, 3)}
    result = {
        "base_model": args.base, "tag": args.tag, "names": names, "folded": folded,
        "n_train": len(tr_ds), "n_dev": len(dv_ds), "n_gold": len(gold),
        "epochs": args.epochs, "lora_r": args.lora_r, "lr": args.lr,
        "load_4bit": args.load_4bit, "seed": args.seed,
        "f1_all": float(np.mean(f1s)),
        "f1_picked": float(np.mean([_f1(p, g["articles"]) for p, g in picked])) if picked else 0.0,
        "n_picked": len(picked),
        "recall_any": recall_any,
        "constant_f1": base,
        "fpr_intact": fpr, "n_neg": len(neg),
        "fpr_reference": {"article_v2": 0.121, "gpt_forward": 0.228,
                          "pool": "표준계약서 온전 지층 n=908 (gpt_channel_fpr와 동일)"},
        "minutes": round((time.time() - t0) / 60, 1),
        "note": ("잣대는 article_v2와 동일: load_ftc_gold('clean') · per-sample F1 · "
                 "조 목록은 models/article_v2/metrics.json. gold 255로는 3%p 미만이 미판정"),
    }
    logger.info("=" * 66)
    logger.info(f"★ {args.tag}  조 F1(전체 {len(gold)})  {result['f1_all']*100:.1f}%  "
                f"vs 상수 k=1 {base[1]*100:.1f}% · k=2 {base[2]*100:.1f}% · k=3 {base[3]*100:.1f}%")
    logger.info(f"★ {args.tag}  조 F1(지목 {len(picked)}건)  {result['f1_picked']*100:.1f}%  "
                f"· 조항 단위 재현 {recall_any*100:.1f}%  · 오경보 {fpr*100:.1f}%")
    logger.info(f"★ 비교 대상 article_v2: 전체 38.5% (상수 36.1%) · 지목분 49.1% (상수 35.5%) "
                f"· 조항 단위 재현 78.4%")
    logger.info("=" * 66)

    adapter_dir = PROJECT_ROOT / f"models/sllm_lora_{args.tag}"
    model.save_pretrained(str(adapter_dir))
    logger.info(f"어댑터 저장: models/sllm_lora_{args.tag} (재학습 없이 다시 잴 수 있다)")
    save_json({**result, "rows": rows,
               "neg_rows": [{"text": r["text"][:100], "raw": q, "flag": f}
                            for r, q, f in zip(neg, neg_raw, neg_flag) if f]},
              OUT_DIR / f"sllm_lora_{args.tag}.json")
    logger.info(f"저장: data/eval/sllm_lora_{args.tag}.json")


if __name__ == "__main__":
    main()
