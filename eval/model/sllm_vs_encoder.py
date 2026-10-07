# backend/eval/sllm_vs_encoder.py
"""EXAONE LoRA와 `article_v2`를 **같은 255건에서 짝지어** 비교한다.

## 왜 따로 재는가

`sllm_lora_eval`은 sLLM 쪽 점수만 낸다. 두 값을 나란히 적는 것과 **짝지어 재는 것**은
다르다 — 같은 조항에 대한 두 점수의 차를 봐야 조항 난이도가 상쇄되고, 그래야
"+4.6%p가 우연인가"에 답할 수 있다. 상수 기준선만 병기하고 CI를 안 내면 규칙 8 위반이다.

    주 검정   per-sample F1 차의 **문서 단위** 부트스트랩 (형제 조항이 독립이 아니다)
    보조      승-무-패 카운트 · 조항 단위 재현
    상수      "항상 상위 k개 조 찍기" — 두 모델 다 이걸 넘어야 의미가 있다

## 잣대를 새로 만들지 않는다

인코더 예측은 배포 임계값(`thresholds.npy`)과 체크포인트의 `max_len`으로 낸다.
여기서 임계값을 다시 고르면 그게 곧 gold 오염이다(`article_gold_eval` 주석과 같은 이유).
sLLM 예측은 `data/eval/model/sllm_lora_<tag>.json`에 저장된 것을 **그대로 읽는다** — 다시
생성하면 같은 모델이라도 값이 흔들린다.

실행:

    .venv/bin/python -m eval.sllm_vs_encoder --gpu 1 --tag exaone
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from backend.training.train_article import constant_baseline_f1, load_ftc_gold
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("sllm_vs_encoder.log")
OUT_PATH = PROJECT_ROOT / "data/eval/model/sllm_vs_encoder.json"


def _f1(pred: list[str], gold: list[str]) -> float:
    p, g = set(pred), set(gold)
    inter = p & g
    pr = len(inter) / len(p) if p else 0.0
    rc = len(inter) / len(g) if g else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def paired_doc_bootstrap(diffs: list[float], docs: list[str],
                         n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    """차의 문서 단위 부트스트랩 CI. 조항 단위로 재표집하면 형제 조항 때문에 좁아진다."""
    by_doc: dict[str, list[float]] = {}
    for d, doc in zip(diffs, docs):
        by_doc.setdefault(doc or "_", []).append(d)
    groups = list(by_doc.values())
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        flat = [v for i in pick for v in groups[i]]
        if flat:
            vals.append(float(np.mean(flat)))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def encoder_predictions(model_dir: str, gpu: int) -> tuple[list[list[str]], list[str]]:
    """배포 상태 그대로 예측한다 — 임계값·max_len을 여기서 고르지 않는다."""
    import torch
    from transformers import AutoTokenizer

    from backend.model.electra import ArticleMultiLabelElectra
    from backend.training.train_article import ArticleDataset

    d = PROJECT_ROOT / model_dir
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    names = list(metrics["article_names"])
    max_len = int((metrics.get("train_config") or {}).get("max_len", 256))
    thr = np.load(d / "thresholds.npy")
    logger.info(f"  인코더 {model_dir}: max_len={max_len} · 임계값 {np.round(thr, 2).tolist()}")

    dev = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = ArticleMultiLabelElectra.load(d).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(str(d))
    gold = load_ftc_gold("clean")
    ds = ArticleDataset([{"text": g["text"], "articles": g["articles"], "group": "g"} for g in gold],
                        tok, max_len, names)
    out = []
    with torch.no_grad():
        for b in torch.utils.data.DataLoader(ds, batch_size=16):
            p = torch.sigmoid(model(b["input_ids"].to(dev), b["attention_mask"].to(dev),
                                    b["token_type_ids"].to(dev))).cpu().numpy()
            out.append(p)
    probs = np.concatenate(out)
    preds = [[names[j] for j in range(len(names)) if probs[i, j] >= thr[j]]
             for i in range(len(probs))]
    return preds, names


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exaone")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--encoder", default="models/article_v2")
    args = ap.parse_args(argv)

    sllm = json.loads((PROJECT_ROOT / f"data/eval/model/sllm_lora_{args.tag}.json").read_text(encoding="utf-8"))
    gold = load_ftc_gold("clean")
    enc_preds, names = encoder_predictions(args.encoder, args.gpu)
    sllm_preds = [r["pred"] for r in sllm["rows"]]
    if not (len(gold) == len(enc_preds) == len(sllm_preds)):
        raise SystemExit(f"길이 불일치: gold {len(gold)} · enc {len(enc_preds)} · sllm {len(sllm_preds)}")

    g_in = [[a for a in g["articles"] if a in names] for g in gold]
    docs = [g["doc_id"] for g in gold]
    f_enc = [_f1(p, g) for p, g in zip(enc_preds, g_in)]
    f_sll = [_f1(p, g) for p, g in zip(sllm_preds, g_in)]
    diffs = [a - b for a, b in zip(f_sll, f_enc)]
    lo, hi = paired_doc_bootstrap(diffs, docs)
    win = sum(d > 1e-9 for d in diffs)
    lose = sum(d < -1e-9 for d in diffs)
    tie = len(diffs) - win - lose

    base = {k: constant_baseline_f1([{"articles": g} for g in g_in], names, k)["f1"] for k in (1, 2, 3)}
    best_k = max(base, key=base.get)
    rec_enc = float(np.mean([bool(p) for p, g in zip(enc_preds, g_in) if g]))
    rec_sll = float(np.mean([bool(p) for p, g in zip(sllm_preds, g_in) if g]))
    d_mean = float(np.mean(diffs))
    verdict = "sLLM 이김" if lo > 0 else ("인코더 이김" if hi < 0 else "미판정")

    logger.info("=" * 70)
    logger.info(f"★ 짝지은 비교 — gold clean {len(gold)}건 · 문서 {len(set(docs))}개 · 같은 잣대")
    logger.info(f"  {'':22}{'조 F1':>9}{'조항 재현':>11}{'지목':>8}")
    logger.info(f"  {'article_v2 (110M)':22}{np.mean(f_enc)*100:>8.1f}%{rec_enc*100:>10.1f}%"
                f"{sum(bool(p) for p in enc_preds):>8}")
    logger.info(f"  {'EXAONE LoRA (7.8B)':22}{np.mean(f_sll)*100:>8.1f}%{rec_sll*100:>10.1f}%"
                f"{sum(bool(p) for p in sllm_preds):>8}")
    logger.info(f"  {'상수 최고(k=' + str(best_k) + ')':22}{base[best_k]*100:>8.1f}%")
    logger.info("")
    logger.info(f"  ★ 차 {d_mean*100:+.1f}%p  95% CI [{lo*100:+.1f}, {hi*100:+.1f}]  "
                f"(문서 단위 부트스트랩) — **{verdict}**")
    logger.info(f"  승-무-패  {win}-{tie}-{lose}")
    logger.info(f"  상수 대비  인코더 {(np.mean(f_enc)-base[best_k])*100:+.1f}%p · "
                f"sLLM {(np.mean(f_sll)-base[best_k])*100:+.1f}%p")
    if verdict == "미판정":
        d = abs(d_mean) / (np.std(diffs) + 1e-9)
        need = int(np.ceil(2 * (1.96 + 0.84) ** 2 / max(d, 1e-3) ** 2))
        logger.info(f"  ⚠ 규칙 8 — 관측 d={d:.2f}. 80% 검정력에 약 n={need}건 필요(보유 {len(gold)})")
    logger.info("=" * 70)

    save_json({
        "population": f"gold clean {len(gold)} · 문서 {len(set(docs))} · 조 목록 {names}",
        "encoder": {"dir": args.encoder, "f1": float(np.mean(f_enc)), "recall_any": rec_enc,
                    "n_picked": sum(bool(p) for p in enc_preds)},
        "sllm": {"tag": args.tag, "base": sllm["base_model"], "f1": float(np.mean(f_sll)),
                 "recall_any": rec_sll, "n_picked": sum(bool(p) for p in sllm_preds)},
        "constant_f1": base, "best_k": best_k,
        "paired_diff": d_mean, "ci95": [lo, hi], "verdict": verdict,
        "win_tie_lose": [win, tie, lose],
        "note": ("짝지은 문서 단위 부트스트랩. 인코더는 배포 임계값·체크포인트 max_len 그대로, "
                 "sLLM은 저장된 생성 결과 그대로 — 어느 쪽도 여기서 다시 고르지 않았다"),
    }, OUT_PATH)
    logger.info(f"저장: {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
