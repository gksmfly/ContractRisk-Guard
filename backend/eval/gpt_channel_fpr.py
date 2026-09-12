# backend/eval/gpt_channel_fpr.py
"""GPT(forward) 채널이 표준계약서를 얼마나 짚는가 — 두 채널 게이트의 전제 검사.

## 왜 이걸 재는가

모델 오경보율은 출처로만 정의한 풀에서 쟀다(`article_gold_eval`). 그런데 **두 채널
게이트**(모델과 GPT가 둘 다 짚어야 표시)를 켤지 말지는 GPT 쪽 오경보율을 모르면
판단할 수 없다. GPT가 아무 데서나 울리면 게이트는 정밀도를 못 올린다.

`project_remaining_work`의 A3다. 오래 "순환이라 측정 불가"로 적혀 있었는데, **준거를
GPT 라벨이 아니라 출처(정부 발행 표준계약서)로 바꾸면서 풀렸다.** 저장된 forward 라벨을
세기만 하면 된다 — API 호출 0.

## 풀은 출처로만 정의한다

    clean.jsonl + noise.jsonl 에서 source == "standard_contract" 인 것 전부

`clean.jsonl`만 읽거나 `agreed_articles`가 빈 것만 고르면 **GPT 라벨로 GPT를 채점**하게
된다. 철회한 "오경보 2.6%"가 정확히 그 경로였다.

## 온전 지층 — 규칙을 코드에 적는다

사용자가 붙여넣는 것은 온전한 조항이므로 배포 모집단은 온전 지층이다. 그런데
`reported_numbers.md`의 **`n=613`을 만든 코드가 레포에 없어 재현되지 않는다**
(철회한 `+8.8%p`와 같은 상태). 그래서 규칙을 여기 명시하고 그 결과를 보고한다:

    종결어미로 끝난다 (다/음/함/임 + 선택적 마침표)   ← 잘린 조각 배제
    텍스트 중복 제거                                ← 표준계약서끼리 문구를 공유한다

헤딩 검사는 넣지 않는다 — 이 풀은 **2,036건 전부가 `제N조`로 시작**해서 아무것도
거르지 못한다. `reported_numbers`의 "헤딩아님 74건"은 다른 기준으로 센 것이고,
그 기준이 무엇인지는 남아 있지 않다.

실행:

    .venv/bin/python -m backend.eval.gpt_channel_fpr
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import numpy as np

from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("gpt_channel_fpr.log")

OUT_PATH = PROJECT_ROOT / "data/eval/gpt_channel_fpr.json"
_END = re.compile(r"(다|음|함|임)\s*\.?\s*$")

# 모델 쪽 비교값(`reported_numbers.md` §2). 여기서 다시 재지 않는다 — GPU가 필요하고,
# 채점 경로가 갈라지면 두 채널을 같은 자로 잰 게 아니게 된다.
_MODEL_FPR = {"all": 0.094, "intact": 0.121}


def _doc(chunk_id: str) -> str:
    return ":".join(str(chunk_id).split(":")[:2])


def load_negative_pool() -> list[dict]:
    """표준계약서 조항 전부. **출처로만 정의한다** — 라벨로 거르면 순환이다."""
    rows = {r["chunk_id"]: r for r in
            load_jsonl(PROJECT_ROOT / "data/fb_check/clean.jsonl")
            + load_jsonl(PROJECT_ROOT / "data/fb_check/noise.jsonl")}
    return [r for r in rows.values() if r.get("source") == "standard_contract"]


def intact_stratum(pool: list[dict]) -> list[dict]:
    """종결어미로 끝나고 중복이 아닌 조항. 규칙은 모듈 docstring에 적혀 있다."""
    seen: set[str] = set()
    out = []
    for r in pool:
        t = r["text"].strip()
        if _END.search(t) and t not in seen:
            seen.add(t)
            out.append(r)
    return out


def doc_bootstrap_ci(pool: list[dict], n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """문서 단위 부트스트랩. 조항 단위 Wilson은 독립을 가정해 틀린다(형제 조항)."""
    by_doc: dict[str, list[int]] = {}
    for r in pool:
        by_doc.setdefault(_doc(r["chunk_id"]), []).append(1 if (r.get("forward_articles") or []) else 0)
    docs = list(by_doc.values())
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(docs), len(docs))
        flat = [v for i in pick for v in docs[i]]
        if flat:
            vals.append(float(np.mean(flat)))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def encoder_flags(pool: list[dict], model_dir: str, gpu: int) -> list[bool]:
    """인코더가 각 조항을 짚었는가. **배포 임계값·체크포인트 max_len 그대로** 쓴다.

    `article_gold_eval`의 오경보는 홀드아웃 12문서(n=300)에서 쟀고 온전 지층 값은
    n=613인데 그 코드가 없다. 여기서는 **GPT와 똑같은 풀**에 돌려 분모를 맞춘다 —
    분모가 다르면 두 채널을 비교했다고 말할 수 없다.
    """
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
    ds = ArticleDataset([{"text": r["text"], "articles": [], "group": "g"} for r in pool],
                        tok, max_len, names)
    out = []
    with torch.no_grad():
        for b in torch.utils.data.DataLoader(ds, batch_size=32):
            out.append(torch.sigmoid(model(b["input_ids"].to(dev), b["attention_mask"].to(dev),
                                           b["token_type_ids"].to(dev))).cpu().numpy())
    probs = np.concatenate(out)
    return [bool((probs[i] >= thr).any()) for i in range(len(probs))]


def _rate(pool: list[dict], label: str) -> dict:
    flagged = [r for r in pool if (r.get("forward_articles") or [])]
    lo, hi = doc_bootstrap_ci(pool)
    d = {"label": label, "n": len(pool), "n_docs": len({_doc(r["chunk_id"]) for r in pool}),
         "n_flagged": len(flagged), "fpr": len(flagged) / len(pool) if pool else 0.0,
         "ci95": [lo, hi]}
    logger.info(f"  {label:24} n={d['n']:>5}  문서 {d['n_docs']:>2}  플래그 {d['n_flagged']:>4}  "
                f"= {d['fpr']*100:5.1f}%  [{lo*100:.1f}, {hi*100:.1f}]")
    return d


def _flag_rate(flags: list[bool], pool: list[dict], label: str) -> dict:
    by_doc: dict[str, list[int]] = {}
    for f, r in zip(flags, pool):
        by_doc.setdefault(_doc(r["chunk_id"]), []).append(int(f))
    groups = list(by_doc.values())
    rng = np.random.default_rng(42)
    vals = []
    for _ in range(2000):
        pick = rng.integers(0, len(groups), len(groups))
        flat = [v for i in pick for v in groups[i]]
        if flat:
            vals.append(float(np.mean(flat)))
    lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    d = {"label": label, "n": len(pool), "n_flagged": int(sum(flags)),
         "fpr": float(np.mean(flags)), "ci95": [lo, hi]}
    logger.info(f"  {label:24} n={d['n']:>5}  플래그 {d['n_flagged']:>4}  "
                f"= {d['fpr']*100:5.1f}%  [{lo*100:.1f}, {hi*100:.1f}]")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="", help="인코더도 같은 풀에서 잰다(예: models/article_v2)")
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()
    pool = load_negative_pool()
    intact = intact_stratum(pool)

    logger.info("★ GPT(forward) 채널 오경보율 — 준거는 '정부 발행 표준계약서는 적법'")
    logger.info("  풀은 출처로만 정의했다. 라벨로 거르면 GPT로 GPT를 채점하게 된다")
    r_all = _rate(pool, "① 전체(파편 포함)")
    r_int = _rate(intact, "② 온전 지층")

    logger.info("")
    logger.info("  ----- 두 채널 비교 (같은 풀, 같은 준거) -----")
    logger.info(f"    {'':24}{'모델':>10}{'GPT':>10}{'비':>8}")
    for key, r in (("all", r_all), ("intact", r_int)):
        m = _MODEL_FPR[key]
        logger.info(f"    {r['label']:24}{m*100:>9.1f}%{r['fpr']*100:>9.1f}%{r['fpr']/m:>8.2f}x")
    logger.info("    ★ 모델 값은 reported_numbers.md §2에서 가져왔다 — 여기서 다시 재지 않는다")
    logger.info("    ⚠ ② 온전 지층의 모델 값 12.1%는 n=613에서 잰 것인데 **그 코드가 레포에 "
                f"없다.** 이 파일의 규칙으로는 n={len(intact)}이라 분모가 다르다. "
                "비 열은 그만큼 느슨하게 읽을 것")

    enc_out = {}
    if args.encoder:
        logger.info("")
        logger.info(f"  ----- 인코더를 **같은 풀**에서 재측정 ({args.encoder}) -----")
        f_all = encoder_flags(pool, args.encoder, args.gpu)
        e_all = _flag_rate(f_all, pool, "① 전체(파편 포함)")
        keep = {id(r) for r in intact}
        f_int = [f for f, r in zip(f_all, pool) if id(r) in keep]
        e_int = _flag_rate(f_int, intact, "② 온전 지층")
        enc_out = {"dir": args.encoder, "all": e_all, "intact": e_int}
        logger.info("")
        logger.info("  ----- 세 채널, 같은 분모 -----")
        logger.info(f"    {'':24}{'인코더':>10}{'GPT':>10}{'비':>8}")
        for e, g in ((e_all, r_all), (e_int, r_int)):
            logger.info(f"    {e['label']:24}{e['fpr']*100:>9.1f}%{g['fpr']*100:>9.1f}%"
                        f"{g['fpr']/max(e['fpr'], 1e-9):>8.2f}x")
        logger.info(f"    ★ 이제 분모가 같다. 기록의 12.1%(n=613)는 여기 ②와 직접 비교할 수 없다")

    by_article = Counter(a for r in intact if (r.get("forward_articles") or [])
                         for a in r["forward_articles"])
    logger.info("")
    logger.info("  ----- GPT가 온전 조항에서 짚은 조 -----")
    for a, n in sorted(by_article.items(), key=lambda x: -x[1]):
        logger.info(f"    {a:>6}  {n:>4}건")

    verdict = ("30% 미만 — 게이트가 자동으로 죽지는 않는다. 실제 효과는 재봐야 안다"
               if r_int["fpr"] < 0.30 else
               "30% 이상 — GPT가 너무 자주 울려 두 채널 게이트가 정밀도를 못 올린다")
    logger.info("")
    logger.info(f"  ★ 판정: GPT 온전 지층 {r_int['fpr']*100:.1f}% → {verdict}")
    logger.info("  ⚠ 이 라벨은 `forward_model`이 gpt-4o다. 운영 Analysis는 gpt-4o-mini이므로 "
                "게이트 자체의 효과(+13.4%p)는 여전히 운영 모델로 재측정해야 한다")

    save_json({
        "pool_definition": "clean.jsonl + noise.jsonl 중 source == standard_contract (출처로만)",
        "intact_rule": "종결어미(다/음/함/임 + 선택적 마침표)로 끝나고 텍스트 중복이 아닌 것",
        "all": r_all, "intact": r_int,
        "encoder_same_pool": enc_out,
        "model_fpr_reference": _MODEL_FPR,
        "by_article_intact": dict(by_article.most_common()),
        "verdict": verdict,
        "caveat": ("reported_numbers.md의 온전 지층 n=613은 이 파일의 규칙으로 재현되지 않는다"
                   f"(여기서는 n={len(intact)}). 613을 만든 코드가 레포에 없다. "
                   "forward 라벨은 gpt-4o이고 운영은 gpt-4o-mini다"),
    }, OUT_PATH)
    logger.info(f"저장: {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
