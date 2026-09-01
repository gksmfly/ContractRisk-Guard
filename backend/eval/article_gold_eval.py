# backend/eval/article_gold_eval.py
"""
조 multi-label 모델을 **외부 정답(FTC 근거_법령)**으로 채점한다 — 최종 성적표.

## 왜 학습 루프와 분리했나

`train_article.py`는 dev split만 본다. held-out gold를 학습 루프에서 한 번이라도 보면
그 순간 오염된다 — 사람이 "gold가 오르는 epoch"를 고르게 되기 때문이다. 그래서 채점은
학습이 완전히 끝난 뒤, 별도 프로세스로, **단 한 번** 돌린다.

이 스크립트가 **하지 않는 것**(코드로 막아둔다):

- 임계값 재튜닝. `metrics.json`의 `thresholds`를 그대로 읽어 얼려 쓴다. 여기서 다시
  고르면 "임계값은 dev에서만"이라는 규칙을 우회하는 셈이다
- gold 기반 체크포인트 선택. 인자로 받은 디렉터리 하나만 채점한다. 여러 체크포인트를
  넘겨 제일 좋은 걸 고르는 것도 같은 위반이라 **디렉터리를 하나만 받는다**

## 무엇을 내나

    per-article F1 (support 병기)     ← 희소한 조는 F1이 0/1로 튄다
    macro F1 (support>0 인 조만)      ← support 0인 조는 측정값이 아니라 상수 0
    macro F1 (제6조 제외)             ← 제6조는 정답의 36%에 붙는 다수 클래스
    per-sample F1                    ← 상수 라벨러와 직접 비교되는 값
    상수 기준선 (상위 1~4개 조 고정)   ← 이걸 못 넘으면 학습이 무의미하다

`hit@any`·`완전일치`는 **내지 않는다.** 라벨링 파일럿에서 둘 다 상수 라벨러와 구분되지
않는 것으로 드러났다(완전일치 26% = "항상 제6조" 26%). 지표가 신호를 못 잡는데 그걸
성공 기준으로 삼아 프롬프트를 세 번 고친 전례가 있다.

실행:
    .venv/bin/python -m backend.eval.article_gold_eval models/_article --gpu 1
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from backend.model.electra import ArticleMultiLabelElectra
from backend.training.train_article import (
    _MAJORITY_ARTICLE,
    ArticleDataset,
    constant_baseline_f1,
    load_ftc_gold,
    per_article_metrics,
)
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("article_gold_eval.log")
OUT_PATH = PROJECT_ROOT / "data/eval/article_gold_report.json"


_STRATUM_NOTE = {
    "clean": "근거_법령 1개. 조항 1개 : 조 1개라 귀속이 모호할 여지가 없다",
    "noisy": "근거_법령 2개+. 파서가 형제 조항을 놓쳐 근거가 귀속된 구간 — 채점 불가",
    "all":   "옛 정의(327건). 22%가 채점 불가 구간이므로 과거 수치 대조에만 쓸 것",
}


def _per_sample(probs: np.ndarray, labels: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """건별 F1 배열. 페어드 비교·부트스트랩 CI에 쓴다."""
    pred = (probs >= thr).astype(int)
    tp = (pred * labels).sum(1)
    denom = pred.sum(1) + labels.sum(1)
    return np.where(denom == 0, 1.0, 2 * tp / np.maximum(denom, 1))


def _sample_f1(probs: np.ndarray, labels: np.ndarray, thr: np.ndarray) -> float:
    """건별 F1의 평균 — 상수 라벨러와 같은 방식으로 계산해야 비교가 성립한다."""
    f1s = []
    for p_row, y_row in zip(probs >= thr, labels > 0.5):
        tp = int((p_row & y_row).sum())
        pr = tp / p_row.sum() if p_row.sum() else 0.0
        rc = tp / y_row.sum() if y_row.sum() else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def _paired_ci(a: np.ndarray, b: np.ndarray, seed: int = 42) -> tuple[float, float, float]:
    """페어드 부트스트랩 95% CI. 점추정만 보고 '이김/짐'을 말하지 않기 위해 쓴다."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    boot = np.sort([d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(5000)])
    return float(d.mean() * 100), float(boot[125]), float(boot[4875])


def clause_level(probs: np.ndarray, thr: np.ndarray, neg_probs: np.ndarray) -> dict:
    """**조항 단위** 지표 — 제품이 실제로 서는 축.

    지금까지 이 파일의 모든 표는 **조 단위**였다("제9조라고 했는데 진짜 제9조인가").
    그런데 사용자가 얻는 것은 다른 것이다("표시한 조항이 실제 위반 조항인가"). 조항을
    열어 읽고 "이거 문제네"를 확인하면 목적은 달성된다 — 우리가 제9조라 했는데 실제로
    제8조 위반이어도 **그 조항을 짚어준 것 자체는 값을 한다.**

    실측 격차가 크고 τ에 따라 벌어진다(clean gold + 표준계약서 holdout 기준):

        τ=0.15   조항 단위 재현 92.5%  vs  조 단위 정밀 29%(r=.10)   차이 +10%p
        τ=0.45   조항 단위 재현 80.8%  vs  조 단위 정밀 44%          차이 +29%p
        τ=0.92   조항 단위 재현 51.4%  vs  조 단위 정밀 54%          차이 +36%p

    teacher가 "개수는 맞추는데 어느 조인지가 안 맞는다"(1.43 vs 1.37)고 나온 것과 같은
    구조다. **이 값을 병기하지 않으면 다음에 누가 조 단위 44%만 보고 "쓸 수 없다"고
    판단한다.**

    정밀도는 비위반 풀이 있어야 정의되므로 표준계약서 holdout을 함께 받는다. 다만
    실제 정밀도는 배포 유병률 r에 따라 달라지므로 **r을 가정한 값으로 병기한다**
    (`prevalence_worksheet` 참고 — r은 아직 측정 전이다).
    """
    out = {}
    for t in (0.15, 0.25, 0.35, 0.45, 0.65, 0.92):
        recall = float(np.mean([(probs[i] >= t).any() for i in range(len(probs))]))
        fa = float(np.mean([(neg_probs[i] >= t).any() for i in range(len(neg_probs))]))
        # `false_alarm`이 아니라 **GPT 라벨과의 불일치율**이다 — 음성 풀의 정답이
        # `agreed_articles`(forward ∩ verify), 즉 GPT 라벨 그 자체다. 모델이 GPT보다
        # 잘 찾아서 짚은 것도 여기 들어간다. 독립 준거로 잰 오경보율이 아니다.
        row = {"recall": recall, "disagree_with_gpt": fa}
        for r in (0.05, 0.10, 0.15):
            shown = r * recall + (1 - r) * fa
            row[f"precision_at_r{r:.2f}"] = (r * recall / shown) if shown else 0.0
        out[f"{t:.2f}"] = row
    # 배포 임계값(조별)에서의 값도 함께
    recall = float(np.mean([(probs[i] >= thr).any() for i in range(len(probs))]))
    fa = float(np.mean([(neg_probs[i] >= thr).any() for i in range(len(neg_probs))]))
    row = {"recall": recall, "disagree_with_gpt": fa}
    for r in (0.05, 0.10, 0.15):
        shown = r * recall + (1 - r) * fa
        row[f"precision_at_r{r:.2f}"] = (r * recall / shown) if shown else 0.0
    out["dev_thresholds"] = row
    return out


def _tune(probs: np.ndarray, labels: np.ndarray, thr0: np.ndarray, idx) -> np.ndarray:
    """건별 F1을 최대화하는 조별 임계값을 탐욕적으로 찾는다. **진단 전용.**"""
    thr = thr0.copy()
    for _ in range(3):
        for j in range(thr.shape[0]):
            best = (_sample_f1(probs[idx], labels[idx], thr), thr[j])
            for c in np.arange(0.05, 0.96, 0.05):
                t = thr.copy(); t[j] = c
                sc = _sample_f1(probs[idx], labels[idx], t)
                if sc > best[0]:
                    best = (sc, c)
            thr[j] = best[1]
    return thr


def threshold_regimes(probs: np.ndarray, labels: np.ndarray, thr: np.ndarray,
                      doc_ids: list[str], seed: int = 42) -> dict:
    """**세 값을 병기한다.** 이 셋의 간격이 곧 보정 불일치의 비용이다.

        dev 임계값   배포하면 실제로 나오는 값        ← 보고값
        교차적합     분포가 맞는 보정셋이 있을 때     ← 편향 없는 추정
        오라클       상한. 진단용, 성능 주장 금지

    ## 왜 고정 분할이 아니라 교차적합인가

    gold를 보정/평가로 영구 분할하면 평가 n이 반토막 나 CI가 ±5%p에서 ±7%p로 벌어진다.
    교차적합은 **327건 전부를 평가에 쓰면서** 각 건의 임계값을 그 건을 안 본 폴드에서
    가져오므로, 편향 없는 추정을 full n에서 얻는다. 정보를 버리지 않는다.

    ## 왜 이게 필요했나

    dev와 gold의 라벨 사전분포가 정반대다 — dev 빈 라벨 67%, gold 0%. dev에서 튜닝한
    임계값은 gold에서 구조적으로 지나치게 보수적이라 모델을 **과소평가**한다. 실측에서
    `ratio 0.5`가 dev 임계값 기준 꼴찌(31.2%)였다가 교차적합에서 2위(47.9%)로 뒤집혔다 —
    임계값 열이 없으면 그 표는 모델 품질이 아니라 임계값 어긋남을 읽게 된다.

    폴드는 **사건 단위**로 가른다(형제 조항 누수 방지).
    """
    docs = sorted(set(doc_ids))
    rng = random.Random(seed); rng.shuffle(docs)
    half = set(docs[:len(docs) // 2])
    A = np.array([i for i, d in enumerate(doc_ids) if d in half])
    B = np.array([i for i in range(len(doc_ids)) if doc_ids[i] not in half])
    if not len(A) or not len(B):
        return {"note": f"폴드 분할 실패 — 사건이 {len(docs)}개뿐이다"}

    thr_a, thr_b = _tune(probs, labels, thr, A), _tune(probs, labels, thr, B)
    cross = np.empty(len(doc_ids))
    cross[B] = _per_sample(probs[B], labels[B], thr_a)     # A에서 맞춘 임계값으로 B 채점
    cross[A] = _per_sample(probs[A], labels[A], thr_b)
    return {
        "dev": float(_sample_f1(probs, labels, thr)),
        "crossfit": float(cross.mean()),
        "oracle_diagnostic_only": float(_sample_f1(probs, labels, _tune(probs, labels, thr, np.arange(len(labels))))),
        "crossfit_per_sample": cross.tolist(),
        "folds": {"A": len(A), "B": len(B), "n_docs": len(docs)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="조 multi-label 모델 외부 정답 채점")
    # 디렉터리를 **하나만** 받는다 — 여러 개를 받으면 gold로 고르게 되고, 그게 곧 누수다.
    ap.add_argument("model_dir", help="채점할 체크포인트 디렉터리 (하나만)")
    ap.add_argument("--stratum", choices=["clean", "noisy", "all"], default="clean",
                    help="clean=근거_법령 1개(n=255, 주 평가) / noisy=2개+(n=72, 채점 불가) / "
                         "all=327(옛 정의, 과거 수치 대조용). 기본이 clean으로 바뀌었으므로 "
                         "예전 리포트와 값이 다르다 — 리포트 머리의 stratum 줄로 확인할 것")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    names = metrics["article_names"]

    # 임계값은 학습 때 dev에서 확정된 값. 여기서 다시 고르지 않는다.
    saved = metrics.get("thresholds") or {}
    if not saved:
        raise SystemExit(
            "metrics.json에 thresholds가 없다 — dev에서 확정한 임계값 없이 채점하면 "
            "여기서 고르게 되고, 그게 곧 gold 오염이다. 재학습해서 임계값을 저장할 것"
        )
    thr = np.array([saved[a] for a in names], dtype=float)

    gold = load_ftc_gold(args.stratum)
    noisy = load_ftc_gold("noisy")
    logger.info(f"========== 외부 정답 채점 | {model_dir.name} ==========")

    # **지층과 상수를 머리에 항상 찍는다.** 기본값이 clean으로 바뀌었으므로, 같은 모델의
    # 같은 지표가 예전 리포트와 다른 값을 낸다. 세 줄이 있으면 값이 달라 보여도 즉시 갈린다.
    # `threshold_regimes()`에 세 값을 병기한 것과 같은 패턴이다.
    logger.info(f"  ★ stratum       {args.stratum} (n={len(gold)})  "
                f"— {_STRATUM_NOTE[args.stratum]}")
    logger.info(f"  ★ noisy 병기     n={len(noisy)} — 근거_법령 2개+. 의결서가 평균 2.89개 위반을 "
                f"서술했는데 파서가 조항을 1개만 찾은 구간(72/72=100%)이라 **채점 불가**. "
                f"파서를 고치면 복구 대상")
    logger.info(f"  체크포인트 선택 기준: {metrics.get('checkpoint_criterion', '?')}")
    logger.info(f"  임계값(dev에서 확정, 재튜닝 없음): "
                f"{ {a: round(t, 2) for a, t in zip(names, thr)} }")

    # 상수 기준선을 **모델보다 먼저** 찍는다 — 모델 점수를 본 뒤 재면 이미 기울어 있다.
    logger.info("  ----- 상수 라벨러 기준선 -----")
    baselines = {}
    for k in (1, 2, 3, 4):
        b = constant_baseline_f1(gold, names, k)
        baselines[f"top{k}"] = b
        logger.info(f"    항상 {b['articles']} → per-sample F1 {b['f1'] * 100:5.1f}%")
    best_base = max(baselines.values(), key=lambda b: b["f1"])
    logger.info(f"  ★ 구간 상수      {best_base['f1'] * 100:.1f}%  {best_base['articles']}  "
                f"← 이 지층의 값. 다른 지층 수치와 직접 비교하지 말 것")

    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    # `load()`를 쓴다 — 직접 조립하면 지문 검사를 건너뛴다. 이 파일이 예전에 인코더를
    # `strict=False`로 밀어넣어 **랜덤 가중치로 채점**했던 바로 그 자리다.
    model = ArticleMultiLabelElectra.load(model_dir).to(device).eval()

    ds = ArticleDataset(gold, tokenizer, args.max_len, names)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    P, Y = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                           batch["token_type_ids"].to(device))
            P.append(torch.sigmoid(logits).cpu().numpy())
            Y.append(batch["labels"].numpy())
    probs, labels = np.concatenate(P), np.concatenate(Y)

    m = per_article_metrics(probs, labels, names, thr)
    m["sample_f1"] = _sample_f1(probs, labels, thr)

    logger.info("  ----- 모델 -----")
    for a, v in m["per_article"].items():
        mark = "" if v["support"] else "   ← macro 제외(support 0)"
        logger.info(f"    {a:<6} F1 {v['f1'] * 100:5.1f}%  정밀 {v['precision'] * 100:5.1f}%  "
                    f"재현 {v['recall'] * 100:5.1f}%  (n={v['support']}){mark}")
    logger.info(f"    macro F1 {m['macro_f1'] * 100:.1f}% ({m['n_scored']}개 조) | "
                f"{_MAJORITY_ARTICLE} 제외 {m['macro_f1_excl_majority'] * 100:.1f}% | "
                f"per-sample F1 {m['sample_f1'] * 100:.1f}%")

    # 점수 차만 찍고 "이김/짐"을 단정하면 안 된다 — n=255에서 CI 반폭이 ±5%p 안팎이라
    # 점추정 +2.5%p는 미판정이다. 오늘 이 표에서 여러 번 헛읽은 지점이다.
    delta = (m["sample_f1"] - best_base["f1"]) * 100
    const_ps = _per_sample(
        np.tile(np.array([1.0 if a in best_base["articles"] else 0.0 for a in names]), (len(labels), 1)),
        labels, np.full(len(names), 0.5))
    d_dev, lo_dev, hi_dev = _paired_ci(_per_sample(probs, labels, thr), const_ps)
    v_dev = "이김" if lo_dev > 0 else ("미판정" if hi_dev > 0 else "못 이김")
    logger.info(f"  → 상수 최고({best_base['articles']}, F1 {best_base['f1'] * 100:.1f}%) 대비 "
                f"{delta:+.1f}%p [{lo_dev:+.1f},{hi_dev:+.1f}] — {v_dev}")

    # ----- 조항 단위 지표 (제품이 서는 축) -----
    from backend.training.train_article import exclude_gold_documents, load_article_records, split_negative_holdout
    _recs = exclude_gold_documents(load_article_records(PROJECT_ROOT / "data/fb_check/clean.jsonl"), gold)
    _, _neg = split_negative_holdout(_recs, 12, 42)
    _neg = [r for r in _neg if not r["articles"]]        # 정답이 빈 것만 = 오경보 판정 대상
    _nds = ArticleDataset([{"text": r["text"], "articles": [], "group": "g"} for r in _neg],
                          tokenizer, args.max_len, names)
    _NP = []
    with torch.no_grad():
        for b in torch.utils.data.DataLoader(_nds, batch_size=args.batch_size):
            _NP.append(torch.sigmoid(model(b["input_ids"].to(device), b["attention_mask"].to(device),
                                           b["token_type_ids"].to(device))).cpu().numpy())
    neg_probs = np.concatenate(_NP)
    cl = clause_level(probs, thr, neg_probs)
    m["clause_level"] = {
        "metrics": cl,
        "population": f"gold clean {len(gold)}(위반) + 표준계약서 holdout {len(_neg)}(비위반) "
                      f"— 조 단위 표와 평가셋이 다름",
        "r_independent": ["recall", "false_alarm"],
        "r_dependent": ["precision_at_r*"],
        "caveat_source_confound": "FTC/표준계약서 혼합에서 측정 — 출처와 정답이 완전히 상관한다. "
                                  "출처 판별 천장 96.9%보다 낮아 순수 지름길은 아니나 부분 기여 미배제",
    }
    logger.info("  ----- 조항 단위 (위반 조항을 짚었나 — 조가 틀려도 사용자에겐 유용) -----")
    logger.info(f"    ★ 모집단: gold clean {len(gold)}건(위반) + 표준계약서 holdout {len(_neg)}건(비위반) "
                f"— **위 조 단위 표({len(gold)}건)와 평가셋이 다르다**")
    # 재현·오경보는 r과 무관한 **모델 속성**, 정밀도는 **분포의 함수**다. 갈라서 낸다 —
    # 붙여 놓으면 "73%"만 떼어져 r 없이 혼자 돌아다닌다.
    logger.info(f"    {'τ':>6}{'재현':>8}{'GPT불일치':>11}   │ 정밀도(r 의존)  r=.05   r=.10   r=.15")
    for t, v in cl.items():
        if t == "dev_thresholds":
            continue
        logger.info(f"    {t:>6}{v['recall'] * 100:>7.1f}%{v['disagree_with_gpt'] * 100:>10.1f}%   │"
                    f"{v['precision_at_r0.05'] * 100:>19.0f}%{v['precision_at_r0.10'] * 100:>7.0f}%"
                    f"{v['precision_at_r0.15'] * 100:>7.0f}%")
    d = cl["dev_thresholds"]
    logger.info(f"    {'배포':>6}{d['recall'] * 100:>7.1f}%{d['disagree_with_gpt'] * 100:>10.1f}%   │"
                f"{d['precision_at_r0.05'] * 100:>19.0f}%{d['precision_at_r0.10'] * 100:>7.0f}%"
                f"{d['precision_at_r0.15'] * 100:>7.0f}%   ← 지금 배포하면 나오는 값")
    logger.info("    ⚠ **'GPT불일치'는 오경보율이 아니다.** 음성 풀의 정답이 GPT 라벨"
                "(forward ∩ verify) 그 자체이므로, 모델이 GPT보다 잘 찾아서 짚은 것도 여기 들어간다. "
                "독립 준거로 잰 오경보율은 **아직 없다** — 표준계약서 조항을 사람이 판단해야 한다")
    logger.info("    ⚠ 따라서 오른쪽 **정밀도 열도 절대값으로 인용하면 안 된다** — 분모에 이 값이 "
                "들어간다. 설정 간 상대 비교에는 쓸 수 있다(모두 같은 자로 쟀다)")
    logger.info("    ── 상수 ──")
    logger.info("    전부 표시        재현 100%  GPT불일치 100%   정밀 = r 그 자체")
    logger.info("    아무것도 안 표시   재현   0%  GPT불일치   0%   정밀 정의 불가")
    logger.info("    ★ 재현·오경보는 **r과 무관한 모델 속성**이라 r이 미확정이어도 보고할 수 있다. "
                "정밀도는 분포의 함수이므로 **r 없이 인용하지 말 것**")
    logger.info("    ⚠ 조항 단위는 FTC/표준계약서 **혼합**에서 측정했고, 이 혼합에서는 출처와 정답이 "
                "완전히 상관한다. 값이 출처 판별 천장(TF-IDF 96.9%)보다 낮아 순수 지름길은 "
                "아니나 **부분 기여는 배제하지 못했다**")
    logger.info("    ★ 조 단위 수치만 보고 판단하지 말 것 — 제품이 주장하는 것은 조항 지목이다")

    # ----- 임계값 체제 세 벌 -----
    reg = threshold_regimes(probs, labels, thr, [g["doc_id"] for g in gold])
    m["threshold_regimes"] = {k: v for k, v in reg.items() if k != "crossfit_per_sample"}
    if "note" in reg:
        logger.warning(f"  교차적합 생략: {reg['note']}")
    else:
        const = np.array([best_base["f1"]] * len(labels))     # 상수는 건별로 상수가 아니다 → 아래서 재계산
        const = _per_sample(np.tile(np.array([1.0 if a in best_base["articles"] else 0.0
                                              for a in names]), (len(labels), 1)),
                            labels, np.full(len(names), 0.5))
        cross = np.array(reg["crossfit_per_sample"])
        d = cross - const
        rng = np.random.default_rng(42)
        boot = np.sort([d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(5000)])
        lo, hi = boot[125], boot[4875]
        verdict = "상수를 이김" if lo > 0 else ("미판정" if hi > 0 else "못 이김")
        logger.info("  ----- 임계값 체제 (셋의 간격 = 보정 불일치 비용) -----")
        logger.info(f"    dev 임계값   {reg['dev'] * 100:5.1f}%   ← 배포하면 실제로 나오는 값 (보고값)")
        logger.info(f"    교차적합     {reg['crossfit'] * 100:5.1f}%   ← 분포 맞는 보정셋이 있을 때 "
                    f"(상수 대비 {(cross.mean() - const.mean()) * 100:+.1f}%p "
                    f"[{lo:+.1f},{hi:+.1f}] — {verdict})")
        logger.info(f"    오라클       {reg['oracle_diagnostic_only'] * 100:5.1f}%   "
                    f"← 상한. **진단용, 성능 주장 금지**")
        m["threshold_regimes"]["crossfit_vs_constant_pp"] = float((cross.mean() - const.mean()) * 100)
        m["threshold_regimes"]["crossfit_ci95_pp"] = [float(lo), float(hi)]

    if delta <= 0:
        logger.warning("  상수 라벨러를 못 넘었다(dev 임계값 기준). **먼저 교차적합 열을 볼 것** — "
                       "dev와 gold의 빈 라벨 비율이 정반대(67% vs 0%)라 dev 임계값은 gold에서 "
                       "구조적으로 보수적이다. 교차적합도 못 넘으면 그때 학습 데이터·라벨 정의를 "
                       "본다(프롬프트·하이퍼파라미터가 아니다)")

    save_json({
        "model_dir": str(model_dir),
        "stratum": args.stratum,
        "n_gold": len(gold),
        "n_noisy_excluded": len(noisy),
        "thresholds": {a: float(t) for a, t in zip(names, thr)},
        "threshold_source": "train_article dev split (재튜닝 없음)",
        "checkpoint_criterion": metrics.get("checkpoint_criterion"),
        "constant_baselines": baselines,
        "model": m,
        "vs_constant_pp": delta,
        "vs_constant_ci95_pp": [lo_dev, hi_dev],
        "beats_constant": bool(lo_dev > 0),
    }, OUT_PATH)
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
