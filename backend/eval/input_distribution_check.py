# backend/eval/input_distribution_check.py
"""
**학습 텍스트와 운영 텍스트가 같은 종류의 문자열인가.** 라벨 없이 지금 잴 수 있다.

## 남은 한 칸

`input_parity_eval`이 "원문 vs 조각"을 닫았다. 그런데 같은 "원문"에도 층이 하나 더 있다:

    학습·평가   r["text"]         FTC 의결서 **PDF에서 파서가 뽑은** 조항
    운영        state["clause"]   사용자가 **붙여넣은** 계약서를 split_clauses가 자른 것

PDF 추출물에는 `체 납된`·`과태 료` 같은 **단어 중간 공백**이 남는다. 붙여넣기에는 없다.
방향은 덜 위험하다(학습이 더 지저분한 쪽이라 운영이 더 깨끗한 입력을 받는다). 그래도
정합 여부를 본 적이 없다.

## 무엇으로 재나 — 라벨이 필요 없는 것만 쓴다

r 워크시트의 실제 약관 99건은 **라벨이 없다.** 그래서 정확도는 못 재지만, 아래는 잰다:

    표면 지표     1글자 토큰 비율 · 길이 · **256토큰 잘림률**   ← PDF 잔해와 절단이 드러난다
    분리 가능성   TF-IDF로 학습 텍스트 vs 실제 약관 판별       ← 눈금이 포화한다. 아래 참고
    모델 반응     지목률 · max 확률 분위수                     ← r과 교락. 아래 경고 참고
    길이 통제     4분위별 지목률 · 앞부분만 자른 사본           ← 길이가 지목을 만드는지

## 지목률 차이를 분포 이동으로 읽지 말 것

실제 약관의 지목률이 낮게 나와도 두 가지가 구분되지 않는다:

    (a) 입력 분포가 이동해 모델이 반응을 못 한다      ← 문제
    (b) 실제 약관에 진짜로 위반이 적다 (r이 낮다)     ← 정상. 그게 r을 재는 이유다

**둘을 가르는 것은 사람 판단 50건이다.** 여기서는 표면 지표와 분리 가능성으로
(a)의 여지를 좁힐 뿐이다 — 표면이 비슷하고 판별이 안 되면 (a)는 닫히고 낮은 지목률은
(b)의 증거가 된다. 그 순서로 읽을 것.

## 실측 결론 (2026-09-01)

PDF 잔해 걱정은 **거의 닫혔다.** 1글자 토큰 비율이 실제 약관 9.8% · 표준계약서 10.3%로
사실상 같다. 오히려 **FTC(양성 쪽)만 14.7%로 지저분하다** — 학습 안에서 표면 잡음이
정답과 약하게 상관한다는 뜻이라 따로 적어둔다(출처 교락의 문자 단위 판본).

대신 **다른 것이 걸렸다 — 절단이다.**

    집단              토큰 중앙   p90   >256 잘림
    ftc_gold            84      216     5.5%
    표준계약서           111      350    19.9%
    실제 약관 99        184      476    35.4%   ← 3분의 1이 잘린다

보고값 78.0%는 5.5%만 잘리는 집단에서 나왔다. **운영에서는 3분의 1이 뒷부분을 잃는다.**
그리고 뒷부분에 신호가 있다 — 실제 약관을 앞 147자(ftc 중앙 길이)로 자르면 지목률이
39.4% → 29.3%로 떨어진다.

실행:
    .venv/bin/python -m backend.eval.input_distribution_check models/article_v1 --gpu 1
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from backend.model.electra import ArticleMultiLabelElectra
from backend.training.train_article import (
    ArticleDataset,
    exclude_gold_documents,
    load_article_records,
    load_ftc_gold,
    split_negative_holdout,
)
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("input_distribution_check.log")

SHEET = Path(os.environ.get("PREVALENCE_SHEET",
                            str(PROJECT_ROOT / "data/eval/prevalence/worksheet.csv")))
CLEAN_PATH = Path(os.environ.get("ARTICLE_LABELS", str(PROJECT_ROOT / "data/fb_check/clean.jsonl")))
OUT_PATH = PROJECT_ROOT / "data/eval/input_distribution_report.json"

# PDF 추출 잔해의 지문: 단어가 공백으로 쪼개지면 **1글자 한글 토큰**이 급증한다
# (`체 납된` → `체`, `과태 료` → `료`). 붙여넣기 텍스트에는 조사·의존명사 정도만 나온다.
_HANGUL_1 = re.compile(r"(?<![가-힣])[가-힣](?![가-힣])")
_TOKEN = re.compile(r"[가-힣A-Za-z0-9]+")


def _surface(texts: list[str]) -> dict:
    """라벨이 필요 없는 표면 지표. PDF 잔해가 여기서 직접 드러난다."""
    one_char, lens, tok_n = [], [], []
    for t in texts:
        toks = _TOKEN.findall(t)
        tok_n.append(len(toks))
        lens.append(len(t))
        one_char.append(sum(1 for x in toks if len(x) == 1 and _HANGUL_1.fullmatch(x)) / max(1, len(toks)))
    return {
        "n": len(texts),
        "len_median": float(np.median(lens)) if lens else 0.0,
        "len_p90": float(np.percentile(lens, 90)) if lens else 0.0,
        "one_char_token_rate": float(np.mean(one_char)) if one_char else 0.0,
        "one_char_p90": float(np.percentile(one_char, 90)) if one_char else 0.0,
        "token_median": float(np.median(tok_n)) if tok_n else 0.0,
    }


def _separability(a: list[str], b: list[str], seed: int = 42) -> dict:
    """TF-IDF + 로지스틱으로 두 집단을 가른다. **AUC가 곧 분포 이동의 크기다.**

    0.5면 구분 불가(같은 분포), 1.0이면 완전히 다른 글이다.

    **이 자는 여기서 포화한다.** 실측에서 세 쌍이 전부 0.995~1.000이 나왔고, 그중 하나는
    **학습에 함께 쓰는 두 코퍼스(ftc vs 표준계약서)**다. 즉 "AUC가 높다"가 "운영 분포가
    이탈했다"를 뜻하지 않는다 — 문서 수십 개짜리 코퍼스 둘은 회사명·서식만으로도 갈린다.
    보정선(세 번째 줄)이 같이 나오지 않으면 이 수치를 인용하지 말 것.
    """
    if len(a) < 20 or len(b) < 20:
        return {"note": f"표본 부족 ({len(a)}, {len(b)})"}
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    X = a + b
    y = np.array([0] * len(a) + [1] * len(b))
    pipe = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return {"auc": float(roc_auc_score(y, p)), "n": [len(a), len(b)]}


def _truncation(texts: list[str], tokenizer, max_len: int) -> dict:
    """`max_len`에서 잘리는 비율. **보고값과 운영이 갈리는 지점이 여기다.**"""
    n = np.array([len(tokenizer(t)["input_ids"]) for t in texts])
    return {"token_median": float(np.median(n)), "token_p90": float(np.percentile(n, 90)),
            "truncated_rate": float((n > max_len).mean())}


def _severity(texts: list[str], tokenizer, max_len: int = 256) -> dict:
    """절단 **정도**. 절단률(몇 %가 잘리나)과 다른 질문이다 — 얼마나 잘리나.

    `max_len` 개입의 무해 확인을 gold 절단분(n=18)에서 하고 "놓침 0"을 얻었는데,
    **그 18건은 전부 살짝 잘린 것들이었다.** gold 절단분의 최악 손실이 87토큰(25.4%)인데
    실제 약관 절단분의 60%가 그보다 심하다. 같은 개입이지만 되찾아주는 양이 다르므로,
    0/18이 덮는 것은 "살짝 잘린 것을 되찾아도 놓침이 안 생긴다"까지다.

    **n이 아니라 구간이 한계다.** 이 표를 0/18 옆에 항상 붙일 것.
    """
    n = np.array([len(tokenizer(t)["input_ids"]) for t in texts])
    cut = n[n > max_len]
    if not len(cut):
        return {"n_truncated": 0}
    lost = (cut - max_len) / cut * 100
    return {"n_truncated": int(len(cut)),
            "tok_median": float(np.median(cut)), "tok_max": float(cut.max()),
            "lost_pct_median": float(np.median(lost)), "lost_pct_max": float(lost.max()),
            "lost_tok_max": int((cut - max_len).max()),
            "over_512": int((cut > 512).sum())}


def _probs(texts: list[str], model, tokenizer, device, names, bs: int, max_len: int) -> np.ndarray:
    recs = [{"text": t, "articles": [], "group": "g"} for t in texts]
    ds = ArticleDataset(recs, tokenizer, max_len, names)
    P = []
    with torch.no_grad():
        for b in torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False):
            P.append(torch.sigmoid(model(b["input_ids"].to(device), b["attention_mask"].to(device),
                                         b["token_type_ids"].to(device))).cpu().numpy())
    return np.concatenate(P) if P else np.zeros((0, len(names)))


def main() -> None:
    ap = argparse.ArgumentParser(description="학습 텍스트 vs 운영 텍스트 분포 비교 (라벨 불필요)")
    ap.add_argument("model_dir")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    names = metrics["article_names"]
    thr = np.array([(metrics.get("thresholds") or {})[a] for a in names], dtype=float)

    gold = load_ftc_gold("clean")
    recs = exclude_gold_documents(load_article_records(CLEAN_PATH), gold)
    _, neg = split_negative_holdout(recs, 12, 42)
    with open(SHEET, encoding="utf-8-sig") as f:
        real = [r["clause_text"].strip() for r in csv.DictReader(f) if r.get("clause_text", "").strip()]

    pops = {
        "ftc_gold (PDF 파서)":      [g["text"] for g in gold],
        "표준계약서 holdout":        [r["text"] for r in neg if not r["articles"]],
        "실제 약관 99 (붙여넣기)":   real,
    }

    logger.info(f"========== 입력 분포 비교 | {model_dir.name} ==========")
    logger.info("  ----- 표면 지표 (라벨 불필요 · r과 무관) -----")
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    logger.info(f"  {'집단':<26}{'n':>5}{'1글자 토큰':>11}{'토큰 중앙':>10}{'p90':>7}{'>' + str(args.max_len) + ' 잘림':>10}")
    surf = {}
    for k, v in pops.items():
        s = surf[k] = {**_surface(v), **_truncation(v, tokenizer, args.max_len)}
        logger.info(f"  {k:<26}{s['n']:>5}{s['one_char_token_rate'] * 100:>10.1f}%"
                    f"{s['token_median']:>10.0f}{s['token_p90']:>7.0f}{s['truncated_rate'] * 100:>9.1f}%")
    logger.info("    1글자 토큰 = PDF 추출이 단어를 쪼갠 흔적(`체 납된`→`체`). 붙여넣기엔 조사 정도만 남는다")
    logger.info("    ★ **잘림률이 진짜 격차다** — 보고값 78.0%는 5%만 잘리는 집단에서 나왔는데 "
                "운영 입력은 3분의 1이 뒷부분을 잃는다")

    # ----- 절단 **정도**. 절단률과 다른 질문이다 -----
    logger.info("  ----- 절단 정도 (얼마나 잘리나 — 절단률과 다른 질문) -----")
    logger.info(f"  {'집단':<26}{'절단':>6}{'토큰중앙':>9}{'최대':>7}{'손실중앙':>9}{'손실최대':>9}{'>512':>7}")
    sev = {}
    for k, v in pops.items():
        e = sev[k] = _severity(v, tokenizer)
        if not e["n_truncated"]:
            logger.info(f"  {k:<26}{0:>6}   (절단 없음)")
            continue
        logger.info(f"  {k:<26}{e['n_truncated']:>6}{e['tok_median']:>9.0f}{e['tok_max']:>7.0f}"
                    f"{e['lost_pct_median']:>8.1f}%{e['lost_pct_max']:>8.1f}%{e['over_512']:>7}")
    g, r = sev.get("ftc_gold (PDF 파서)", {}), sev.get("실제 약관 99 (붙여넣기)", {})
    if g.get("n_truncated") and r.get("n_truncated"):
        rn = np.array([len(tokenizer(t)["input_ids"]) for t in real])
        beyond = int(((rn[rn > 256] - 256) > g["lost_tok_max"]).sum())
        logger.info(f"    ★ gold 절단분의 최악 손실은 {g['lost_tok_max']}토큰({g['lost_pct_max']:.1f}%)이고 "
                    f"**그보다 심한 gold 레코드는 0건**이다. 실제 약관 절단분은 "
                    f"{beyond}/{r['n_truncated']}건({beyond / r['n_truncated'] * 100:.0f}%)이 그 밖에 있다")
        logger.info("    → gold에서 얻은 무해 확인(놓침 0/18)이 덮는 것은 **살짝 잘린 구간까지**다. "
                    "n이 아니라 **구간**이 한계다")

    logger.info("  ----- 분리 가능성 (TF-IDF char 2-4gram, 5-fold CV AUC) -----")
    sep = {
        "ftc_gold vs 실제약관":     _separability(pops["ftc_gold (PDF 파서)"], real),
        "표준계약서 vs 실제약관":    _separability(pops["표준계약서 holdout"], real),
        "ftc_gold vs 표준계약서":    _separability(pops["ftc_gold (PDF 파서)"], pops["표준계약서 holdout"]),
    }
    for k, v in sep.items():
        logger.info(f"    {k:<26}{'AUC ' + format(v['auc'], '.3f') if 'auc' in v else v['note']}")
    logger.info("    ⚠ **세 줄이 다 0.99+면 이 자는 포화한 것이다.** 세 번째 줄은 학습에 함께 쓰는 "
                "두 코퍼스인데도 0.995다 — 문서 수십 개짜리 코퍼스는 회사명·서식만으로 갈린다. "
                "여기서 '운영 분포가 이탈했다'를 읽어내지 말 것")

    model = ArticleMultiLabelElectra.load(model_dir).to(device).eval()

    logger.info("  ----- 모델 반응 (⚠ r과 교락 — 아래 경고 먼저 읽을 것) -----")
    logger.info(f"  {'집단':<26}{'지목률':>9}{'max확률 중앙':>13}{'p75':>8}{'p90':>8}{'평균 조 수':>11}")
    resp = {}
    for k, v in pops.items():
        P = _probs(v, model, tokenizer, device, names, args.batch_size, args.max_len)
        mx = P.max(1)
        flag = (P >= thr).any(1)
        resp[k] = {"flag_rate": float(flag.mean()), "max_p50": float(np.median(mx)),
                   "max_p75": float(np.percentile(mx, 75)), "max_p90": float(np.percentile(mx, 90)),
                   "articles_per_flagged": float((P >= thr).sum(1)[flag].mean()) if flag.any() else 0.0}
        r = resp[k]
        logger.info(f"  {k:<26}{r['flag_rate'] * 100:>8.1f}%{r['max_p50']:>13.3f}"
                    f"{r['max_p75']:>8.3f}{r['max_p90']:>8.3f}{r['articles_per_flagged']:>11.2f}")

    # ----- 길이 통제. 지목률 차이가 길이 때문인지 내용 때문인지 가른다 -----
    logger.info("  ----- 길이 통제 (지목률이 길이의 부산물인가) -----")
    for k, v in pops.items():
        L = np.array([len(t) for t in v])
        P = _probs(v, model, tokenizer, device, names, args.batch_size, args.max_len)
        flag = (P >= thr).any(1)
        bins = np.digitize(L, np.quantile(L, [.25, .5, .75]))
        cells = "  ".join(f"Q{i + 1} {flag[bins == i].mean() * 100:5.1f}%(n={(bins == i).sum():>3})"
                          for i in range(4))
        resp[k]["by_length_quartile"] = [float(flag[bins == i].mean()) for i in range(4)]
        logger.info(f"    {k:<24}{cells}")
    logger.info("    → 사분위 간 평탄하면 길이가 지목을 만들어내는 게 아니다(실측: 평탄)")

    cut = int(np.median([len(t) for t in pops["ftc_gold (PDF 파서)"]]))
    P_cut = _probs([t[:cut] for t in real], model, tokenizer, device, names, args.batch_size, args.max_len)
    resp["실제 약관 99 (붙여넣기)"]["flag_rate_head_only"] = float((P_cut >= thr).any(1).mean())
    logger.info(f"    실제 약관을 앞 {cut}자(ftc 중앙 길이)만 남기면 "
                f"{resp['실제 약관 99 (붙여넣기)']['flag_rate'] * 100:.1f}% → "
                f"{(P_cut >= thr).any(1).mean() * 100:.1f}%  "
                f"← **뒷부분에 신호가 있다.** 잘림이 공짜가 아니라는 뜻")

    logger.info("    ⚠ 실제 약관의 지목률이 낮다고 '분포 이동'으로 읽지 말 것 — **r이 낮아서**일 수 있다. "
                "둘을 가르는 것은 사람 판단 50건이다. 여기서는 표면·분리 가능성으로 "
                "분포 이동의 여지를 좁힐 뿐이다")

    save_json({"model_dir": str(model_dir), "surface": surf, "separability": sep, "model_response": resp,
               "note": "실제 약관 99건은 라벨이 없다. 표면·분리 가능성은 r과 무관하고, 지목률은 r과 교락한다",
               "read_order": "① 표면이 비슷한가 → ② 판별이 되는가 → ③ 그러고 나서 지목률"},
              OUT_PATH)
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
