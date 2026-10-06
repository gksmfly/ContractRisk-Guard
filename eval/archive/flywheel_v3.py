# backend/eval/flywheel_v3.py
"""**모델이 세 번째 투표자였다면 나쁜 라벨을 골라냈을까.** 학습 노출로 층화해서 본다.

## 왜 v3인가

원본 Flywheel 측정은 `models/v4`(risk 3단계)가 3-way 다수결에 투표한 결과였다
(제외 17건 · 유지 114건, +19.6%p [-0.5,+38.0] 미판정 · MW p=0.0306). v4는 조 헤드가
없어 조 multi-label에 투표할 방법 자체가 없으므로 **재현이 아니라 새 측정**이다.
원본 수치는 지우지 않고 병기한다.

또한 **현행 파이프라인에는 모델 투표가 없다** — 판정이 `L == L'`(forward == verify)이라
CLEAN 3,352건 중 forward≠verify가 0건이다. 즉 여기서 재는 것은 실제로 돌아가는 기구가
아니라 **"모델을 투표자로 넣었다면"이라는 반사실**이다.

## ⚠ 첫 판(2026-09-06 오전)은 철회됐다 — 규칙이 결과를 만들었다

제외 규칙을 **비대칭**으로 썼다: `모델 조 ∩ GPT 조 == 공집합`이면 제외.
**GPT 라벨이 빈 레코드는 무엇과 교집합해도 공집합이라 항상 제외 집단에 들어간다.**
그리고 준거(FTC 근거_법령)는 비어있지 않으므로 그 건들의 F1은 **구성상 0점**이다.

    gold문서 층 제외 55건 중 30건(55%)이 GPT 빈 라벨 → 자동 제외 + 자동 0점

즉 +36.7%p는 라벨 품질 신호가 아니라 **규칙의 산술**이었다. 08-31에 이미
"3-way 교집합 요구가 빈 라벨을 쓸어버린다"로 기각했던 규칙이 재도입된 것이다
(`noise_reason_audit`의 빈 배열 조건과 같은 함정).

**대칭 규칙으로 고쳤다: 양쪽이 다 비면 일치로 본다.** 그리고 GPT 빈 라벨 건은
**근거로 집계하지 않고 별도 층으로 분리 출력**한다 — `article_gold_eval`이 임계값
재튜닝을 거부하는 것과 같은 패턴으로, 코드가 규칙을 강제한다.

## 학습 노출이 결과를 만든다 — 그래서 층화한다

모델은 `agreed` 라벨로 학습했고 CLEAN의 상당수가 학습셋이다:

    학습에서 본 레코드   그 라벨을 외웠으므로 교집합이 안 빈다  → 유지 집단
    못 본 레코드        일반화 실패하면 교집합이 빈다         → 제외 집단

그러면 "제외 집단의 라벨이 나쁘다"가 **"모델이 나쁜 라벨을 골라냈다"인지 "모델이 안 본
레코드라 못 맞혔다"인지 구분되지 않는다.** 논문의 Flywheel 주장이 원래 걸렸던 지점이다 —
자기 산출물로 학습한 모델은 그 산출물에 동의하도록 훈련됐다.

    효과가 dev(미학습) 층에도 있다   →  실제로 나쁜 라벨을 골라낸다. 성립
    train 층에만 있다               →  암기의 그림자. 성립 안 함

⚠ dev도 완전 미노출은 아니다 — 임계값과 체크포인트를 dev로 골랐다. 다만 경사 갱신은
없으므로 train보다 훨씬 약한 노출이다.

## 사전 등록된 판정 규칙 (2026-09-01, 그대로 적용)

    주 검정   평균차 부트스트랩   "제외 집단의 평균 품질이 낮은가"
    보조     Mann-Whitney       유계·양극 분포에 강건. 참고로만
    판정은 주 검정으로 한다. 둘이 갈리면 "주 검정 기준 미판정, 보조는 유의"로 적고
    결론을 내지 않는다.

실행:
    .venv/bin/python -m eval.flywheel_v3 --gpu 1
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from backend.labeling.articles import normalize
from backend.model.electra import ArticleMultiLabelElectra
from backend.training.train import split_by_document
from backend.training.train_article import (
    ArticleDataset,
    exclude_gold_documents,
    load_article_records,
    load_ftc_gold,
    select_article_labels,
    split_negative_holdout,
)
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("flywheel_v3.log")
OUT_PATH = PROJECT_ROOT / "data/eval/flywheel_v3_split.json"
# 철회된 비대칭 규칙의 산출은 `flywheel_v3_withdrawn.json`에 그대로 둔다 —
# 덮으면 "그때 무엇이 계산됐는지"가 사라진다(측정 로그를 안 고치는 것과 같은 기준).


def _f1(pred: set, ref: set) -> float:
    if not pred and not ref:
        return 1.0
    i = len(pred & ref)
    p = i / len(pred) if pred else 0.0
    r = i / len(ref) if ref else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def _mean_diff_ci(a: np.ndarray, b: np.ndarray, seed: int = 42) -> tuple[float, float, float]:
    """유지(a) − 제외(b). 양수면 '제외 집단의 품질이 낮다' = flywheel 주장과 같은 방향."""
    rng = np.random.default_rng(seed)
    boot = np.sort([a[rng.integers(0, len(a), len(a))].mean()
                    - b[rng.integers(0, len(b), len(b))].mean() for _ in range(5000)])
    return (a.mean() - b.mean()) * 100, boot[125] * 100, boot[4875] * 100


def main() -> None:
    ap = argparse.ArgumentParser(description="Flywheel v3 — 학습 노출로 층화한 반사실 측정")
    ap.add_argument("--model-dir", default="models/article_v2")
    ap.add_argument("--gpu", type=int, default=1)
    a = ap.parse_args()

    cases = json.loads((PROJECT_ROOT / "data/raw/ftc_cases/ftc_cases_parsed.json")
                       .read_text(encoding="utf-8"))["사례"]
    ref: dict[str, set[str]] = {}
    meta: dict[str, tuple[int, int]] = {}          # doc -> (조항 수, 근거_법령 수)
    for c in cases:
        doc = str((c.get("셀_데이터") or {}).get("사건번호") or c.get("사건명", ""))
        arts = {x for g in (c.get("근거_법령") or []) if (x := normalize(str(g)))}
        if doc and arts:
            ref[f"ftc_case:{doc}"] = arts
            meta[f"ftc_case:{doc}"] = (len(c.get("조항_원문") or []), len(arts))

    gold = load_ftc_gold("clean")
    pool, _ = split_negative_holdout(
        exclude_gold_documents(load_article_records(PROJECT_ROOT / "data/fb_check/clean.jsonl"), gold),
        12, 42)
    names0 = select_article_labels(dict(Counter(x for r in pool for x in r["articles"])), 5)
    for r in pool:
        r["has_article"] = bool([x for x in r["articles"] if x in names0])
    tr, dev = split_by_document(pool, 0.2, 42, stratify_key="has_article")
    # 세 번째 층: gold 문서. 가중치·체크포인트·임계값 **전부** 미노출이라 dev의 잔여 노출
    # (임계값을 dev의 agreed 라벨에 F1 최대로 맞췄다)이 없다.
    _seen = {r["text"] for r in tr} | {r["text"] for r in dev}
    excl = [r for r in load_article_records(PROJECT_ROOT / "data/fb_check/clean.jsonl")
            if r["text"] not in _seen]

    md = Path(a.model_dir)
    m = json.loads((md / "metrics.json").read_text(encoding="utf-8"))
    names = m["article_names"]
    thr = np.array([m["thresholds"][x] for x in names])
    device = torch.device(f"cuda:{a.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    tok = AutoTokenizer.from_pretrained(str(md))
    mdl = ArticleMultiLabelElectra.load(md).to(device).eval()

    logger.info(f"========== Flywheel v3 | {md.name} ==========")
    logger.info("  ★ 반사실 측정이다 — 현행 판정은 L==L'이라 모델이 투표하는 자리가 없다")
    logger.info("  ★ 사전 등록: 주=평균차 부트스트랩 · 보조=Mann-Whitney. 갈리면 미판정")

    report = {}
    for label, rows in (("train (가중치 학습됨)", tr),
                        ("dev (가중치 미학습·임계값 노출)", dev),
                        ("gold문서 (전부 미노출)", excl)):
        rs = [r for r in rows if r.get("source") == "ftc_case" and r["group"] in ref]
        if len(rs) < 30:
            logger.warning(f"  {label}: {len(rs)}건 — 30건 미만이라 판정하지 않는다")
            continue
        ds = ArticleDataset([{"text": r["text"], "articles": [], "group": "g"} for r in rs],
                            tok, m["train_config"]["max_len"], names)
        P = []
        with torch.no_grad():
            for b in torch.utils.data.DataLoader(ds, batch_size=32):
                P.append(torch.sigmoid(mdl(b["input_ids"].to(device), b["attention_mask"].to(device),
                                           b["token_type_ids"].to(device))).cpu().numpy())
        pred = np.concatenate(P) >= thr

        # **3분할.** ①은 규칙이 자동으로 제외 집단에 넣고 구성상 0점을 주므로
        # 근거로 집계하지 않는다 — 비교는 ②(유지) vs ③(모델이 실제로 반대) 뿐이다.
        #
        #   ① GPT 라벨이 빔      제외 여부를 규칙이 강제한다. **집계 제외**
        #   ② 교집합 있음        모델이 GPT에 동의 → 유지
        #   ③ 둘 다 비지 않았는데 교집합 없음 → 모델이 **실제로** 반대한 건
        #
        # 대칭성: 모델도 비고 GPT도 비면 "일치"다(②). 비대칭 규칙이 첫 판을 망쳤다.
        rowsq, empty, silent = [], [], []
        for r, pr in zip(rs, pred):
            mine = {x for x, v in zip(names, pr) if v}
            gptl = set(r["articles"])
            nc, na = meta[r["group"]]
            q = _f1(gptl, ref[r["group"]])
            if not gptl:
                empty.append(q)                       # ① 규칙 산물. 어느 쪽 근거도 아니다
            elif not mine:
                silent.append((q, nc, na))            # ② 난이도 가설. 별도 보고
            else:
                rowsq.append((q, nc, na, bool(mine & gptl)))   # ③ 이것만 flywheel 근거

        def _run(sub, tag):
            keep = np.array([q for q, _, _, k in sub if k])
            drop = np.array([q for q, _, _, k in sub if not k])
            if len(drop) < 10 or len(keep) < 10:
                logger.warning(f"    {tag}: 소집단 부족 (유지 {len(keep)} · 제외 {len(drop)})")
                return None
            d, lo, hi = _mean_diff_ci(keep, drop)
            from scipy.stats import mannwhitneyu
            _, pv = mannwhitneyu(keep, drop, alternative="greater")
            v = "제외 집단 품질이 낮다(유의)" if lo > 0 else ("미판정" if hi > 0 else "반대 방향")
            logger.info(f"    {tag:<18} 유지 {len(keep):>4}건 {keep.mean() * 100:5.1f}%  "
                        f"제외 {len(drop):>4}건 {drop.mean() * 100:5.1f}%  "
                        f"{d:+6.1f}%p [{lo:+.1f},{hi:+.1f}]  MW p={pv:.4f}  {v}")
            return {"keep": len(keep), "drop": len(drop), "keep_f1": float(keep.mean()),
                    "drop_f1": float(drop.mean()), "diff_pp": d, "ci95_pp": [lo, hi],
                    "mw_p": float(pv), "verdict": v}

        logger.info(f"  ----- {label} (n={len(rs)}) -----")
        logger.info(f"    [① GPT 빈 라벨 {len(empty)}건 · 조 F1 "
                    f"{np.mean(empty) * 100 if empty else 0:.1f}% — **집계에서 뺀다.** "
                    f"규칙이 자동 제외하고 준거가 비지 않아 구성상 0점이다]")
        if silent:
            sq = np.array([q for q, _, _ in silent])
            k0 = np.array([q for q, _, _, k in rowsq if k])
            d2 = _mean_diff_ci(k0, sq) if len(sq) >= 10 and len(k0) >= 10 else None
            logger.info(f"    [② GPT 라벨 있으나 모델 침묵 {len(silent)}건 · 조 F1 {sq.mean() * 100:.1f}%"
                        + (f" · 유지 대비 {d2[0]:+.1f}%p [{d2[1]:+.1f},{d2[2]:+.1f}]" if d2 else "")
                        + " — **난이도 가설. flywheel 근거 아님**]")
        # **준거 잡음이 두 집단에 균등한가.** 제외 쪽에 다조항 사건이 몰려 있으면
        # 그 집단의 F1이 라벨 품질이 아니라 **귀속 불가** 때문에 낮아진다.
        kk = [(nc, na) for _, nc, na, k in rowsq if k]
        dd = [(nc, na) for _, nc, na, k in rowsq if not k]
        f_k = sum(1 for nc, _ in kk if nc == 1) / max(1, len(kk))
        f_d = sum(1 for nc, _ in dd if nc == 1) / max(1, len(dd))
        logger.info(f"    [균등성] 조항1개 비율  유지 {f_k * 100:4.1f}%  제외 {f_d * 100:4.1f}%   "
                    f"근거수 중앙  유지 {np.median([a for _, a in kk]):.0f}  "
                    f"제외 {np.median([a for _, a in dd]):.0f}")
        res = {"n": len(rs), "n_empty_excluded": len(empty),
               "single_clause_rate": [f_k, f_d], "n_silent": len(silent), "all": _run(rowsq, "③ 선별 후보")}
        clean = [x for x in rowsq if x[1] == 1 and x[2] == 1]      # gold clean과 같은 조건
        res["clean_only"] = _run(clean, f"clean조건({len(clean)})")
        report[label] = res

    logger.info("  ----- 해석 -----")
    logger.info("    dev 층에도 효과가 있으면 flywheel 성립. train 층에만 있으면 암기의 그림자")
    logger.info("    ⚠ dev도 완전 미노출은 아니다 — 임계값·체크포인트를 dev로 골랐다(경사 갱신은 없음)")
    logger.info("    ※ v4 기반 원본(제외 17 · 유지 114, +19.6%p [-0.5,+38.0], MW p=0.0306)은 병기용")
    save_json({"model": str(md), "strata": report,
               "note": "반사실 측정. 현행 파이프라인에 모델 투표는 없다",
               "preregistered": "주=평균차 부트스트랩, 보조=Mann-Whitney, 갈리면 미판정"}, OUT_PATH)
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
