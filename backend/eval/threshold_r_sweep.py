# backend/eval/threshold_r_sweep.py
"""배포 임계값 — **위반 유병률 r을 고르지 않고 축으로 낸다.**

## 왜 이게 남았나

보정 격차가 유일한 미해결 병목이다. clean gold에서:

    교차적합 임계값   +6.2%p [+0.9,+11.7]   상수를 이김
    dev 임계값(배포)  +2.5%p [-3.7, +9.2]   미판정
                     ─────────────────────
                     차이 3.9%p = 보정 불일치

배포 분포를 닮은 보정셋이 필요한데 가진 것 셋이 다 아니다(dev 빈 라벨 67% / gold 0% /
표준계약서 holdout 97%). 의결서 대조표를 조사했으나 **유병률 1.0%, 근거_법령 1개 후보
0/6으로 재료가 없다**(`comparison_table_survey`). 남은 정직한 수는 이것뿐이다.

## r을 고르지 않는다

임의의 r 하나로 보고하면 방어할 수 없다("왜 그 값인가"). 대신 r을 훑어 곡선으로 낸다 —
독자가 자기 값을 대입할 수 있고, **우리 시스템이 쓸 만해지는 r 구간**도 함께 나온다.

## 부분표집이 아니라 가중 평가

r마다 물리적으로 섞으면 풀 크기에 묶인다:

    r=0.05 → 비위반 156 : 위반 8건    ← 위반이 한 자릿수. 그런데 낮은 r이 배포 현실이다

가중 평가는 모든 r에서 411건을 다 쓴다:

    combined(τ, r) = r · (위반 풀 255건 지표) + (1−r) · (비위반 풀 156건 지표)

**가정**: 두 풀이 각각 위반/비위반 조항을 대표하고 **유병률만 다르다.** 부분표집과 같은
가정을 쓰면서 추정만 효율적이다. 이 가정이 틀리면(예: 실제 계약서의 위반 조항이 FTC
의결 조항과 문체가 다르면) 곡선 전체가 이동한다 — 그 경우 실제 계약서 소량 라벨링이
다음 수다.

## 목적함수를 명시한다 — 이건 제품 판단이다

    F1-max        두 오류를 대칭으로 본다
    recall_floor  **위반 쪽**에 하한을 걸고 그 안에서 최대화 — 놓침을 더 나쁘게 본다

처음엔 두 번째를 "비위반 쪽 하한(≥0.90)"으로 잡았는데 F1 최적해가 이미 만족해서 두
곡선이 완전히 같았다. 구속력 없는 제약이었다. 아래 표가 보여주듯 **낮은 r에서 최적
τ가 침묵으로 수렴하므로**, 갈라야 할 축은 "위반을 최소 얼마나 잡을 것인가"다.

`out_of_scope`에서 "거짓 안심이 누락보다 나쁘다"로 정했지만 **그건 범위 밖 표시에 대한
판단**이었고 위험 조항 탐지는 반대일 수 있다. 지금 고르지 않고 **두 벌을 낸다** — 그
선택이 결과를 얼마나 바꾸는지가 그 자체로 정보다.

## τ 튜닝과 평가를 분리한다

같은 411건에서 맞추고 채점하면 낙관 편향이다. 두 풀을 각각 문서 단위 2-fold로 갈라
한쪽에서 τ를 맞추고 반대쪽에서 채점한다.

실행:
    .venv/bin/python -m backend.eval.threshold_r_sweep --model-dir models/_article_rNone --gpu 1
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from backend.model.electra import ArticleMultiLabelElectra
from backend.training.train_article import (ArticleDataset, exclude_gold_documents,
                                            load_article_records, load_ftc_gold,
                                            split_negative_holdout)
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("threshold_r_sweep.log")
OUT_PATH = PROJECT_ROOT / "data/eval/threshold_r_sweep.json"
_GRID = np.arange(0.05, 0.96, 0.01)
# **위반 쪽**에 하한을 건다. 처음엔 비위반 쪽에 걸었는데(≥0.90) F1 최적해가 이미
# 그걸 만족해서 두 곡선이 완전히 같게 나왔다 — 구속력 없는 제약이었다.
# 실제로 갈리는 축은 "위반을 최소 얼마나 잡을 것인가"다.
_VIOLATION_FLOOR = 0.38


def _probs(model_dir: str, texts: list[str], device) -> tuple[np.ndarray, list[str]]:
    m = ArticleMultiLabelElectra.load(Path(model_dir)).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    recs = [{"text": t, "articles": [], "group": "g"} for t in texts]
    out = []
    with torch.no_grad():
        for b in DataLoader(ArticleDataset(recs, tok, 256, m.article_names), batch_size=32):
            out.append(torch.sigmoid(m(input_ids=b["input_ids"].to(device),
                                       attention_mask=b["attention_mask"].to(device),
                                       token_type_ids=b["token_type_ids"].to(device))).cpu().numpy())
    return np.vstack(out), m.article_names


def _f1_rows(P: np.ndarray, names: list[str], golds: list[frozenset], tau: float) -> np.ndarray:
    """건별 F1 배열. 둘 다 비면 1.0 — '위반 없음'을 맞힌 것도 맞힌 것이다."""
    out = np.empty(len(golds))
    for i, g in enumerate(golds):
        pred = {n for n, v in zip(names, P[i]) if v >= tau}
        if not pred and not g:
            out[i] = 1.0
        elif not pred or not g:
            out[i] = 0.0
        else:
            inter = len(pred & g)
            out[i] = 2 * inter / (len(pred) + len(g))
    return out


def _tune(pv: np.ndarray, pn: np.ndarray, r: float, objective: str) -> float:
    """combined(τ, r)를 최대화하는 전역 τ. pv/pn은 τ별 건별 점수 행렬(그리드 × 건)."""
    viol, non = pv.mean(1), pn.mean(1)
    combined = r * viol + (1 - r) * non
    if objective == "recall_floor":
        ok = viol >= _VIOLATION_FLOOR
        if ok.any():
            combined = np.where(ok, combined, -np.inf)
    return float(_GRID[int(np.argmax(combined))])


def main() -> None:
    ap = argparse.ArgumentParser(description="배포 위반 유병률 r 스윕")
    ap.add_argument("--model-dir", default="models/_article_rNone")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")

    gold = load_ftc_gold("clean")
    recs = exclude_gold_documents(load_article_records(PROJECT_ROOT / "data/fb_check/clean.jsonl"), gold)
    _, neg = split_negative_holdout(recs, 12, 42)
    Gv = [frozenset(g["articles"]) for g in gold]
    Gn = [frozenset(r["articles"]) for r in neg]

    logger.info(f"========== 배포 임계값 r 스윕 | {a.model_dir} ==========")
    logger.info(f"  위반 풀   FTC clean gold {len(Gv)}건 (빈 라벨 {sum(1 for g in Gv if not g) / len(Gv) * 100:.1f}%)")
    logger.info(f"  비위반 풀 표준계약서 holdout {len(Gn)}건 (빈 라벨 {sum(1 for g in Gn if not g) / len(Gn) * 100:.1f}%)")
    logger.info("  가중 평가 — r을 바꿔도 표본은 411건 그대로, 가중치만 바뀐다")

    Pv, names = _probs(a.model_dir, [g["text"] for g in gold], device)
    Pn, _ = _probs(a.model_dir, [r["text"] for r in neg], device)
    # 그리드 × 건 점수 행렬을 한 번만 만든다
    Sv = np.vstack([_f1_rows(Pv, names, Gv, t) for t in _GRID])
    Sn = np.vstack([_f1_rows(Pn, names, Gn, t) for t in _GRID])

    # 문서 단위 2-fold — τ를 맞춘 쪽과 채점하는 쪽을 분리한다
    rng = random.Random(a.seed)
    dv = sorted({g["doc_id"] for g in gold}); rng.shuffle(dv)
    dn = sorted({r["group"] for r in neg});   rng.shuffle(dn)
    hv, hn = set(dv[:len(dv) // 2]), set(dn[:len(dn) // 2])
    Av = np.array([i for i, g in enumerate(gold) if g["doc_id"] in hv])
    Bv = np.array([i for i in range(len(gold)) if i not in set(Av)])
    An = np.array([i for i, r in enumerate(neg) if r["group"] in hn])
    Bn = np.array([i for i in range(len(neg)) if i not in set(An)])
    logger.info(f"  교차적합 fold — 위반 {len(Av)}/{len(Bv)} · 비위반 {len(An)}/{len(Bn)}")

    # "항상 침묵" 상수의 비위반 쪽 점수 = 그 풀의 빈 라벨 비율
    silent_score = float(np.mean([1.0 if not g else 0.0 for g in Gn]))
    logger.info(f"  상수 기준선 '항상 침묵' — 위반 쪽 0%, 비위반 쪽 {silent_score * 100:.1f}%")

    report = {}
    for objective in ("f1", "recall_floor"):
        logger.info(f"  ----- 목적함수: {objective}"
                    + (f" (위반 쪽 하한 {_VIOLATION_FLOOR:.2f})" if objective != "f1" else "") + " -----")
        logger.info(f"    {'r':>6}{'τ(A)':>7}{'τ(B)':>7}{'위반쪽 F1':>11}{'비위반 침묵':>12}"
                    f"{'결합':>8}{'항상침묵':>10}{'차이':>10}")
        rows = []
        for r in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
            tA = _tune(Sv[:, Av], Sn[:, An], r, objective)
            tB = _tune(Sv[:, Bv], Sn[:, Bn], r, objective)
            iA, iB = int(np.argmin(abs(_GRID - tA))), int(np.argmin(abs(_GRID - tB)))
            # A에서 맞춘 τ로 B를 채점, 그 반대도 — 두 폴드를 합쳐 편향 없는 추정
            viol = float(np.concatenate([Sv[iA, Bv], Sv[iB, Av]]).mean())
            non = float(np.concatenate([Sn[iA, Bn], Sn[iB, An]]).mean())
            comb = r * viol + (1 - r) * non
            # **상수 기준선: 아무 말도 안 하기.** 위반 쪽 0점, 비위반 쪽은 만점에 가깝다.
            # 이걸 안 넘으면 모델을 배포하는 것이 침묵보다 나쁘다.
            silent = (1 - r) * silent_score
            rows.append({"r": r, "tau_A": tA, "tau_B": tB, "violation_f1": viol,
                         "nonviolation_silence": non, "combined": comb,
                         "always_silent": silent, "vs_silent_pp": (comb - silent) * 100})
            logger.info(f"    {r:>6.2f}{tA:>7.2f}{tB:>7.2f}{viol * 100:>10.1f}%{non * 100:>11.1f}%"
                        f"{comb * 100:>7.1f}%{silent * 100:>9.1f}%{(comb - silent) * 100:>+9.1f}p")
        report[objective] = rows

    for obj, rows in report.items():
        loss = [x["r"] for x in rows if x["vs_silent_pp"] < 0]
        if loss:
            logger.warning(f"  ★ {obj}: r ≤ {max(loss):.2f}에서 **침묵보다 나쁘다** — "
                           f"그 유병률에서는 배포하지 않는 편이 낫다")
        win = [x["r"] for x in rows if x["vs_silent_pp"] > 0]
        if win:
            logger.info(f"  ★ {obj}: r ≥ {min(win):.2f}부터 침묵을 이긴다")

    f1r, pr = report["f1"], report["recall_floor"]
    logger.info("  ----- 목적함수가 결과를 얼마나 바꾸나 -----")
    for x, y in zip(f1r, pr):
        logger.info(f"    r={x['r']:.2f}  위반쪽 {x['violation_f1'] * 100:5.1f}% → {y['violation_f1'] * 100:5.1f}%"
                    f"  ({y['violation_f1'] * 100 - x['violation_f1'] * 100:+.1f}%p)  |  "
                    f"비위반 {x['nonviolation_silence'] * 100:5.1f}% → {y['nonviolation_silence'] * 100:5.1f}%"
                    f"  ({y['nonviolation_silence'] * 100 - x['nonviolation_silence'] * 100:+.1f}%p)")

    save_json({"model_dir": a.model_dir, "n_violation": len(Gv), "n_nonviolation": len(Gn),
               "violation_floor": _VIOLATION_FLOOR, "curves": report,
               "assumption": "두 풀이 각각 위반/비위반 조항을 대표하고 유병률만 다르다. "
                             "틀리면 곡선 전체가 이동한다 — 그 경우 실제 계약서 소량 라벨링이 다음 수",
               "note": "τ는 문서 단위 2-fold 교차적합으로 맞췄다(맞춘 폴드와 채점 폴드가 다름)"},
              OUT_PATH)
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
