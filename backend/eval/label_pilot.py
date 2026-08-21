# backend/eval/label_pilot.py
"""
라벨링 파일럿 — LLM이 지목한 약관규제법 조가 공정위 확정과 맞는가.

## 왜 필요한가

라벨 taxonomy를 2개 도메인(해지·책임제한)에서 약관규제법 제6~14조 9개로 바꿨다.
전량(2,000건+) 재라벨링 전에, **그 판정이 맞는지 먼저 확인해야 한다.**

지금까지 이 프로젝트에는 LLM 라벨을 채점할 방법이 없었다. `clean.jsonl`의 정확도는
"GPT 판정을 GPT로 검증"한 값이라 순환이었다. 유형을 약관규제법 조로 바꾸면서 처음으로
**외부 정답**이 생겼다 — FTC 의결서 1,092건 중 975건(89.3%)에 `근거_법령`이 달려 있고,
그게 곧 "공정위가 이 조항을 어느 조 위반으로 판단했는가"다.

    조항 원문  →  LLM  →  ["제9조", "제7조"]
                              ↕  대조
    의결서 근거_법령  →  {"제9조", "제6조"}

## 무엇을 재나

케이스 단위 정답(`근거_법령`)을 그 케이스의 조항에 부여한다. 한 의결서에 조항이 여러 개면
어느 조항이 어느 조에 걸리는지까지는 알 수 없으므로, **조항 1개짜리 케이스만** 쓴다 —
그래야 (조항 ↔ 근거 조) 대응이 1:1로 확정된다.

    hit@any   : LLM이 지목한 조 중 하나라도 정답에 있는가 (관대)
    precision : 지목한 조 중 정답 비율
    recall    : 정답 조 중 지목한 비율
    exact     : 집합이 완전히 일치

`--model`로 여러 모델을 같은 표본에 돌려 비교한다 — 9유형 분류를 gpt-4o-mini가
감당하는지가 전량 실행 전 결정 사항이다.

## 한계

`근거_법령`은 **의결서 전체**의 근거이고 조항 단위로 분해된 것이 아니다. 조항이 1개인
케이스로 제한해 이 문제를 줄였지만, 의결서가 절차 조항(제17조 시정조치 등)을 함께
인용하는 경우가 있어 정답 집합에 노이즈가 남는다. 그래서 `hit@any`를 주 지표로 보고
`exact`는 참고로만 쓴다.

실행:
    .venv/bin/python -m backend.eval.label_pilot --n 100
    .venv/bin/python -m backend.eval.label_pilot --n 100 --model gpt-4o-mini --model gpt-4o
"""

import argparse
import json
import random
from collections import Counter

from openai import OpenAI

from backend.fb_check.consistency_verification import run_verify
from backend.fb_check.forward_labeling import run_forward
from backend.labeling.articles import normalize
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("label_pilot.log")

FTC_PATH = PROJECT_ROOT / "data/raw/ftc_cases/ftc_cases_parsed.json"
OUT_PATH = PROJECT_ROOT / "data/eval/label_pilot_report.json"

_MIN_CLAUSE_CHARS = 30


def build_samples(n: int, seed: int = 42) -> list[dict]:
    """조항이 1개인 FTC 케이스만 뽑는다 — (조항 ↔ 근거 조) 대응이 1:1로 확정되는 것만."""
    cases = json.loads(FTC_PATH.read_text(encoding="utf-8"))["사례"]
    pool = []
    for c in cases:
        clauses = [str(x).strip() for x in (c.get("조항_원문") or []) if len(str(x).strip()) >= _MIN_CLAUSE_CHARS]
        if len(clauses) != 1:
            continue
        gold = {a for g in (c.get("근거_법령") or []) if (a := normalize(str(g)))}
        if not gold:
            continue
        pool.append({"case_name": c.get("사건명", ""), "clause": clauses[0], "gold": sorted(gold)})
    logger.info(f"  조항 1개 + 근거_법령 있는 케이스 {len(pool)}건 (전체 {len(cases)}건 중)")
    random.seed(seed)
    return random.sample(pool, min(n, len(pool)))


def score(pred: list[str], gold: list[str]) -> dict:
    p, g = set(pred), set(gold)
    inter = p & g
    return {
        "hit_any":   bool(inter),
        "exact":     p == g,
        "precision": len(inter) / len(p) if p else 0.0,
        "recall":    len(inter) / len(g) if g else 0.0,
        "pred": sorted(p), "gold": sorted(g),
    }


def run_one(client: OpenAI, sample: dict, model: str) -> dict | None:
    """forward → verify까지 실제 파이프라인 함수를 그대로 쓴다(프롬프트 분기 방지)."""
    fwd = run_forward(client, sample["clause"], model=model)
    if not fwd:
        return None
    fwd_arts = fwd.get("articles", [])
    span = fwd.get("evidence_span", "")

    mode = "span" if fwd_arts and span else "clause"
    ver = run_verify(client, span if mode == "span" else sample["clause"], mode=mode, model=model)
    ver_arts = (ver or {}).get("articles", [])
    agreed = [a for a in fwd_arts if a in set(ver_arts)]

    return {
        "case_name": sample["case_name"],
        "clause": sample["clause"][:200],
        "forward_articles": fwd_arts, "verify_articles": ver_arts, "agreed_articles": agreed,
        "forward_risk": fwd.get("risk_level"), "verify_risk": (ver or {}).get("risk_level"),
        "label_match": bool(fwd.get("risk_level") and fwd.get("risk_level") == (ver or {}).get("risk_level")),
        "evidence_span": span, "verify_mode": mode,
        "score_forward": score(fwd_arts, sample["gold"]),
        "score_agreed":  score(agreed,  sample["gold"]),
    }


def summarize(rows: list[dict], key: str) -> dict:
    n = len(rows)
    if not n:
        return {}
    s = [r[key] for r in rows]
    return {
        "hit_any":       sum(x["hit_any"] for x in s) / n,
        "exact":         sum(x["exact"] for x in s) / n,
        "precision":     sum(x["precision"] for x in s) / n,
        "recall":        sum(x["recall"] for x in s) / n,
        "empty_pred":    sum(1 for x in s if not x["pred"]) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", action="append", default=None, help="반복 지정 가능 (기본: .env의 FORWARD_MODEL)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    samples = build_samples(a.n, a.seed)
    logger.info(f"  평가 표본 {len(samples)}건")

    import os
    models = a.model or [os.environ["FORWARD_MODEL"]]
    client = OpenAI()
    report: dict = {"n": len(samples), "seed": a.seed, "models": {}}

    for model in models:
        logger.info(f"===== {model} =====")
        rows = []
        for i, s in enumerate(samples, 1):
            r = run_one(client, s, model)
            if r:
                rows.append(r)
            if i % 20 == 0:
                logger.info(f"  {i}/{len(samples)}")

        fwd_sum, agr_sum = summarize(rows, "score_forward"), summarize(rows, "score_agreed")
        label_match = sum(r["label_match"] for r in rows) / len(rows) if rows else 0
        art_dist = Counter(a_ for r in rows for a_ in r["forward_articles"])
        gold_dist = Counter(a_ for s in samples for a_ in s["gold"])

        report["models"][model] = {
            "n_ok": len(rows), "forward": fwd_sum, "agreed": agr_sum,
            "label_match_rate": label_match,
            "pred_article_dist": dict(art_dist.most_common()),
            "gold_article_dist": dict(gold_dist.most_common()),
            "rows": rows,
        }
        logger.info(f"  forward 지목:  hit@any {fwd_sum['hit_any']*100:.1f}%  정밀 {fwd_sum['precision']*100:.1f}%  "
                    f"재현 {fwd_sum['recall']*100:.1f}%  완전일치 {fwd_sum['exact']*100:.1f}%  빈배열 {fwd_sum['empty_pred']*100:.1f}%")
        logger.info(f"  합의(F∩V):     hit@any {agr_sum['hit_any']*100:.1f}%  정밀 {agr_sum['precision']*100:.1f}%  "
                    f"재현 {agr_sum['recall']*100:.1f}%  완전일치 {agr_sum['exact']*100:.1f}%  빈배열 {agr_sum['empty_pred']*100:.1f}%")
        logger.info(f"  L == L' 일치율: {label_match*100:.1f}%")
        logger.info(f"  지목 분포: {dict(art_dist.most_common(5))}")
        logger.info(f"  정답 분포: {dict(gold_dist.most_common(5))}")

    save_json(report, OUT_PATH)
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
