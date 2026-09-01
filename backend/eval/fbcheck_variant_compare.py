# backend/eval/fbcheck_variant_compare.py
"""
FB-Check 판정 규칙 변형 비교 — 세 번째 투표자(KoELECTRA)가 검증인가 노이즈인가.

## 배경

현행 FB-Check는 forward(GPT-4o) · verify(GPT-4o) · backward(KoELECTRA) 세 표 중 2표가
일치하면 CLEAN으로 판정한다. 세 번째 표를 넣은 의도는 "KoELECTRA는 GPT와 다른 모델이라
같은 편향을 공유하지 않으므로 진짜 교차검증이 된다"였다.

측정해보니 그 전제가 성립하지 않는다(세 투표자가 모두 라벨을 낸 744건 기준):

```
forward(GPT) ↔ verify(GPT)         78.0%
forward(GPT) ↔ backward(KoELECTRA) 52.2%
KoELECTRA가 "Low"만 찍는 상수였다면 59.5%   ← 실제보다 높다
```

**상수 투표자보다 못하다.** 게다가 이 표가 전체 2,218건 중 990건(44.6%)에서 캐스팅보트를
쥐었고, CLEAN 694건 중 114건(16.4%)의 라벨을 정했다. 그리고 그 CLEAN으로 다시 KoELECTRA를
학습시키므로(README의 "Data Flywheel") **검증자가 자기 산출물로 학습되는 순환**이다.

## 비교하는 규칙 4종

| 이름 | 규칙 |
|---|---|
| `current` | forward·verify·KoELECTRA 2/3 다수결 (현행) |
| `two_way` | forward == verify (KoELECTRA는 `check_snippet_exists` 필터 역할만) |
| `exaone_3way` | forward·verify·**EXAONE** 2/3 다수결 (독립 검증자로 교체) |
| `unanimous` | forward == verify == EXAONE (전원 일치, 가장 엄격) |

`check_snippet_exists`(E⊂C)는 순수 문자열 매칭이라 모델과 무관하며, 모든 변형에서 사전 조건으로
동일하게 적용된다 — 그래서 "KoELECTRA를 필터로만 쓴다"(C안)는 `two_way`와 동일하다.

## 무엇으로 좋고 나쁨을 가리나

정답 라벨이 없으므로 **간접 지표 세 개**를 함께 본다:

1. **수율** — CLEAN 건수. 너무 적으면 학습이 안 된다
2. **출처 교락도** — `출처만 보고 최빈 라벨 찍기` 정확도. **낮을수록 좋다.**
   현행 CLEAN은 77.8%로, 라벨이 출처에 거의 결정된다
3. **라벨 균형** — 특정 클래스로 쏠리면 다수 클래스 기준선만 올라간다

최종 판단은 각 변형으로 만든 CLEAN에 재학습해서 하지만, 그 전에 이 지표로 후보를 좁힌다.

실행:
    .venv/bin/python -m backend.eval.fbcheck_variant_compare              # EXAONE 라벨 캐시 사용
    .venv/bin/python -m backend.eval.fbcheck_variant_compare --build-exaone   # 캐시 생성(로컬 GPU, 비용 0)
"""

import argparse
import json
from collections import Counter, defaultdict

from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("fbcheck_variant_compare.log")

CLEAN_PATH = PROJECT_ROOT / "data/fb_check/clean.jsonl"
NOISE_PATH = PROJECT_ROOT / "data/fb_check/noise.jsonl"
EXAONE_CACHE = PROJECT_ROOT / "data/fb_check/exaone_votes.jsonl"
OUT_PATH = PROJECT_ROOT / "data/eval/fbcheck_variant_report.json"

_LABELS = ("High", "Medium", "Low")

_EXAONE_SYSTEM = (
    "너는 한국 계약법 전문가다. 계약 조항을 읽고 불공정 위험도를 판정해라.\n"
    "High: 사업자에게 일방적으로 유리해 무효가 될 소지가 큰 조항\n"
    "Medium: 다툼의 여지가 있는 조항\n"
    "Low: 통상적이고 문제없는 조항\n"
    '반드시 JSON만 출력: {"risk_level": "High|Medium|Low"}'
)
_EXAONE_FEWSHOT = [
    {"role": "user", "content": "조항:\n회사는 사전 통지 없이 언제든지 계약을 해지할 수 있으며 이로 인한 손해에 어떠한 책임도 지지 않는다."},
    {"role": "assistant", "content": '{"risk_level": "High"}'},
    {"role": "user", "content": "조항:\n당사자는 30일 전 서면 통지로 본 계약을 해지할 수 있다."},
    {"role": "assistant", "content": '{"risk_level": "Low"}'},
]


def build_exaone_votes(rows: list[dict]) -> None:
    """EXAONE으로 각 조항의 risk_level을 예측해 캐시에 append한다(로컬 GPU, API 비용 0).

    한 건씩 즉시 append한다 — 2,000건 넘는 추론이라 중간에 끊겨도 이어서 진행할 수 있어야 한다.
    """
    from backend.eval.local_llm import generate_json

    done = {r["fb_id"] for r in load_jsonl(EXAONE_CACHE)} if EXAONE_CACHE.exists() else set()
    todo = [r for r in rows if r.get("fb_id") not in done and (r.get("text") or "").strip()]
    logger.info(f"  EXAONE 투표 생성: 이미 {len(done)}건 / 남은 {len(todo)}건")

    EXAONE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(EXAONE_CACHE, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            out = generate_json(_EXAONE_SYSTEM, f"조항:\n{(r['text'] or '')[:800]}",
                                max_new_tokens=30, fewshot=_EXAONE_FEWSHOT)
            label = (out or {}).get("risk_level")
            f.write(json.dumps({"fb_id": r.get("fb_id"), "chunk_id": r.get("chunk_id"),
                                "exaone_risk": label if label in _LABELS else None},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                logger.info(f"    {i}/{len(todo)}")


def _majority(votes: list[str | None]) -> tuple[str | None, int]:
    present = [v for v in votes if v]
    if not present:
        return None, 0
    return Counter(present).most_common(1)[0]


def decide(r: dict, rule: str, exaone: str | None) -> tuple[bool, str | None]:
    """규칙별 (CLEAN 여부, 확정 라벨). check_snippet_exists는 모든 규칙의 공통 사전 조건."""
    if r.get("snippet_exists") is False:
        return False, None
    f, v, b = r.get("forward_label"), r.get("verify_label"), r.get("backward_risk")

    if rule == "current":
        lab, cnt = _majority([f, v, b])
        return cnt >= 2, lab
    if rule == "two_way":
        return bool(f and v and f == v), f
    if rule == "exaone_3way":
        lab, cnt = _majority([f, v, exaone])
        return cnt >= 2, lab
    if rule == "unanimous":
        ok = bool(f and v and exaone and f == v == exaone)
        return ok, f
    raise ValueError(rule)


def source_confound(rows: list[tuple[str, str]]) -> float:
    """출처만 보고 최빈 라벨을 찍었을 때의 정확도 — 낮을수록 교락이 적다."""
    by: dict[str, Counter] = defaultdict(Counter)
    for src, lab in rows:
        by[src][lab] += 1
    total = sum(sum(c.values()) for c in by.values())
    if not total:
        return float("nan")
    return sum(c.most_common(1)[0][1] for c in by.values()) / total


def main(build: bool = False) -> None:
    rows = load_jsonl(CLEAN_PATH) + load_jsonl(NOISE_PATH)
    logger.info(f"  FB-Check 전체 {len(rows)}건")

    if build:
        build_exaone_votes(rows)

    exa = {r["fb_id"]: r.get("exaone_risk") for r in load_jsonl(EXAONE_CACHE)} if EXAONE_CACHE.exists() else {}
    if exa:
        logger.info(f"  EXAONE 투표 캐시 {len(exa)}건 (라벨 있음 {sum(1 for v in exa.values() if v)}건)")

    rules = ["current", "two_way"] + (["exaone_3way", "unanimous"] if exa else [])
    if not exa:
        logger.warning("  EXAONE 캐시가 없어 B/A+B+C 변형은 건너뛴다 (--build-exaone 으로 생성)")

    results = {}
    for rule in rules:
        kept = [(r, lab) for r in rows
                for ok, lab in [decide(r, rule, exa.get(r.get("fb_id")))] if ok and lab]
        labels = Counter(lab for _, lab in kept)
        conf = source_confound([(r.get("source"), lab) for r, lab in kept])
        by_src = defaultdict(Counter)
        for r, lab in kept:
            by_src[r.get("source")][lab] += 1
        results[rule] = {
            "clean_n": len(kept), "yield": len(kept) / len(rows),
            "labels": dict(labels), "source_confound": conf,
            "by_source": {s: dict(c) for s, c in by_src.items()},
        }

    save_json({"n_total": len(rows), "results": results}, OUT_PATH)

    logger.info("===== FB-Check 판정 규칙 비교 =====")
    logger.info(f"  {'규칙':<14}{'CLEAN':>8}{'수율':>8}{'출처교락':>10}   라벨 분포")
    for rule in rules:
        v = results[rule]
        dist = " ".join(f"{k}={v['labels'].get(k, 0)}" for k in _LABELS)
        logger.info(f"  {rule:<14}{v['clean_n']:>8}{v['yield'] * 100:>7.1f}%{v['source_confound'] * 100:>9.1f}%   {dist}")
    logger.info("  ※ 출처교락이 낮을수록 좋다(라벨이 출처로 결정되지 않는다는 뜻)")
    for rule in rules:
        logger.info(f"  [{rule}] 출처별: {results[rule]['by_source']}")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build-exaone", action="store_true", help="EXAONE 투표 캐시 생성(로컬 GPU)")
    a = p.parse_args()
    main(build=a.build_exaone)
