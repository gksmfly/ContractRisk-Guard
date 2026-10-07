# eval/requirement_format_check.py
"""V1/A1 프롬프트 형식 점검 (사전 등록: eval/prereg_requirement_verification.md §4)

**채점이 아니다.** 비-gold 풀에서 최대 30건을 뽑아 V1·A1을 돌리고, 파싱 실패율과 "판단불가" 비율 같은
형식 지표만 본다. 정답과 비교하지 않고, 그 결과로 프롬프트의 판정 규칙을 고치지 않는다 —
고치는 것은 JSON 스키마·지시문이 깨진 경우뿐이다(고쳤다면 `PROMPT_VERSION`을 올리고 기록한다).

풀: FB-Check ftc_case 레코드에서 **gold 255 문서를 문서 단위로 제외**한 것(gold 텍스트도 이중 제외).

    .venv/bin/python -m eval.requirement_format_check --plan      # 표본·호출 수만 출력, API 0
    .venv/bin/python -m eval.requirement_format_check --run       # API 호출 (캐시, 끊겨도 이어서)
    .venv/bin/python -m eval.requirement_format_check --report    # 캐시에서 형식 지표 집계, API 0
"""

import argparse
import json
import os
import random
import subprocess
import time
from collections import Counter

from backend.fb_check.api_errors import raise_if_fatal
from backend.training.train_article import load_ftc_gold
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json
from eval.channel.fbcheck_external_eval import CLEAN_PATH, NOISE_PATH, _doc_of
from eval.requirement_verifier import PROMPT_VERSION, build_messages, load_cards, parse

logger = load_logger("requirement_format_check.log")

OUT_DIR = PROJECT_ROOT / "data/eval/requirement_format_check" / PROMPT_VERSION   # 버전별 캐시 — v1 결과를 보존한다
MODEL = "gpt-4o"            # .env의 FORWARD_MODEL을 쓰지 않는다 (7-4와 같은 이유)
N_RECORDS = 30
SEED = 42
RETRIES = 3                 # 파싱 실패 재시도 횟수(총 시도 = 1 + RETRIES - 1)
# 호출당 입력 약 2,000토큰(V1) · TPM 30,000 → 분당 15건 이하. 7-4와 같은 계정 한도를 쓴다
MIN_INTERVAL_S = 4.5


def _pool() -> list[dict]:
    gold = load_ftc_gold("clean")
    gold_docs = {f"ftc_case:{g['doc_id']}" for g in gold}
    gold_text = {g["text"].strip() for g in gold}
    cards = load_cards()
    out = []
    for p in (CLEAN_PATH, NOISE_PATH):
        for r in load_jsonl(p):
            if r.get("source") != "ftc_case" or _doc_of(r.get("chunk_id", "")) in gold_docs:
                continue
            if r["text"].strip() in gold_text:
                continue
            arts = [a for a in (r.get("forward_articles") or []) if a in cards]
            if arts:
                out.append({"chunk_id": r["chunk_id"], "text": r["text"],
                            "evidence": r.get("evidence_span") or "", "articles": arts})
    return out


def sample() -> list[dict]:
    """조 8·13처럼 드문 조가 빠지지 않도록, 조마다 한 건을 먼저 뽑고 나머지는 무작위로 채운다."""
    rng = random.Random(SEED)
    pool = sorted(_pool(), key=lambda r: r["chunk_id"])
    chosen, seen = [], set()
    by_art: dict[str, list[dict]] = {}
    for r in pool:
        for a in r["articles"]:
            by_art.setdefault(a, []).append(r)
    for a in sorted(by_art, key=lambda x: len(by_art[x])):          # 드문 조부터
        if a not in {x for c in chosen for x in c["articles"]}:
            r = rng.choice(by_art[a])
            if r["chunk_id"] not in seen:
                chosen.append(r)
                seen.add(r["chunk_id"])
    rest = [r for r in pool if r["chunk_id"] not in seen]
    rng.shuffle(rest)
    chosen += rest[:max(0, N_RECORDS - len(chosen))]
    return chosen[:N_RECORDS]


def _tasks(recs: list[dict]) -> list[tuple[str, dict, str]]:
    return [(v, r, a) for v in ("V1", "A1") for r in recs for a in r["articles"]]


def _key(variant: str, rec: dict, art: str) -> str:
    return f"{variant}|{rec['chunk_id']}|{art}"


def _load_cache(variant: str) -> dict[str, dict]:
    path = OUT_DIR / f"{variant}.jsonl"
    return {r["key"]: r for r in load_jsonl(path)} if path.exists() else {}


def plan() -> None:
    recs = sample()
    tasks = _tasks(recs)
    arts = Counter(a for r in recs for a in r["articles"])
    print(f"표본 {len(recs)}건 · 호출 {len(tasks)}건(V1 {len(tasks) // 2} + A1 {len(tasks) // 2}) · "
          f"최소 소요 약 {len(tasks) * MIN_INTERVAL_S / 60:.0f}분")
    print("조별 (조항, 조) 쌍:", dict(sorted(arts.items(), key=lambda x: int(x[0][1:-1]))))
    done = sum(len(_load_cache(v)) for v in ("V1", "A1"))
    print(f"캐시 완료 {done}건 → 남은 호출 {len(tasks) - done}건")


def run() -> None:
    from dotenv import load_dotenv
    from openai import OpenAI
    # 7-4와 같은 계정 TPM(30,000)을 나눠 쓰면 429가 재시도까지 소진해 빈 판정이 섞인다
    if subprocess.run(["pgrep", "-f", "e_teacher_repeat"], capture_output=True).returncode == 0:
        raise SystemExit("7-4(e_teacher_repeat)가 실행 중이다 — 끝난 뒤에 돌린다")
    load_dotenv(PROJECT_ROOT / ".env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    cards = load_cards()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recs = sample()
    for variant in ("V1", "A1"):
        cache = _load_cache(variant)
        todo = [(v, r, a) for v, r, a in _tasks(recs) if v == variant and _key(v, r, a) not in cache]
        logger.info(f"=== {variant}: 남음 {len(todo)} ===")
        with open(OUT_DIR / f"{variant}.jsonl", "a", encoding="utf-8") as f:
            for i, (v, r, a) in enumerate(todo, 1):
                t0 = time.time()
                msgs = build_messages(v, r["text"], r["evidence"], a, cards)
                parsed, attempts, raws = None, 0, []
                for attempts in range(1, RETRIES + 1):
                    try:
                        resp = client.chat.completions.create(
                            model=MODEL, messages=msgs, temperature=0,
                            response_format={"type": "json_object"})
                    except Exception as e:
                        raise_if_fatal(e, "요건 검증 형식 점검")     # 크레딧·키 오류는 즉시 중단
                        logger.warning(f"  호출 실패 {_key(v, r, a)} ({attempts}/{RETRIES}): {e}")
                        time.sleep(2 ** attempts)
                        continue
                    raw = resp.choices[0].message.content
                    raws.append(raw)
                    parsed = parse(v, raw, a, cards)
                    if parsed is not None:
                        break
                f.write(json.dumps({"key": _key(v, r, a), "variant": v, "article": a, "attempts": attempts,
                                    "parsed": parsed, "raw": raws[-1] if raws else None, "model": MODEL,
                                    "prompt": PROMPT_VERSION}, ensure_ascii=False) + "\n")
                f.flush()
                if i % 10 == 0:
                    logger.info(f"  [{i}/{len(todo)}]")
                time.sleep(max(0.0, MIN_INTERVAL_S - (time.time() - t0)))
    report()


def report() -> None:
    """형식 지표만 집계한다. 정답과 비교하지 않는다."""
    out = {"prompt_version": PROMPT_VERSION, "model": MODEL, "n_records": N_RECORDS}
    for variant in ("V1", "A1"):
        rows = list(_load_cache(variant).values())
        if not rows:
            continue
        ok = [r for r in rows if r["parsed"] is not None]
        if variant == "V1":
            verdicts = Counter(i["verdict"] for r in ok for i in r["parsed"]["items"])
        else:
            verdicts = Counter(r["parsed"]["verdict"] for r in ok)
        total_v = sum(verdicts.values()) or 1
        out[variant] = {
            "n_calls": len(rows), "parse_fail": len(rows) - len(ok),
            "parse_fail_rate": (len(rows) - len(ok)) / len(rows),
            "retried": sum(1 for r in rows if r["attempts"] > 1),
            "verdicts": dict(verdicts),
            "undecidable_rate": verdicts.get("판단불가", 0) / total_v,
            "all_undecidable_pairs": sum(
                1 for r in ok if (all(i["verdict"] == "판단불가" for i in r["parsed"]["items"])
                                  if variant == "V1" else r["parsed"]["verdict"] == "판단불가")),
        }
    save_json(out, OUT_DIR / "summary.json")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--report", action="store_true")
    a = ap.parse_args()
    plan() if a.plan else run() if a.run else report()
