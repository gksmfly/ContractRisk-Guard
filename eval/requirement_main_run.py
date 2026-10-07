# eval/requirement_main_run.py
"""요건 검증 본 실행 — gold clean 255에서 V1·A1 1회 (사전 등록: eval/prereg_requirement_verification.md §1·§8·§9)

**gold를 한 번 쓰는 실행이다.** `--run`은 확인 플래그가 없으면 시작하지 않는다.
결과를 본 뒤 프롬프트를 고치면 안 된다(고쳤다면 그 사실과 사유를 논문에 쓴다).

    .venv/bin/python -m eval.requirement_main_run --selftest             # API 0: 채점 경로 검증(V1=A1=전부 유지)
    .venv/bin/python -m eval.requirement_main_run --run --gold-once-ok    # API 호출 (캐시, 끊겨도 이어서)
    .venv/bin/python -m eval.requirement_main_run --score                # API 0: 캐시에서 채점·보고

조건: B0(forward 그대로) · B1(FB-Check, noise→빈 예측) · B2(V1/A1이 제거한 개수만큼 무작위 제거, 시드 1,000)
      · V1 · A1.  단위는 (조항, 조) 라벨. 주 지표 = 제6조 제외 micro F1, V1 − B0, 조항 단위 paired bootstrap.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time

import numpy as np

from backend.fb_check.api_errors import raise_if_fatal
from backend.training.train_article import load_ftc_gold
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json
from eval.channel.fbcheck_external_eval import CLEAN_PATH, NOISE_PATH, _doc_of
from eval.requirement_verifier import PROMPT_VERSION, build_messages, keep, load_cards, parse

logger = load_logger("requirement_main_run.log")

OUT_DIR = PROJECT_ROOT / "data/eval/requirement_main_run" / PROMPT_VERSION
# 요청 단계에서 스냅샷을 고정한다(별칭 "gpt-4o"는 실행 중에 바뀔 수 있다). 2026-10-07 별칭이 가리킨 스냅샷.
MODEL = "gpt-4o-2024-08-06"
# B0의 forward 라벨은 별칭 "gpt-4o"로 만들어졌고 레코드에 스냅샷이 남지 않았다 → 같은 스냅샷인지 확인할 수 없다.
B0_MODEL_NOTE = "forward 라벨 생성 시 별칭 'gpt-4o'(스냅샷 미기록) — V1·A1과 같은 스냅샷인지 확인 불가"
RETRIES = 3
MIN_INTERVAL_S = 4.5
BOOT_N = 10_000
B2_SEEDS = 1_000
SEED = 42
ART6 = "제6조"


def load_clauses() -> list[dict]:
    """gold 255를 **문서 ID**로 FB-Check 레코드와 매칭한다(텍스트 매칭은 중복 때문에 245건뿐)."""
    gold = load_ftc_gold("clean")
    by_doc: dict[str, tuple[str, dict]] = {}
    for tag, p in (("clean", CLEAN_PATH), ("noise", NOISE_PATH)):
        for r in load_jsonl(p):
            if r.get("source") == "ftc_case":
                by_doc.setdefault(_doc_of(r["chunk_id"]), (tag, r))
    out = []
    for g in gold:
        tag, r = by_doc["ftc_case:" + g["doc_id"]]
        assert r["text"].strip() == g["text"].strip(), g["doc_id"]
        out.append({"doc": g["doc_id"], "text": g["text"], "evidence": r.get("evidence_span") or "",
                    "ref": set(g["articles"]), "fwd": list(r.get("forward_articles") or []),
                    "b1_pass": tag == "clean"})
    return out


def _key(variant: str, doc: str, art: str) -> str:
    return f"{variant}|{doc}|{art}"


def _cache(variant: str) -> dict[str, dict]:
    path = OUT_DIR / f"{variant}.jsonl"
    return {r["key"]: r for r in load_jsonl(path)} if path.exists() else {}


def run() -> None:
    from dotenv import load_dotenv
    from openai import OpenAI
    if subprocess.run(["pgrep", "-f", "eval.label.e_teacher_repeat"], capture_output=True).returncode == 0:
        raise SystemExit("7-4(e_teacher_repeat)가 실행 중이다 — 같은 계정 TPM을 나눠 쓰지 않는다")
    load_dotenv(PROJECT_ROOT / ".env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    cards, clauses = load_cards(), load_clauses()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for variant in ("V1", "A1"):
        cache = _cache(variant)
        todo = [(c, a) for c in clauses for a in c["fwd"] if a in cards and _key(variant, c["doc"], a) not in cache]
        logger.info(f"=== {variant}: 남음 {len(todo)} ===")
        with open(OUT_DIR / f"{variant}.jsonl", "a", encoding="utf-8") as f:
            for i, (c, a) in enumerate(todo, 1):
                t0 = time.time()
                msgs = build_messages(variant, c["text"], c["evidence"], a, cards)
                parsed, raws, snap = None, [], None
                for attempt in range(1, RETRIES + 1):
                    try:
                        resp = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0,
                                                              response_format={"type": "json_object"})
                    except Exception as e:
                        raise_if_fatal(e, "요건 검증 본 실행")
                        logger.warning(f"  호출 실패 {_key(variant, c['doc'], a)} ({attempt}/{RETRIES}): {e}")
                        time.sleep(2 ** attempt)
                        continue
                    raws.append(resp.choices[0].message.content)
                    snap = resp.model                       # 실제로 응답한 스냅샷
                    if snap != MODEL:                       # 요청한 스냅샷과 다르면 V1·A1이 다른 모델로 섞이므로 중단
                        raise SystemExit(f"스냅샷 불일치: 요청 {MODEL} ≠ 응답 {snap} — 실행 중단")
                    parsed = parse(variant, raws[-1], a, cards)
                    if parsed is not None:
                        break
                if not raws:
                    # API 오류로 응답을 한 번도 못 받았다. **파싱 실패(→유지)와 다르다.** 기록하지 않고 실행을 중단한다 —
                    # 그러지 않으면 이 라벨이 조용히 "유지"로 들어가 V1이 B0와 같아진다. 캐시가 있으니 다시 돌리면 이어진다.
                    raise SystemExit(f"API 오류로 {_key(variant, c['doc'], a)} 응답을 받지 못했다({RETRIES}회) — 실행 중단, 같은 명령으로 이어서 돌린다")
                f.write(json.dumps({"key": _key(variant, c["doc"], a), "variant": variant, "article": a,
                                    "attempts": attempt, "parsed": parsed, "raw": raws[-1],
                                    "model": MODEL, "model_snapshot": snap, "prompt": PROMPT_VERSION}, ensure_ascii=False) + "\n")
                f.flush()
                if i % 25 == 0:
                    logger.info(f"  [{i}/{len(todo)}]")
                time.sleep(max(0.0, MIN_INTERVAL_S - (time.time() - t0)))
    score()


# ───────────────────────── 채점 (API 0) ─────────────────────────

def _counts(clauses: list[dict], kept: list[set[str]], drop6: bool) -> np.ndarray:
    """조항별 (TP, FP, FN)."""
    out = np.zeros((len(clauses), 3))
    for i, (c, k) in enumerate(zip(clauses, kept)):
        ref, pred = set(c["ref"]), set(k)
        if drop6:
            ref.discard(ART6)
            pred.discard(ART6)
        out[i] = (len(ref & pred), len(pred - ref), len(ref - pred))
    return out


def _f1(cnt: np.ndarray) -> float:
    tp, fp, fn = cnt.sum(0)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def _prec_rec(cnt: np.ndarray) -> tuple[float, float]:
    tp, fp, fn = cnt.sum(0)
    return (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)


def _decisions(variant: str, clauses: list[dict], cards: dict, rule: str = "main") -> tuple[list[set[str]], dict]:
    """조건별로 유지된 라벨 집합. rule: main(사전 등록) · drop_undecided(민감도 a) · v1_old(민감도 b, V1만)."""
    cache = _cache(variant)
    kept, n_fail, n_pairs = [], 0, 0
    for c in clauses:
        ks = set()
        for a in c["fwd"]:
            r = cache.get(_key(variant, c["doc"], a))
            if r is None:
                raise SystemExit(f"{variant} 캐시에 {_key(variant, c['doc'], a)} 없음 — --run을 끝까지 돌려야 한다")
            if not r.get("raw"):
                raise SystemExit(f"{_key(variant, c['doc'], a)}: raw 없음(API 오류 기록) — 채점하지 않는다")
            p = parse(variant, r["raw"], a, cards)
            n_pairs += 1
            n_fail += p is None
            if rule == "main":
                k = keep(variant, p)
            elif p is None:
                k = True
            elif rule == "drop_undecided":
                k = (any(i["verdict"] == "충족" for i in p["items"]) if variant == "V1" else p["verdict"] == "해당")
            else:  # v1_old: 모든 호가 판단불가일 때만 유지
                vs = [i["verdict"] for i in p["items"]]
                k = "충족" in vs or all(v == "판단불가" for v in vs)
            if k:
                ks.add(a)
        kept.append(ks)
    return kept, {"pairs": n_pairs, "parse_fail": n_fail}


def _repro(selftest: bool) -> dict:
    """재현 정보: 커밋 해시, 작업 트리 변경 여부, 프롬프트 버전, 모델 스냅샷."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    watched = ["eval/requirement_verifier.py", "eval/requirement_main_run.py", "eval/requirement_cards.py",
               "eval/prereg_requirement_verification.md"]
    snaps = set()
    if not selftest:
        for v in ("V1", "A1"):
            snaps |= {r.get("model_snapshot") for r in _cache(v).values()}
    def sha(path):
        return hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
    return {"cards_sha256": sha("data/eval/requirement_cards.json"),      # 둘 다 .gitignore 대상이라 해시로 고정한다
            "law_json_sha256": sha("data/raw/law/260021.json"),
            "b0_model_note": B0_MODEL_NOTE,
            "git_head": git("rev-parse", "HEAD"), "dirty_files": git("status", "--porcelain", "--", *watched).splitlines(),
            "prompt_version": PROMPT_VERSION, "model": MODEL, "model_snapshots": sorted(x for x in snaps if x)}


def _b2(clauses: list[dict], kept: list[set[str]]) -> dict[bool, np.ndarray]:
    """kept가 제거한 (제7~14조, 제6조) 라벨 수에 맞춰 forward 라벨을 무작위 제거. 시드 1,000의 조항별 기대 (TP,FP,FN)."""
    pairs = [(i, a) for i, c in enumerate(clauses) for a in c["fwd"]]
    p6 = [k for k, (_, a) in enumerate(pairs) if a == ART6]
    p7 = [k for k, (_, a) in enumerate(pairs) if a != ART6]
    rem6 = sum(1 for i, a in pairs if a == ART6 and a not in kept[i])
    rem7 = sum(1 for i, a in pairs if a != ART6 and a not in kept[i])
    acc = {True: np.zeros((len(clauses), 3)), False: np.zeros((len(clauses), 3))}
    f1s = {True: [], False: []}
    for seed in range(B2_SEEDS):
        rng = np.random.default_rng(seed)
        gone = set(rng.choice(p7, rem7, replace=False).tolist()) | set(rng.choice(p6, rem6, replace=False).tolist())
        ks = [set() for _ in clauses]
        for k, (i, a) in enumerate(pairs):
            if k not in gone:
                ks[i].add(a)
        for d6 in (True, False):
            cnt = _counts(clauses, ks, d6)
            acc[d6] += cnt / B2_SEEDS
            f1s[d6].append(_f1(cnt))
    return {d6: (acc[d6], np.array(f1s[d6])) for d6 in (True, False)}


def _boot_diff(a: np.ndarray, b: np.ndarray) -> dict:
    """micro F1 차(a − b)의 조항 단위 paired bootstrap."""
    rng = np.random.default_rng(SEED)
    n = len(a)
    idx = rng.integers(0, n, (BOOT_N, n))
    d = np.array([_f1(a[i]) - _f1(b[i]) for i in idx]) * 100
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff_pp": float((_f1(a) - _f1(b)) * 100), "ci95_pp": [float(lo), float(hi)]}


def score(selftest: bool = False) -> dict:
    cards, clauses = load_cards(), load_clauses()
    n = len(clauses)
    fwd = [set(c["fwd"]) for c in clauses]
    b1 = [set(c["fwd"]) if c["b1_pass"] else set() for c in clauses]
    if selftest:
        dec = {"V1": (fwd, {"pairs": 0, "parse_fail": 0}), "A1": (fwd, {"pairs": 0, "parse_fail": 0})}
    else:
        dec = {v: _decisions(v, clauses, cards) for v in ("V1", "A1")}
    res: dict = {"prompt_version": PROMPT_VERSION, "n_clauses": n, "selftest": selftest, "parse": {v: dec[v][1] for v in dec},
                 "repro": _repro(selftest)}
    cond = {"B0": fwd, "B1": b1, "V1": dec["V1"][0], "A1": dec["A1"][0]}
    b2 = {"B2_V1": _b2(clauses, dec["V1"][0]), "B2_A1": _b2(clauses, dec["A1"][0])}
    for d6, scope in ((True, "제6조제외"), (False, "전체")):
        cnt = {k: _counts(clauses, v, d6) for k, v in cond.items()}
        cnt.update({k: v[d6][0] for k, v in b2.items()})
        table = {}
        for k, c in cnt.items():
            p, r = _prec_rec(c)
            table[k] = {"f1": _f1(c) * 100, "precision": p * 100, "recall": r * 100}
        for k in ("V1", "A1"):
            kept_labels = sum(len([a for a in s if (a != ART6) or not d6]) for s in cond[k])
            all_labels = sum(len([a for a in s if (a != ART6) or not d6]) for s in fwd)
            table[k]["retention"] = kept_labels / all_labels * 100
            f1s = b2["B2_" + k][d6][1] * 100
            table[k]["b2_mean_f1"] = float(f1s.mean())
            table[k]["pct_in_b2"] = float((f1s < table[k]["f1"]).mean() * 100)
        res[scope] = {"conditions": table, "diffs": {
            "V1-B0 (주 판정)": _boot_diff(cnt["V1"], cnt["B0"]),
            "V1-A1 (법조문 효과)": _boot_diff(cnt["V1"], cnt["A1"]),
            "V1-B1": _boot_diff(cnt["V1"], cnt["B1"]),
            "V1-B2(기대값)": _boot_diff(cnt["V1"], b2["B2_V1"][d6][0]),
            "A1-B0": _boot_diff(cnt["A1"], cnt["B0"]),
            "B1-B0": _boot_diff(cnt["B1"], cnt["B0"])}}
    # 조별 F1 (제8·9조를 따로)
    per = {}
    for art in ("제7조", "제8조", "제9조", "제10조", "제11조", "제12조", "제13조", "제14조", ART6):
        row = {}
        for k, v in cond.items():
            tp = sum(art in c["ref"] and art in s for c, s in zip(clauses, v))
            fp = sum(art not in c["ref"] and art in s for c, s in zip(clauses, v))
            fn = sum(art in c["ref"] and art not in s for c, s in zip(clauses, v))
            row[k] = 2 * tp / (2 * tp + fp + fn) * 100 if (2 * tp + fp + fn) else None
        per[art] = row
    res["조별_F1"] = per
    # 민감도 (부록)
    if not selftest:
        res["민감도"] = {}
        for rule, variants in (("drop_undecided", ("V1", "A1")), ("v1_old", ("V1",))):
            for v in variants:
                k, _ = _decisions(v, clauses, cards, rule)
                res["민감도"][f"{v}/{rule}"] = {"제6조제외_f1": _f1(_counts(clauses, k, True)) * 100,
                                              "전체_f1": _f1(_counts(clauses, k, False)) * 100}
    # 손익분기 대조
    b0 = _counts(clauses, fwd, True)
    tp0, fp0, fn0 = b0.sum(0)
    k = res["제6조제외"]["conditions"]["V1"]["retention"] / 100 * (tp0 + fp0)
    need = _f1(b0) * (k + tp0 + fn0) / 2
    res["손익분기"] = {"V1_유지라벨": float(k), "필요_정밀도": float(need / k * 100) if k else None,
                      "V1_실제_정밀도": res["제6조제외"]["conditions"]["V1"]["precision"]}
    # 판정 (사전 등록 §8.3)
    d = res["제6조제외"]["diffs"]["V1-B0 (주 판정)"]
    res["판정"] = ("중단 — V1이 B0를 넘지 못함(점추정 ≤ 0), 반복 실행 불필요" if d["diff_pp"] <= 0 else
                  "CI 하한 > 0 — 성공 주장 전에 V1·B0 반복 실행으로 실행 간 변동 확인" if d["ci95_pp"][0] > 0 else
                  "미판정 — 점추정은 양수이나 CI가 0을 포함(검정력 한계 명시)")
    if not selftest:
        save_json(res, OUT_DIR / "result.json")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


def selftest() -> None:
    """V1=A1=전부 유지이면 B0와 같고 V1−B0=0이어야 한다. B1은 약 43.1%(제6조 제외), B0는 54.6%여야 한다."""
    res = score(selftest=True)
    c = res["제6조제외"]["conditions"]
    assert abs(c["B0"]["f1"] - c["V1"]["f1"]) < 1e-9 and res["제6조제외"]["diffs"]["V1-B0 (주 판정)"]["diff_pp"] == 0
    assert res["n_clauses"] == 255 and abs(c["B0"]["f1"] - 54.6) < 0.1 and abs(c["B1"]["f1"] - 43.1) < 0.1, c
    print("\nselftest OK — B0 %.1f · B1 %.1f · V1(전부 유지) %.1f" % (c["B0"]["f1"], c["B1"]["f1"], c["V1"]["f1"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--score", action="store_true")
    ap.add_argument("--gold-once-ok", action="store_true", help="gold를 한 번 쓰는 실행임을 확인")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.run:
        if not a.gold_once_ok:
            raise SystemExit("gold 255를 한 번 쓰는 실행이다 — 확인 후 --gold-once-ok를 붙여 다시 실행")
        run()
    else:
        score()
