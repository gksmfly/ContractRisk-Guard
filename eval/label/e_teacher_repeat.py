# eval/label/e_teacher_repeat.py
"""GPT 반복 실행 — GPT 우위의 실행 변동 확인 (사전 등록: eval/paper_plan.md §7-4, 2026-10-06)

gpt-4o forward(fwd-v5, temperature 0)를 gold clean 255 + 홀드아웃 300에 **5회 독립 실행**하고,
배포 KoELECTRA(article_v2)와의 차를 실행 간 분산까지 넣어 판정한다.

    .venv/bin/python -m eval.label.e_teacher_repeat --run          # API 호출 (실행별 캐시, 끊겨도 이어서)
    .venv/bin/python -m eval.label.e_teacher_repeat --analyze      # API 0
    .venv/bin/python -m eval.label.e_teacher_repeat --analyze-4    # 사전 등록 이탈판 (아래)

## 사전 등록 이탈 (2026-10-07, 사용자 결정)

크레딧이 두 번 소진돼 10-07 실행은 2회(1·2회차)만 끝났다. 남은 3회에 약 12달러가 더 드는데, 2회 + 기존 실행 2개
(08-30 teacher 캐시, FB-Check v5 forward — 같은 gpt-4o·fwd-v5·temperature 0)로 본 **실행 간 표준편차가 0.5~0.6%p**로
평가셋 표본 오차보다 훨씬 작아 3회를 더해도 구간이 거의 줄지 않는다. 그래서 **4회로 판정**하고 이 사실을 문서에 밝힌다.
"""

import argparse
import json
import os
import time

import numpy as np

from backend.training.retrain import eval_sets, f1_rows
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json
from eval.analysis.confound_articles import _f1
from eval.model.e8b_matched import COMMON, _groups, _load

logger = load_logger("e_teacher_repeat.log")

HOLDOUT = PROJECT_ROOT / "backend/training/negative_holdout_docs.json"
OUT_DIR = PROJECT_ROOT / "data/eval/teacher_repeat"
N_RUNS  = 5
MODEL   = "gpt-4o"          # .env의 FORWARD_MODEL(mini일 수 있음)을 쓰지 않는다
# 요금제 한도 TPM 30,000 · 호출당 약 2,500토큰 → 분당 12건 남짓. 동시 호출하면 429가 재시도까지 소진해
# 빈 판정이 결과에 섞인다(10-06 첫 시도에서 확인). 한 건씩, 호출 간격을 둔다
MIN_INTERVAL_S = 5.5


def _items() -> list[dict]:
    gold, neg = eval_sets(json.loads(HOLDOUT.read_text(encoding="utf-8")))
    return ([{"key": f"gold|{i}", "text": g["text"]} for i, g in enumerate(gold)] +
            [{"key": f"neg|{r['chunk_id']}", "text": r["text"]} for r in neg])


def run() -> None:
    from dotenv import load_dotenv
    from openai import OpenAI

    from backend.fb_check.forward_labeling import PROMPT_VERSION, run_forward
    load_dotenv(PROJECT_ROOT / ".env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = _items()
    for k in range(1, N_RUNS + 1):
        path = OUT_DIR / f"run{k}.jsonl"
        for attempt in range(5):                                   # 실패분은 쓰지 않고 다음 바퀴에 다시 시도
            done = {r["key"] for r in load_jsonl(path)} if path.exists() else set()
            todo = [it for it in items if it["key"] not in done]
            logger.info(f"=== 실행 {k}/{N_RUNS} (바퀴 {attempt + 1}): 남음 {len(todo)} ===")
            if not todo:
                break
            with open(path, "a", encoding="utf-8") as f:
                for i, it in enumerate(todo, 1):
                    t0 = time.time()
                    out = run_forward(client, it["text"], model=MODEL)
                    if out is not None:
                        f.write(json.dumps({"key": it["key"], "articles": out.get("articles", []), "ok": True,
                                            "model": MODEL, "prompt": PROMPT_VERSION}, ensure_ascii=False) + "\n")
                        f.flush()
                    if i % 100 == 0:
                        logger.info(f"  [{i}/{len(todo)}]")
                    time.sleep(max(0.0, MIN_INTERVAL_S - (time.time() - t0)))


def analyze() -> None:
    gold, neg = eval_sets(json.loads(HOLDOUT.read_text(encoding="utf-8")))
    G = [frozenset(a for a in g["articles"] if a in COMMON) for g in gold]
    labels = np.array([[a in g for a in COMMON] for g in G])
    P, Pn, thr = _load("article_v2")
    k_f1 = f1_rows(P >= thr, labels)
    k_rec = (P >= thr).any(1).astype(float)
    k_fpr = (Pn >= thr).any(1).astype(float)
    rg = (P / thr).max(1)
    g_groups = _groups([g["doc_id"] for g in gold])
    n_groups = _groups([":".join(str(r["chunk_id"]).split(":")[:2]) for r in neg])

    runs = []
    for k in range(1, N_RUNS + 1):
        rows = {r["key"]: r for r in load_jsonl(OUT_DIR / f"run{k}.jsonl")}
        n_fail = sum(not r["ok"] for r in rows.values())
        T = [frozenset(a for a in rows[f"gold|{i}"]["articles"] if a in COMMON) for i in range(len(gold))]
        t_f1 = np.array([_f1(t, g) if t else 0.0 for t, g in zip(T, G)])
        t_rec = np.array([bool(t) for t in T], dtype=float)
        t_fpr = np.array([bool([a for a in rows[f"neg|{r['chunk_id']}"]["articles"] if a in COMMON]) for r in neg], dtype=float)
        cut = np.sort(rg)[::-1][int(np.ceil(t_rec.mean() * len(rg))) - 1]       # 이 실행의 재현에 학생을 맞춘다
        m_f1 = f1_rows(P >= thr * cut, labels)
        m_fpr = (Pn >= thr * cut).any(1).astype(float)
        runs.append({"n_fail": n_fail, "T": T, "f1": t_f1, "rec": t_rec, "fpr": t_fpr, "m_f1": m_f1, "m_fpr": m_fpr})

    def combined(diffs: list[np.ndarray], groups: list[np.ndarray]) -> dict:
        """실행별 건별 차 → 실행 평균 차. CI = 평균 ± 1.96·√(군집 부트스트랩 분산 + 실행 간 분산/n)."""
        mean_diff = np.mean(diffs, axis=0)
        rng = np.random.default_rng(42)
        boot = [mean_diff[np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])].mean()
                for _ in range(3000)]
        per_run = [d.mean() for d in diffs]
        se = float(np.sqrt(np.var(boot) + np.var(per_run, ddof=1) / len(per_run)))
        pt = float(mean_diff.mean())
        lo, hi = pt - 1.96 * se, pt + 1.96 * se
        return {"diff_pp": round(pt * 100, 2), "ci95_pp": [round(lo * 100, 2), round(hi * 100, 2)],
                "per_run_pp": [round(x * 100, 2) for x in per_run],
                "verdict": "GPT 우위" if lo > 0 else ("KoELECTRA 우위" if hi < 0 else "미판정")}

    # 실행 간 일치: 조 집합이 실행 1과 같은 비율
    agree = [float(np.mean([a == b for a, b in zip(runs[0]["T"], r["T"])])) for r in runs[1:]]
    rep = {
        "rule": "eval/paper_plan.md §7-4", "model": MODEL, "n_runs": N_RUNS,
        "n_gold": len(gold), "n_holdout": len(neg), "api_failures": [r["n_fail"] for r in runs],
        "gpt_per_run": [{"f1": round(float(r["f1"].mean()) * 100, 2), "recall": round(float(r["rec"].mean()) * 100, 2),
                         "holdout_fpr": round(float(r["fpr"].mean()) * 100, 2)} for r in runs],
        "gpt_mean_sd": {k: [round(float(np.mean([r[k].mean() for r in runs])) * 100, 2),
                            round(float(np.std([r[k].mean() for r in runs], ddof=1)) * 100, 2)] for k in ("f1", "rec", "fpr")},
        "run_agreement_vs_run1": [round(a, 4) for a in agree],
        "koelectra_deploy": {"f1": round(float(k_f1.mean()) * 100, 2), "recall": round(float(k_rec.mean()) * 100, 2),
                             "holdout_fpr": round(float(k_fpr.mean()) * 100, 2)},
        "f1_gpt_minus_deploy": combined([r["f1"] - k_f1 for r in runs], g_groups),
        "f1_gpt_minus_matched": combined([r["f1"] - r["m_f1"] for r in runs], g_groups),
        "recall_gpt_minus_deploy": combined([r["rec"] - k_rec for r in runs], g_groups),
        "fpr_gpt_minus_deploy": combined([r["fpr"] - k_fpr for r in runs], n_groups),
        "fpr_gpt_minus_matched": combined([r["fpr"] - r["m_fpr"] for r in runs], n_groups),
    }
    save_json(rep, PROJECT_ROOT / "data/eval/e_teacher_repeat.json")
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


def analyze_4() -> None:
    """이탈판: 10-07 1·2회차 + 08-30 teacher 캐시 + FB-Check v5 forward. 판정 규칙은 §7-4와 같다."""
    import re
    gold, _ = eval_sets(json.loads(HOLDOUT.read_text(encoding="utf-8")))
    G = [frozenset(a for a in g["articles"] if a in COMMON) for g in gold]
    labels = np.array([[a in g for a in COMMON] for g in G])
    P, _, thr = _load("article_v2")
    k_f1, rg = f1_rows(P >= thr, labels), (P / thr).max(1)
    groups = _groups([g["doc_id"] for g in gold])
    runs = {}
    for k in (1, 2):
        rows = {r["key"]: r for r in load_jsonl(OUT_DIR / f"run{k}.jsonl")}
        runs[f"10-07 #{k}"] = [frozenset(a for a in rows[f"gold|{i}"]["articles"] if a in COMMON) for i in range(len(gold))]
    old = {r["key"]: r for r in load_jsonl(PROJECT_ROOT / "data/eval/label/teacher_on_gold_raw.jsonl")}
    runs["08-30"] = [frozenset(a for a in old[g["doc_id"] + "|" + g["text"][:40]]["articles"] if a in COMMON) for g in gold]
    norm = lambda t: re.sub(r"\s+", "", t or "")  # noqa: E731
    fb = {norm(r["text"]): r for p in ("clean", "noise") for r in load_jsonl(PROJECT_ROOT / f"data/fb_check/{p}.jsonl")
          if r.get("source") == "ftc_case"}
    runs["FB-Check v5"] = [frozenset(a for a in fb[norm(g["text"])]["forward_articles"] if a in COMMON) for g in gold]

    per, dd, dm = {}, [], []
    for name, T in runs.items():
        tf = np.array([_f1(t, g) if t else 0.0 for t, g in zip(T, G)])
        rec = float(np.mean([bool(t) for t in T]))
        cut = np.sort(rg)[::-1][int(np.ceil(rec * len(rg))) - 1]
        mf = f1_rows(P >= thr * cut, labels)
        dd.append(tf - k_f1)
        dm.append(tf - mf)
        per[name] = {"f1": round(float(tf.mean()) * 100, 2), "recall": round(rec * 100, 2),
                     "vs_deploy_pp": round(float((tf - k_f1).mean()) * 100, 2),
                     "vs_matched_pp": round(float((tf - mf).mean()) * 100, 2)}
    rng = np.random.default_rng(42)

    def ci(D):
        m = np.mean(D, axis=0)
        b = [m[np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(3000)]
        run_sd = float(np.std([d.mean() for d in D], ddof=1))
        se = float(np.sqrt(np.var(b) + run_sd ** 2 / len(D)))
        pt = float(m.mean())
        return {"diff_pp": round(pt * 100, 2), "ci95_pp": [round((pt - 1.96 * se) * 100, 2), round((pt + 1.96 * se) * 100, 2)],
                "run_sd_pp": round(run_sd * 100, 2), "verdict": "GPT 우위" if pt - 1.96 * se > 0 else "미판정"}
    rep = {"rule": "eval/paper_plan.md §7-4 (이탈: 4회)", "runs": per,
           "f1_gpt_minus_deploy": ci(dd), "f1_gpt_minus_matched": ci(dm)}
    save_json(rep, PROJECT_ROOT / "data/eval/label/e_teacher_repeat_4runs.json")
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--analyze-4", action="store_true")
    args = ap.parse_args()
    if args.analyze_4:
        analyze_4()
    if args.run:
        run()
    if args.analyze:
        analyze()


if __name__ == "__main__":
    main()
