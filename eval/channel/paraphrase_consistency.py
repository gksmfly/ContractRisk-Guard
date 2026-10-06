# eval/channel/paraphrase_consistency.py
"""Paraphrase 강건성 — 같은 뜻으로 바꿔 쓴 조항에서 판정이 유지되나 (2026-10-05)

## 질문

불공정 조항 분류기는 문구 변형에 민감하다는 보고가 있다. 같은 조항을 **뜻은 그대로 두고 어휘·문장
구조만 바꿔** 넣었을 때 (1) 위반 조항을 계속 짚는가(재현 유지) (2) 표준계약서에 새로 경보를 내지
않는가(오경보 유지) (3) 지목하는 조가 유지되는가.

B4(문체 변환, `style_transfer_fpr.py`)는 **음성만**, GPT로 바꿨다. 여기서는 양성(gold 255)까지 넣고
로컬 모델(API 0)로 바꾼다.

## 설계 (사전 등록)

- 대상: gold clean 255(위반) + 음성 홀드아웃 300(표준계약서 12문서). 원문 1,500자 초과는 제외(개수 보고)
- 바꿔 쓰기: Qwen2.5-14B-Instruct 4bit, greedy. 당사자·권리의무·조건·기한·숫자·단서는 유지, 어휘·어순·제목 표현만 변경
- **자동 품질 검사(사람 검수 불가 — 한계로 명시):**
  숫자 보존(숫자열 멀티셋 동일) · 길이 비율 0.6~1.6 · 원문과 문자 유사도 < 0.9(실제로 바뀌었나)
  → 통과 쌍을 주 분석, 전체를 병기
- 모델: KoELECTRA `article_v2`(배포) · KLUE-RoBERTa-base(E8 RB, 3시드) · KoELECTRA C 팔(3시드) ·
  TF-IDF+LR(balanced, 어휘 의존 대조군). 전부 **자기 배포 임계값 그대로**
- 지표 (원문 vs 변형문, 건별 짝):
  - 판정 뒤집힘률: "조를 하나라도 냈나"가 바뀐 비율 (gold·홀드아웃 따로)
  - 재현 변화(gold) · 오경보 변화(홀드아웃): McNemar + 문서 군집 부트스트랩 CI
  - 조 집합 일치(Jaccard, 둘 다 뭔가 낸 건에서) · 조 F1 변화(gold)
- 판정: 재현·오경보 변화의 CI가 0을 포함하면 "강건(변화 미검출, MDE 병기)". 모델 간 비교는 뒤집힘률 차이의 CI

## 실행

    python -m eval.channel.paraphrase_consistency --generate --gpu 0     # 바꿔 쓰기 (캐시)
    python -m eval.channel.paraphrase_consistency --analyze --gpu 0      # 모델 채점
"""

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from backend.training.retrain import eval_sets, f1_rows, predict
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("paraphrase_consistency.log")

HOLDOUT  = PROJECT_ROOT / "backend/training/negative_holdout_docs.json"
CACHE    = PROJECT_ROOT / "data/eval/paraphrase_raw.jsonl"
OUT_PATH = PROJECT_ROOT / "data/eval/paraphrase_consistency.json"
MAX_CHARS = 1500
MODEL_KEY = "qwen2.5-14b"

SYSTEM = """당신은 한국어 계약서 문장을 같은 뜻으로 다시 쓰는 편집자입니다.

규칙:
1. 당사자, 권리와 의무, 조건, 기한, 금액·비율·횟수 같은 숫자, 예외(단서 조항)를 하나도 바꾸거나 빼거나 더하지 마세요.
2. 숫자는 원문에 적힌 그대로(아라비아 숫자는 아라비아 숫자로) 쓰세요.
3. 어휘와 문장 구조는 최대한 바꾸세요 — 동의어, 어순, 능동/수동, 문장 나누기·합치기.
4. 조항 번호는 그대로 둬도 되지만 제목 표현은 바꾸세요.
5. 법률적 평가나 설명을 덧붙이지 마세요. 원문에 없는 내용을 넣지 마세요.

반드시 다음 JSON 형식으로만 답하세요: {"paraphrase": "다시 쓴 조항 전문"}"""

_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def _items() -> list[dict]:
    gold, neg = eval_sets(json.loads(HOLDOUT.read_text(encoding="utf-8")))
    rows = [{"key": f"gold|{g['doc_id']}|{i}", "kind": "gold", "group": g["doc_id"], "text": g["text"],
             "articles": g["articles"]} for i, g in enumerate(gold)]
    rows += [{"key": f"neg|{r['chunk_id']}", "kind": "neg", "group": ":".join(str(r["chunk_id"]).split(":")[:2]),
              "text": r["text"], "articles": []} for r in neg]
    return rows


def quality(orig: str, para: str) -> dict:
    nums_o, nums_p = sorted(_NUM.findall(orig)), sorted(_NUM.findall(para))
    ratio = len(para) / max(len(orig), 1)
    sim = SequenceMatcher(None, re.sub(r"\s+", "", orig), re.sub(r"\s+", "", para)).ratio()
    ok = {"numbers_kept": nums_o == nums_p, "length_ok": 0.6 <= ratio <= 1.6, "changed": sim < 0.9}
    return {**ok, "pass": all(ok.values()), "len_ratio": round(ratio, 3), "char_sim": round(sim, 3)}


# ── 생성 ─────────────────────────────────────────────────────────────────────

def generate(gpu: int) -> None:
    from eval.label.e9_local_labeling import _generate, load_model

    done = {r["key"] for r in load_jsonl(CACHE)} if CACHE.exists() else set()
    todo = [r for r in _items() if r["key"] not in done and len(r["text"]) <= MAX_CHARS]
    logger.info(f"=== 바꿔 쓰기 [{MODEL_KEY}] 남음 {len(todo)} (완료 {len(done)}) ===")
    if not todo:
        return
    model, tok, dev = load_model(MODEL_KEY, gpu)
    with open(CACHE, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            n_tok = len(tok(r["text"])["input_ids"])
            out = _generate(model, tok, dev, SYSTEM, [], f"계약 조항:\n{r['text']}",
                            max_new_tokens=min(1500, int(n_tok * 1.8) + 64))
            para = (out or {}).get("paraphrase") if isinstance(out, dict) else None
            row = {"key": r["key"], "paraphrase": para if isinstance(para, str) else None}
            if row["paraphrase"]:
                row["quality"] = quality(r["text"], row["paraphrase"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(todo):
                logger.info(f"  [{i}/{len(todo)}]")


# ── 채점 ─────────────────────────────────────────────────────────────────────

def _tfidf_predictor():
    from backend.eval.label.e_base_baselines import HEADS, _train_tfidf_lr, _training_split
    from backend.training.train_article import tune_thresholds
    train, dev = _training_split()
    vec, clf, mlb = _train_tfidf_lr(train, balanced=True)
    dp = clf.predict_proba(vec.transform([r["text"] for r in dev]))
    dy = np.array([[a in r["articles"] for a in mlb.classes_] for r in dev], dtype=float)
    thr = tune_thresholds(dp, dy, list(mlb.classes_))
    return lambda texts: (clf.predict_proba(vec.transform(texts)) >= thr), list(HEADS)


def _neural_predictor(model_dir: Path, gpu: int):
    def f(texts):
        P, thr, _ = predict(model_dir, texts, gpu)
        return P >= thr
    names = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))["article_names"]
    return f, list(names)


def _cluster_ci(diff: np.ndarray, groups: list[str], n: int = 3000, seed: int = 42):
    idx: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    cl = [np.array(v) for v in idx.values()]
    rng = np.random.default_rng(seed)
    b = [diff[np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])].mean() for _ in range(n)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return [round(float(lo) * 100, 2), round(float(hi) * 100, 2)]


def _mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy.stats import binomtest
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"orig_only": n10, "para_only": n01, "p": round(float(p), 4)}


def score(pred_o: np.ndarray, pred_p: np.ndarray, names: list[str], rows: list[dict]) -> dict:
    kind = np.array([r["kind"] for r in rows])
    groups = [r["group"] for r in rows]
    fo, fp = pred_o.any(1), pred_p.any(1)
    out = {}
    for k in ("gold", "neg"):
        m = kind == k
        g = [x for x, mm in zip(groups, m) if mm]
        diff = fp[m].astype(float) - fo[m].astype(float)
        out[k] = {"n": int(m.sum()), "rate_orig": round(float(fo[m].mean()), 4), "rate_para": round(float(fp[m].mean()), 4),
                  "change_pp": round(float(diff.mean()) * 100, 2), "ci95_pp": _cluster_ci(diff, g),
                  "flip_rate": round(float((fo[m] != fp[m]).mean()), 4), "mcnemar": _mcnemar(fo[m], fp[m])}
    both = fo & fp
    jac = [len(set(np.where(a)[0]) & set(np.where(b)[0])) / len(set(np.where(a)[0]) | set(np.where(b)[0]))
           for a, b in zip(pred_o[both], pred_p[both])]
    out["article_jaccard_when_both_flag"] = round(float(np.mean(jac)), 4) if jac else None
    gm = kind == "gold"
    labels = np.array([[a in r["articles"] for a in names] for r in rows])[gm]
    f_o, f_p = f1_rows(pred_o[gm], labels), f1_rows(pred_p[gm], labels)
    out["gold_f1"] = {"orig": round(float(f_o.mean()) * 100, 2), "para": round(float(f_p.mean()) * 100, 2),
                      "change_ci95_pp": _cluster_ci(f_p - f_o, [g for g, mm in zip(groups, gm) if mm])}
    return out


def analyze(gpu: int) -> None:
    items = {r["key"]: r for r in _items()}
    cache = [r for r in load_jsonl(CACHE) if r.get("paraphrase")]
    n_long = sum(len(r["text"]) > MAX_CHARS for r in items.values())
    rows_all = [{**items[c["key"]], "para": c["paraphrase"], "q": c["quality"]} for c in cache]
    rows_pass = [r for r in rows_all if r["q"]["pass"]]
    q = {k: int(sum(r["q"][k] for r in rows_all)) for k in ("numbers_kept", "length_ok", "changed", "pass")}
    logger.info(f"쌍 {len(rows_all)} (1,500자 초과 제외 {n_long}) · 품질 통과 {q}")

    models = {"KoELECTRA article_v2": [PROJECT_ROOT / "models/article_v2"],
              "KoELECTRA C 팔 (3시드)": [PROJECT_ROOT / f"models/_e8_C_s{s}" for s in (42, 7, 123)],
              "KLUE-RoBERTa RB (3시드)": [PROJECT_ROOT / f"models/_e8_RB_s{s}" for s in (42, 7, 123)]}
    report = {"population": f"gold 255 + 홀드아웃 300 중 1,500자 이하, 바꿔 쓰기 성공 {len(rows_all)}쌍",
              "paraphraser": f"{MODEL_KEY} (4bit, greedy)", "quality": q, "excluded_long": n_long, "models": {}}
    preds = {}
    for subset_name, rows in (("pass", rows_pass), ("all", rows_all)):
        texts_o, texts_p = [r["text"] for r in rows], [r["para"] for r in rows]
        for name, dirs in models.items():
            dirs = [d for d in dirs if (d / "metrics.json").exists()]
            if not dirs:
                continue
            per = []
            for d in dirs:
                fn, names = _neural_predictor(d, gpu)
                po, pp = fn(texts_o), fn(texts_p)
                per.append(score(po, pp, names, rows))
                preds[(subset_name, name, d.name)] = (po, pp)
            report["models"].setdefault(name, {})[subset_name] = per if len(per) > 1 else per[0]
        fn, names = _tfidf_predictor()
        report["models"].setdefault("TF-IDF+LR (balanced)", {})[subset_name] = score(fn(texts_o), fn(texts_p), names, rows)
    save_json(report, OUT_PATH)
    for name, v in report["models"].items():
        s = v["pass"] if isinstance(v["pass"], dict) else v["pass"][0]
        logger.info(f"  [{name}] gold 재현 {s['gold']['rate_orig']}→{s['gold']['rate_para']} {s['gold']['ci95_pp']} "
                    f"뒤집힘 {s['gold']['flip_rate']} | 오경보 {s['neg']['rate_orig']}→{s['neg']['rate_para']} "
                    f"{s['neg']['ci95_pp']} | 조 Jaccard {s['article_jaccard_when_both_flag']}")
    logger.info(f"저장: {OUT_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    if args.generate:
        generate(args.gpu)
    if args.analyze:
        analyze(args.gpu)


if __name__ == "__main__":
    main()
