# eval/label/recall_check_gpt.py
"""누락 검사 ② — 운영 GPT-4o 학습 데이터 (사전 등록: eval/paper_plan.md §7-20, 2026-10-12, API 0)

문서 단위 5겹: 겹마다 나머지 겹의 FB-Check CLEAN(forward 라벨)으로 KoELECTRA를 학습해 빼 둔 겹을 예측한다.
대조군은 같은 겹 구조의 TF-IDF 출처 판별기(문체만 본다). 출처 가정으로 채점한다.

    .venv/bin/python -m eval.label.recall_check_gpt --train --gpu 1   # 5겹 학습·예측 (이어서 가능)
    .venv/bin/python -m eval.label.recall_check_gpt --analyze
"""

import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from backend.training.retrain import predict
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json, save_jsonl
from eval.model.e8_downstream_ablation import _doc_of, prepare_arms, train_arm

logger = load_logger("recall_check_gpt.log")
DIR = PROJECT_ROOT / "data/eval/label/rc_gpt"
OUT = PROJECT_ROOT / "data/eval/label/recall_check_gpt.json"
K, SEED, BOOT = 5, 42, 5000
HOLD = set(json.loads((PROJECT_ROOT / "backend/training/negative_holdout_docs.json").read_text(encoding="utf-8")))


def records() -> list[dict]:
    return load_jsonl(prepare_arms()["A"])


def folds(recs: list[dict]) -> np.ndarray:
    docs = sorted({_doc_of(r["chunk_id"]) for r in recs})
    rng = np.random.default_rng(SEED)
    rng.shuffle(docs)
    f = {d: i % K for i, d in enumerate(docs)}
    return np.array([f[_doc_of(r["chunk_id"])] for r in recs])


def train(gpu: int) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    recs = records()
    fo = folds(recs)
    for k in range(K):
        out = DIR / f"pred_fold{k}.npz"
        if out.exists():
            logger.info(f"  fold {k}: 예측 있음 — 건너뜀")
            continue
        # 고정 음성 홀드아웃 문서는 학습기가 반드시 찾아서 떼어 낸다(없으면 멈춘다) — 겹과 무관하게 넣는다.
        # 학습기가 학습에서 빼므로 빼 둔 겹 예측에 새지 않는다
        tr = [r for r, f in zip(recs, fo)
              if (f != k and r.get("status") == "CLEAN") or _doc_of(r["chunk_id"]) in HOLD]
        path = DIR / f"train_fold{k}.jsonl"
        save_jsonl(tr, path)
        d = train_arm(f"RC{k}", SEED, path, gpu)       # D 팔과 같은 설정(forward 라벨, 음성 홀드아웃)
        idx = np.where(fo == k)[0]
        P, thr, names = predict(d, [recs[i]["text"] for i in idx], gpu)
        np.savez_compressed(out, idx=idx, P=P, thr=thr, names=np.array(names))
        w = d / "model.safetensors"
        if w.exists():
            w.unlink()
        logger.info(f"  fold {k}: 학습 {len(tr)} · 예측 {len(idx)} · 가중치 삭제")


def _cluster_ci(groups: list[str], fn) -> list[float]:
    idx: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    cl = [np.array(v) for v in idx.values()]
    rng = np.random.default_rng(42)
    b = [fn(np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])) for _ in range(BOOT)]
    return [round(float(x) * 100, 2) for x in np.percentile(b, [2.5, 97.5])]


def analyze() -> None:
    recs = records()
    n = len(recs)
    fo = folds(recs)
    enc = np.zeros(n, bool)
    for k in range(K):
        z = np.load(DIR / f"pred_fold{k}.npz")
        enc[z["idx"]] = (z["P"] >= z["thr"]).any(1)
    # 문체 기준선: 같은 겹 구조의 TF-IDF 출처 판별기 — 빼 둔 겹의 p(공정위)
    is_ftc = np.array([r["source"] == "ftc_case" for r in recs])
    texts = [r["text"] for r in recs]
    p_ftc = np.zeros(n)
    for k in range(K):
        tr, te = fo != k, fo == k
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=50_000, sublinear_tf=True)
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(vec.fit_transform([texts[i] for i in np.where(tr)[0]]), is_ftc[tr])
        p_ftc[te] = clf.predict_proba(vec.transform([texts[i] for i in np.where(te)[0]]))[:, 1]

    clean = np.array([r.get("status") == "CLEAN" for r in recs])
    lab = np.array([bool(r.get("forward_articles")) for r in recs])
    empty = clean & ~lab
    flag = empty & enc
    m = int(flag.sum())
    order = np.argsort(-np.where(empty, p_ftc, -1))      # 빈 라벨 중 p(공정위) 상위 m개
    style = np.zeros(n, bool)
    style[order[:m]] = True
    groups = [_doc_of(r["chunk_id"]) for r in recs]

    def prec(i, mask):
        s = mask[i]
        return float(is_ftc[i][s].mean()) if s.any() else 0.0

    def style_top(i):
        e = np.where(empty[i])[0]
        mm = int(flag[i].sum())
        if mm == 0 or len(e) == 0:
            return 0.0
        top = e[np.argsort(-p_ftc[i][e])[:mm]]
        return float(is_ftc[i][top].mean())

    main_d = prec(np.arange(n), flag) - prec(np.arange(n), style)
    main_ci = _cluster_ci(groups, lambda i: prec(i, flag) - style_top(i))
    base = float(is_ftc[empty].mean())
    sub_ci = _cluster_ci(groups, lambda i: prec(i, flag) - (float(is_ftc[i][empty[i]].mean()) if empty[i].any() else 0.0))

    # 데이터 수준 추정 오류 (출처 가정): 유지 레코드 중 표준계약서 위반 라벨 + 공정위 빈 라벨
    def est_err(lbl):
        k = clean
        return {"false_alarm_std_pp": round(float(lbl[k & ~is_ftc].mean()) * 100, 2),
                "missed_ftc_pp": round(float((~lbl[k & is_ftc]).mean()) * 100, 2),
                "error_pp": round(float((lbl[k] != is_ftc[k]).mean()) * 100, 2)}
    lab2 = lab | flag
    rep = {"rule": "eval/paper_plan.md §7-20", "n": n, "n_clean": int(clean.sum()), "n_empty_clean": int(empty.sum()),
           "n_flagged": m, "flag_ftc_pp": round(prec(np.arange(n), flag) * 100, 2),
           "style_baseline_ftc_pp": round(prec(np.arange(n), style) * 100, 2), "base_rate_ftc_pp": round(base * 100, 2),
           "주_encoder_minus_style_pp": round(main_d * 100, 2), "주_ci95_pp": main_ci, "주_pass": main_ci[0] > 0,
           "보조_encoder_minus_base_ci95_pp": sub_ci, "보조_pass": sub_ci[0] > 0,
           "P1_fbcheck": est_err(lab), "P2_plus_recall_check": est_err(lab2),
           "encoder_flag_rate_all_pp": round(float(enc.mean()) * 100, 2)}
    rep["verdict"] = ("내용 신호 확인(문체 기준선 초과)" if rep["주_pass"] else "미판정 — 출처 문체로 설명될 수 있음") + \
                     (" · 기저율 초과" if rep["보조_pass"] else "")
    save_json(rep, OUT)
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--gpu", type=int, default=1)
    a = ap.parse_args()
    if a.train:
        train(a.gpu)
    if a.analyze:
        analyze()


if __name__ == "__main__":
    main()
