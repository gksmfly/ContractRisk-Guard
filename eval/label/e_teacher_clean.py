# eval/label/e_teacher_clean.py
"""E_teacher 재채점 — gpt-4o 직접 추론 vs KoELECTRA를 **같은 clean 255에서 짝지어** 비교 (P1-4)

## 왜 다시 재나 (2026-09-30)

`teacher_on_gold.py`(09-28 보고)는 비교가 세 겹으로 어긋나 있었다.

1. 모집단: teacher는 gold **327**(clean 255 + noisy 72, noisy는 채점 불가 지층)
2. 학생 값: 교차적합 임계값 F1(성능 주장 금지 값), 그것도 327 기준 숫자를 인자로 받음
3. 페어드 아님: 학생 건별 점수가 없어 평균 차만 냈다

지금은 teacher 캐시(`teacher_on_gold_raw.jsonl`, 327건)와 E8b 캐시(`article_v2`의 gold·홀드아웃
확률)가 있어서 **API 0으로** clean 255에서 건별로 짝지을 수 있다. 오경보도 같은 홀드아웃
300건에서 잴 수 있다 — 그 조항들의 gpt-4o forward 라벨이 FB-Check 산출물에 이미 있다.

## 지표

- 조 F1(건별, 빈 예측 0점) — gold 255. 상수(제6·8·9조) 병기
- 조항 재현(조를 하나라도 냈나) — gold 255
- 홀드아웃 오경보(조를 하나라도 냈나) — 표준계약서 12문서
- 차이는 gold 문서 / 홀드아웃 문서 군집 부트스트랩 95% CI

⚠ KoELECTRA는 dev 임계값(배포값), teacher는 임계값이 없다. 동작점이 다르므로 재현·오경보는
"각자의 배포 동작점"끼리의 비교이고, 우열 판정은 조 F1과 (재현, 오경보) 쌍을 함께 본다.

실행:
    python -m eval.label.e_teacher_clean
"""

import json

import numpy as np

from backend.training.retrain import eval_sets, f1_rows
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json
from eval.analysis.confound_articles import _f1, best_constant
from eval.model.e8b_matched import CACHE_DIR, COMMON, _groups, _load

logger = load_logger("e_teacher_clean.log")

RAW      = PROJECT_ROOT / "data/eval/label/teacher_on_gold_raw.jsonl"
HOLDOUT  = PROJECT_ROOT / "backend/training/negative_holdout_docs.json"
FB_PATHS = [PROJECT_ROOT / "data/fb_check/clean.jsonl", PROJECT_ROOT / "data/fb_check/noise.jsonl"]
OUT_PATH = PROJECT_ROOT / "data/eval/label/e_teacher_clean.json"


def _boot(a: np.ndarray, b: np.ndarray, groups: list[np.ndarray], seed: int = 42, n: int = 5000):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        d.append(a[i].mean() - b[i].mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff_pp": round(float(a.mean() - b.mean()) * 100, 2),
            "ci95_pp": [round(float(lo) * 100, 2), round(float(hi) * 100, 2)],
            "verdict": "A > B" if lo > 0 else ("B > A" if hi < 0 else "미판정")}


def main() -> None:
    gold, neg = eval_sets(json.loads(HOLDOUT.read_text(encoding="utf-8")))
    raw = {r["key"]: r for r in load_jsonl(RAW)}
    keys = [g["doc_id"] + "|" + g["text"][:40] for g in gold]
    missing = [k for k in keys if k not in raw]
    if missing:
        raise SystemExit(f"teacher 캐시에 없는 gold {len(missing)}건 — teacher_on_gold를 먼저 돌릴 것")

    G = [frozenset(g["articles"]) for g in gold]
    T = [frozenset(a for a in raw[k]["articles"] if a in COMMON) for k in keys]
    teacher_f1 = np.array([_f1(t, g) if t else 0.0 for t, g in zip(T, G)])
    teacher_rec = np.array([bool(t) for t in T], dtype=float)

    labels = np.array([[a in g for a in COMMON] for g in G])
    P, Pn, thr = _load("article_v2")
    student_f1 = f1_rows(P >= thr, labels)
    student_rec = (P >= thr).any(1).astype(float)
    student_fpr = (Pn >= thr).any(1).astype(float)

    fb = {r["chunk_id"]: r for p in FB_PATHS for r in load_jsonl(p)}
    teacher_fpr = np.array([bool([a for a in (fb[r["chunk_id"]].get("forward_articles") or []) if a in COMMON])
                            for r in neg], dtype=float)

    # 동작점 맞춤(규칙 9): 학생 임계값에 배율을 곱해 teacher의 재현율에 맞춘다(비교량 — 배포에 안 씀)
    rg = (P / thr).max(1)
    cut = np.sort(rg)[::-1][int(np.ceil(teacher_rec.mean() * len(rg))) - 1]
    m_pred = P >= thr * cut
    m_f1, m_rec = f1_rows(m_pred, labels), m_pred.any(1).astype(float)
    m_fpr = (Pn >= thr * cut).any(1).astype(float)

    cf, cset = best_constant([("x", g) for g in G])
    const_f1 = np.array([_f1(cset, g) for g in G])

    g_groups = _groups([g["doc_id"] for g in gold])
    n_groups = _groups([":".join(str(r["chunk_id"]).split(":")[:2]) for r in neg])
    report = {
        "population": f"gold clean {len(gold)} · 홀드아웃 {len(neg)}({len(n_groups)}문서)",
        "teacher": {"model": "gpt-4o forward (fwd-v5)", "f1": round(float(teacher_f1.mean()), 4),
                    "clause_recall": round(float(teacher_rec.mean()), 4),
                    "holdout_fpr": round(float(teacher_fpr.mean()), 4)},
        "student": {"model": "article_v2 (dev 임계값 = 배포)", "f1": round(float(student_f1.mean()), 4),
                    "clause_recall": round(float(student_rec.mean()), 4),
                    "holdout_fpr": round(float(student_fpr.mean()), 4)},
        "constant": {"articles": sorted(cset), "f1": round(float(cf), 4)},
        "teacher_vs_constant_f1": _boot(teacher_f1, const_f1, g_groups),
        "student_vs_constant_f1": _boot(student_f1, const_f1, g_groups),
        "teacher_vs_student": {"f1": _boot(teacher_f1, student_f1, g_groups),
                               "clause_recall": _boot(teacher_rec, student_rec, g_groups),
                               "holdout_fpr": _boot(teacher_fpr, student_fpr, n_groups)},
        "matched_to_teacher_recall": {
            "scale": round(float(cut), 3),
            "student": {"f1": round(float(m_f1.mean()), 4), "clause_recall": round(float(m_rec.mean()), 4),
                        "holdout_fpr": round(float(m_fpr.mean()), 4)},
            "teacher_minus_student_f1": _boot(teacher_f1, m_f1, g_groups),
            "teacher_minus_student_fpr": _boot(teacher_fpr, m_fpr, n_groups),
        },
        "note": "teacher는 임계값 없음, 학생은 dev 임계값. 동작점이 달라 재현·오경보는 쌍으로 읽을 것",
        "cache_check": str(CACHE_DIR / "article_v2.npz"),
    }
    save_json(report, OUT_PATH)
    logger.info(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
