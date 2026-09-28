# backend/training/retrain.py
"""
새 라벨이 들어왔을 때의 재학습 — **데이터만 바꾸고** 학습 → 무해 확인 → 기록 → (명시하면) 승격.

## 순서가 곧 판정 순서다

    1. 스냅숏      라벨 파일 지문·건수·빈 라벨 비율·조 비중·라벨러 출처
    2. 변경 확인    현행 모델이 학습한 지문과 같으면 멈춘다 (--force로 무시)
    3. 라벨러 확인  라벨러 조합(forward/verify 모델·프롬프트)이 하나가 아니거나 이전 기록과
                   다르면 **멈춘다** (--allow-labeler-change로 무시)
    4. 분포 이동    빈 라벨 비율·조 비중이 5%p 넘게 움직이면 경고한다. 판정에는 안 쓴다
    5. 학습         현행 모델의 metrics.json에서 설정을 그대로 읽어 재현한다. 바뀌는 건 데이터뿐
    6. 무해 확인    사전 등록 규칙 (아래)
    7. 기록         `data/eval/retrain_history.jsonl`에 한 줄. 세대 추이가 곧 모니터링이다
    8. 승격         --promote-as를 줬고 6을 통과했을 때만. **서빙 전환(MODEL_DIR)은 사람이 한다**

## 사전 등록 판정 — 고르는 게 아니라 무해 확인이다

`article_v1 → v2` 승격(`maxlen_harm_check`)과 같은 형태다. 축을 하나 더 둔다:

    주 판정 ①  gold clean 255 조항 단위 재현     후보−현행 페어드 CI 상한 < 0  → 기각
    주 판정 ②  표준계약서 홀드아웃 12문서 오경보   후보−현행 문서 부트스트랩 CI 하한 > 0  → 기각
    기록       조 F1 차이 · 후보의 상수 대비 · 누적 추이 — 판정에 안 쓴다

**통과가 개선의 증거는 아니다.** gold 255로는 조 F1 +3%p도 가릴 수 없다(n≈3,400 필요,
`reported_numbers.md` §4). 데이터를 늘리는 이유는 분포 커버리지이고 그건 gold가 못 잰다.
그래서 판정은 "두 축 어디서도 유의하게 나빠지지 않았다"까지이고, 승격 기록에도 그렇게 남긴다.

⚠ 12문서 부트스트랩은 구간을 과소추정한다(`reported_numbers.md` §2). 구간이 좁으면
기각이 쉬워지므로 **보수적인 방향**의 오차다.

## 홀드아웃 문서를 고정하는 이유

`split_negative_holdout`의 시드 추첨은 그 시점 문서 목록 위에서 돈다. 라벨링이 진행돼
표준계약서 문서가 늘면 다른 12개가 뽑혀, 현행 모델이 학습한 문서가 후보의 평가셋이 된다.
그래서 목록을 `negative_holdout_docs.json`에 얼려 두고 두 세대 모두 그걸로 잰다.
(2026-09-28에 그날 데이터로 뽑은 목록이다. CLEAN 283건으로 `article_v2` 채점 기록의 건수와 같지만,
`article_v2` 학습 때 뗀 문서 ID는 남아 있지 않아 **문서 단위 일치는 확인하지 못했다.**)

## 모델 붕괴 모니터링이 왜 이 파일에 있나

붕괴는 모델이 자기 출력으로 다시 학습할 때 생긴다. 지금 라벨은 전부 gpt-4o의
forward∩verify이고 인코더는 투표하지 않는다(`backend/fb_check/__main__.py`). **붕괴 조건이
구조적으로 없다.** 대신 실제로 있는 위험이 둘이다:

- 라벨러가 조용히 바뀌는 것. `.env`의 gpt-4o-mini로 스모크가 돈 전례가 있다 → 3에서 멈춘다
- 한 세대씩은 유의하지 않은 하락이 누적되는 것. 6은 한 세대의 유의한 하락만 잡는다 → 7의
  이력에서 첫 기록 대비 누적 하락을 경고한다

인코더나 sLLM의 출력을 라벨에 넣게 되면(Flywheel) 그때가 붕괴 조건이고, 3의 라벨러 확인이
먼저 막는다.

실행:
    .venv/bin/python -m backend.training.retrain --check-only            # GPU 없이 1~4만
    .venv/bin/python -m backend.training.retrain --gpu 1                 # 학습·판정·기록
    .venv/bin/python -m backend.training.retrain --gpu 1 --promote-as models/article_v3
"""

import argparse
import datetime
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from backend.training.train_article import (
    LABELED_PATH,
    file_sha256,
    load_article_records,
    load_ftc_gold,
)
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger

logger = load_logger("retrain.log")

BASE_DIR = Path(os.environ.get("MODEL_DIR", str(PROJECT_ROOT / "models/article_v2")))
HISTORY_PATH = Path(os.environ.get("RETRAIN_HISTORY", str(PROJECT_ROOT / "data/eval/retrain_history.jsonl")))
NOISE_PATH = Path(os.environ.get("NOISE_LABELS", str(PROJECT_ROOT / "data/fb_check/noise.jsonl")))
HOLDOUT_DOCS_PATH = Path(__file__).with_name("negative_holdout_docs.json")

# 분포 이동 경고선. 판정이 아니라 "사람이 한 번 볼 것"의 기준이다
_DRIFT_PP = 5.0
# 첫 기록 대비 누적 하락 경고선(%p). 한 세대의 유의성 검정이 못 잡는 느린 하락용
_CUMULATIVE_PP = 3.0


# ─────────────────────────────────────────────────────────────────────────────
# 1~4. 데이터 확인 (GPU 불필요)
# ─────────────────────────────────────────────────────────────────────────────

def labeler_key(row: dict) -> str:
    """라벨을 만든 조합. 모델이 같아도 프롬프트 버전이 바뀌면 다른 라벨러다."""
    return "|".join(str(row.get(k) or "?") for k in
                    ("forward_model", "forward_prompt", "verify_model", "verify_prompt"))


def snapshot(path: Path, label_source: str = "agreed") -> dict:
    """라벨 파일의 요약. `chunk_ids`는 이력에 따로 저장하고 나머지는 한 줄로 남긴다."""
    rows = load_jsonl(path)
    recs = load_article_records(path, label_source=label_source)
    n = max(1, len(recs))
    arts = Counter(a for r in recs for a in r["articles"])
    return {
        "sha256": file_sha256(path),
        "rows": len(rows),
        "records": len(recs),
        "empty_ratio": sum(1 for r in recs if not r["articles"]) / n,
        "source_counts": dict(Counter(r["source"] for r in recs)),
        "article_share": {a: c / n for a, c in sorted(arts.items())},
        "labelers": dict(Counter(labeler_key(r) for r in rows)),
        "chunk_ids": sorted({str(r.get("chunk_id")) for r in rows}),
    }


def check_labelers(snap: dict, prev: dict | None) -> list[str]:
    """라벨러 조합 문제를 모은다. 비어 있으면 통과."""
    problems = []
    if len(snap["labelers"]) != 1:
        problems.append(f"라벨러 조합이 {len(snap['labelers'])}개다: {snap['labelers']} — "
                        f"한 학습셋에 서로 다른 라벨러가 섞이면 라벨 정의가 조용히 바뀐다")
    if prev and set(snap["labelers"]) != set(prev.get("labelers") or {}):
        problems.append(f"이전 기록의 라벨러와 다르다: {sorted(prev['labelers'])} → {sorted(snap['labelers'])}")
    return problems


def drift_warnings(snap: dict, prev: dict | None) -> list[str]:
    """빈 라벨 비율·조 비중의 이동. 판정이 아니라 경고다 — 늘어난 데이터는 원래 다를 수 있다."""
    if not prev:
        return []
    out = []
    d = (snap["empty_ratio"] - prev["empty_ratio"]) * 100
    if abs(d) > _DRIFT_PP:
        out.append(f"빈 라벨 비율 {prev['empty_ratio'] * 100:.1f}% → {snap['empty_ratio'] * 100:.1f}% ({d:+.1f}%p)")
    for a in sorted(set(snap["article_share"]) | set(prev["article_share"]), key=lambda x: int(x.strip("제조"))):
        before, after = prev["article_share"].get(a, 0.0), snap["article_share"].get(a, 0.0)
        if abs(after - before) * 100 > _DRIFT_PP:
            out.append(f"{a} 비중 {before * 100:.1f}% → {after * 100:.1f}% ({(after - before) * 100:+.1f}%p)")
    return out


def load_history(path: Path = HISTORY_PATH) -> list[dict]:
    return load_jsonl(path) if path.exists() else []


# ─────────────────────────────────────────────────────────────────────────────
# 5. 학습 — 현행 설정 재현
# ─────────────────────────────────────────────────────────────────────────────

def build_train_argv(base: dict, labels: Path, out_dir: Path, holdout_file: Path, gpu: int) -> list[str]:
    """현행 모델의 metrics.json에서 학습 인자를 복원한다. 바뀌는 것은 `--labels`뿐이어야 한다."""
    if base.get("train_subsample"):
        raise SystemExit("현행 모델이 부분표집으로 학습됐다 — 데이터 갱신 재학습의 기준으로 쓸 수 없다")
    if not base.get("negative_holdout_docs"):
        raise SystemExit("현행 모델에 음성 홀드아웃이 없다 — 오경보 축을 공정하게 잴 문서가 없다")
    tc = base["train_config"]
    argv = [
        "--labels", str(labels), "--model-dir", str(out_dir),
        "--epochs", str(tc["epochs"]), "--batch-size", str(tc["batch_size"]), "--lr", str(tc["lr"]),
        "--max-len", str(tc["max_len"]), "--min-support", str(tc["min_support"]),
        # 시드는 metrics에 남기기 전(2026-09-28) 모델이면 train_article 기본값 42였다
        "--seed", str(tc.get("seed", 42)),
        "--label-source", base.get("label_source", "agreed"),
        "--negative-holdout", str(base["negative_holdout_docs"]),
        "--negative-holdout-docs", str(holdout_file),
        "--gpu", str(gpu),
    ]
    if base.get("negative_ratio") is not None:
        argv += ["--negative-ratio", str(base["negative_ratio"])]
    if base.get("augment"):
        argv.append("--fulltext-augment")
    return argv


# ─────────────────────────────────────────────────────────────────────────────
# 6. 무해 확인
# ─────────────────────────────────────────────────────────────────────────────

def predict(model_dir: Path, texts: list[str], gpu: int, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """체크포인트를 **자기 학습 설정 그대로**(max_len·dev 임계값) 돌린다. 배포가 그렇게 도니까."""
    import torch
    from transformers import AutoTokenizer

    from backend.model.electra import ArticleMultiLabelElectra
    from backend.training.train_article import ArticleDataset

    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    names = list(metrics["article_names"])
    thr = np.array([metrics["thresholds"][a] for a in names], dtype=float)
    max_len = int(metrics["train_config"]["max_len"])
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = ArticleMultiLabelElectra.load(model_dir).to(device).eval()      # 지문 검사가 도는 경로
    ds = ArticleDataset([{"text": t, "articles": [], "group": "g"} for t in texts], tok, max_len, names)
    out = []
    with torch.no_grad():
        for b in torch.utils.data.DataLoader(ds, batch_size=batch_size):
            out.append(torch.sigmoid(model(b["input_ids"].to(device), b["attention_mask"].to(device),
                                           b["token_type_ids"].to(device))).cpu().numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(out), thr, names


def f1_rows(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """건별 F1. 예측·정답이 둘 다 비면 0 — 보고값 38.5%(`_sample_f1`)와 같은 정의."""
    tp = (pred & labels).sum(1)
    p = np.where(pred.sum(1) > 0, tp / np.maximum(pred.sum(1), 1), 0.0)
    r = np.where(labels.sum(1) > 0, tp / np.maximum(labels.sum(1), 1), 0.0)
    return np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)


def paired_ci(diff: np.ndarray, groups: list[str] | None = None,
              n_boot: int = 5000, seed: int = 42) -> tuple[float, float, float]:
    """페어드 차이의 평균과 95% CI(%p). `groups`를 주면 그 단위로 재표집한다(형제 조항)."""
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is None:
        boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    else:
        idx: dict[str, list[int]] = {}
        for i, g in enumerate(groups):
            idx.setdefault(g, []).append(i)
        clusters = [np.array(v) for v in idx.values()]
        boot = []
        for _ in range(n_boot):
            pick = np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
            boot.append(diff[pick].mean())
    return float(diff.mean() * 100), float(np.percentile(boot, 2.5) * 100), float(np.percentile(boot, 97.5) * 100)


def harm_verdict(recall_ci: tuple[float, float, float], fpr_ci: tuple[float, float, float]) -> dict:
    """사전 등록 규칙. 둘 중 하나라도 유의하게 나빠졌으면 기각한다."""
    reasons = []
    if recall_ci[2] < 0:
        reasons.append(f"gold 조항 재현이 유의하게 떨어졌다 {recall_ci[0]:+.1f}%p [{recall_ci[1]:+.1f},{recall_ci[2]:+.1f}]")
    if fpr_ci[1] > 0:
        reasons.append(f"홀드아웃 오경보가 유의하게 올랐다 {fpr_ci[0]:+.1f}%p [{fpr_ci[1]:+.1f},{fpr_ci[2]:+.1f}]")
    return {"passed": not reasons, "reasons": reasons}


def eval_sets(holdout_docs: list[str], labels: Path = LABELED_PATH) -> tuple[list[dict], list[dict]]:
    """고정 평가셋 두 개. gold는 FTC 외부 정답, 음성 풀은 **출처로만** 정의한다(라벨로 거르면 순환)."""
    gold = load_ftc_gold("clean")
    held = set(holdout_docs)
    rows = {r["chunk_id"]: r for r in load_jsonl(labels) + load_jsonl(NOISE_PATH)}
    neg = [r for r in rows.values()
           if r.get("source") == "standard_contract"
           and ":".join(str(r["chunk_id"]).split(":")[:2]) in held]
    return gold, neg


def evaluate_pair(base_dir: Path, cand_dir: Path, holdout_docs: list[str], gpu: int,
                  labels: Path = LABELED_PATH) -> dict:
    gold, neg = eval_sets(holdout_docs, labels)
    neg_groups = [":".join(str(r["chunk_id"]).split(":")[:2]) for r in neg]
    out: dict[str, Any] = {"n_gold": len(gold), "n_holdout": len(neg), "n_holdout_docs": len(set(neg_groups))}
    rows = {}
    for tag, d in (("base", base_dir), ("cand", cand_dir)):
        P, thr, names = predict(d, [g["text"] for g in gold], gpu)
        Pn, *_ = predict(d, [r["text"] for r in neg], gpu)
        pred = P >= thr
        labels = np.array([[a in g["articles"] for a in names] for g in gold])
        rows[tag] = {"recall": pred.any(1).astype(float), "fpr": (Pn >= thr).any(1).astype(float),
                     "f1": f1_rows(pred, labels)}
        out[tag] = {"dir": str(d), "article_names": names,
                    "gold_recall": float(rows[tag]["recall"].mean()),
                    "holdout_fpr": float(rows[tag]["fpr"].mean()),
                    "gold_f1": float(rows[tag]["f1"].mean())}
        if tag == "cand":
            # 상수 기준선은 후보의 조 목록으로 — 상수도 같은 라벨 공간에서 재야 비교가 된다
            from backend.training.train_article import constant_baseline_f1
            best = max((constant_baseline_f1(gold, names, k) for k in (1, 2, 3, 4)), key=lambda b: b["f1"])
            const = np.tile(np.array([a in best["articles"] for a in names]), (len(gold), 1))
            out["cand_vs_constant"] = {"constant": best,
                                       "diff_ci_pp": paired_ci(rows[tag]["f1"] - f1_rows(const, labels))}

    b, c = rows["base"], rows["cand"]
    out["recall_diff_ci_pp"] = paired_ci(c["recall"] - b["recall"])
    out["fpr_diff_ci_pp"] = paired_ci(c["fpr"] - b["fpr"], neg_groups)
    out["f1_diff_ci_pp"] = paired_ci(c["f1"] - b["f1"])
    return out


def cumulative_warnings(history: list[dict], cand: dict) -> list[str]:
    """첫 기록의 현행 모델 대비 누적 이동. 세대마다 '유의하지 않음'이 쌓이는 것을 잡는다."""
    if not history:
        return []
    first = history[0]["eval"]["base"]
    out = []
    d_r = (cand["gold_recall"] - first["gold_recall"]) * 100
    d_f = (cand["holdout_fpr"] - first["holdout_fpr"]) * 100
    if d_r < -_CUMULATIVE_PP:
        out.append(f"첫 기록({first['dir']}) 대비 gold 재현 누적 {d_r:+.1f}%p")
    if d_f > _CUMULATIVE_PP:
        out.append(f"첫 기록({first['dir']}) 대비 홀드아웃 오경보 누적 {d_f:+.1f}%p")
    return out


# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="새 라벨 데이터로 재학습 → 무해 확인 → 기록 → (선택) 승격")
    ap.add_argument("--base", default=str(BASE_DIR), help="현행 배포 체크포인트 (기본: MODEL_DIR)")
    ap.add_argument("--labels", default=str(LABELED_PATH))
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--check-only", action="store_true", help="1~4(데이터 확인)만 하고 멈춘다. GPU 불필요")
    ap.add_argument("--force", action="store_true", help="라벨 지문이 같아도 학습한다")
    ap.add_argument("--allow-labeler-change", action="store_true",
                    help="라벨러 조합 변경을 알고 진행한다. 이력에 그대로 남는다")
    ap.add_argument("--promote-as", default="", metavar="DIR",
                    help="무해 확인을 통과하면 이 이름으로 승격한다(예: models/article_v3). 서빙 전환은 별도")
    args = ap.parse_args(argv)

    base_dir, labels = Path(args.base), Path(args.labels)
    base = json.loads((base_dir / "metrics.json").read_text(encoding="utf-8"))
    history = load_history()
    prev = history[-1]["snapshot"] if history else None
    holdout_docs = json.loads(HOLDOUT_DOCS_PATH.read_text(encoding="utf-8"))

    logger.info(f"========== 재학습 | 현행 {base_dir} | 라벨 {labels} ==========")
    snap = snapshot(labels, base.get("label_source", "agreed"))
    chunk_ids = snap.pop("chunk_ids")
    logger.info(f"  ① 스냅숏  행 {snap['rows']} · 학습 후보 {snap['records']} · 빈 라벨 {snap['empty_ratio'] * 100:.1f}% "
                f"· 출처 {snap['source_counts']}")

    base_sha = base.get("labels_sha256") or (prev or {}).get("sha256")
    if base_sha is None:
        logger.info("  ② 현행 모델의 학습 데이터 지문이 없다(기록 도입 이전 모델) — 변경 여부를 알 수 없어 진행한다")
    elif base_sha == snap["sha256"] and not args.force:
        logger.info("  ② 라벨 파일이 현행 모델 학습 때와 같다 — 새 데이터가 없어 멈춘다 (--force로 무시)")
        return
    new_records = None
    if history and Path(history[-1].get("chunk_ids_file", "")).exists():
        before = set(json.loads(Path(history[-1]["chunk_ids_file"]).read_text(encoding="utf-8")))
        new_records = len(set(chunk_ids) - before)
        logger.info(f"  ② 이전 기록 대비 새 레코드 {new_records}건")

    problems = check_labelers(snap, prev)
    for p in problems:
        logger.warning(f"  ③ ★ {p}")
    if problems and not args.allow_labeler_change:
        raise SystemExit("라벨러 확인 실패 — 원인을 확인하고, 의도한 변경이면 --allow-labeler-change")
    if not problems:
        logger.info(f"  ③ 라벨러 단일 조합 확인: {next(iter(snap['labelers']))}")

    drift = drift_warnings(snap, prev)
    for w in drift:
        logger.warning(f"  ④ 분포 이동(판정 아님, 볼 것): {w}")
    if not prev:
        logger.info("  ④ 분포 이동: 비교할 이전 기록이 없다(첫 기록)")

    recorded = base.get("negative_holdout_doc_ids")
    if recorded and sorted(recorded) != sorted(holdout_docs):
        raise SystemExit(f"현행 모델의 홀드아웃 문서가 고정 목록과 다르다 — {HOLDOUT_DOCS_PATH.name}를 확인할 것")
    present = {":".join(c.split(":")[:2]) for c in chunk_ids}
    missing = sorted(set(holdout_docs) - present)
    if missing:
        raise SystemExit(f"고정 홀드아웃 문서 {len(missing)}개가 라벨 파일에 없다: {missing[:5]}")
    logger.info(f"  홀드아웃 고정 문서 {len(holdout_docs)}개 확인 ({HOLDOUT_DOCS_PATH.name})")

    if args.check_only:
        logger.info("  --check-only: 데이터 확인까지만 했다")
        return

    run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M")
    cand_dir = PROJECT_ROOT / f"models/_retrain_{run_id}"
    train_argv = build_train_argv(base, labels, cand_dir, HOLDOUT_DOCS_PATH, args.gpu)
    logger.info(f"  ⑤ 학습 → {cand_dir}")
    logger.info(f"     인자(현행 설정 재현): {' '.join(train_argv)}")
    from backend.training.train_article import main as train_main
    train_main(train_argv)

    logger.info("  ⑥ 무해 확인 (사전 등록: 재현 CI 상한<0 또는 오경보 CI 하한>0 이면 기각)")
    ev = evaluate_pair(base_dir, cand_dir, holdout_docs, args.gpu, labels)
    verdict = harm_verdict(ev["recall_diff_ci_pp"], ev["fpr_diff_ci_pp"])
    for tag, name in (("base", "현행"), ("cand", "후보")):
        r = ev[tag]
        logger.info(f"     {name}  gold 재현 {r['gold_recall'] * 100:5.1f}% · 홀드아웃 오경보 "
                    f"{r['holdout_fpr'] * 100:4.1f}% · 조 F1 {r['gold_f1'] * 100:5.1f}%")
    for key, label in (("recall_diff_ci_pp", "재현 차"), ("fpr_diff_ci_pp", "오경보 차"), ("f1_diff_ci_pp", "조 F1 차(기록)")):
        d, lo, hi = ev[key]
        logger.info(f"     {label:<12} {d:+.1f}%p [{lo:+.1f},{hi:+.1f}]")
    d, lo, hi = ev["cand_vs_constant"]["diff_ci_pp"]
    logger.info(f"     후보 조 F1 상수 대비(기록) {d:+.1f}%p [{lo:+.1f},{hi:+.1f}] "
                f"— 상수 {ev['cand_vs_constant']['constant']['articles']}")
    if ev["base"]["article_names"] != ev["cand"]["article_names"]:
        logger.warning(f"     조 목록이 바뀌었다 {ev['base']['article_names']} → {ev['cand']['article_names']} "
                       f"— 조 F1 차는 라벨 공간이 달라 참고만")
    logger.info(f"     판정: {'통과 — 두 축 어디서도 유의하게 나빠지지 않았다(개선의 증거는 아니다)' if verdict['passed'] else '기각'}")
    for r in verdict["reasons"]:
        logger.warning(f"     ★ {r}")

    cumulative = cumulative_warnings(history, ev["cand"])
    for w in cumulative:
        logger.warning(f"  ⑦ ★ 누적 하락 경고: {w}")

    promoted = None
    if args.promote_as:
        if not verdict["passed"]:
            logger.warning(f"  ⑧ 무해 확인을 통과하지 못해 {args.promote_as}로 승격하지 않는다")
        else:
            from backend.scripts.promote_checkpoint import promote
            d_r, lo_r, hi_r = ev["recall_diff_ci_pp"]
            d_f, lo_f, hi_f = ev["fpr_diff_ci_pp"]
            promote(cand_dir, Path(args.promote_as),
                    f"데이터 갱신 재학습({snap['records']}건, sha {snap['sha256'][:12]}). 사전 등록 무해 확인 통과 — "
                    f"gold 재현 {d_r:+.1f}%p [{lo_r:+.1f},{hi_r:+.1f}] · 홀드아웃 오경보 {d_f:+.1f}%p "
                    f"[{lo_f:+.1f},{hi_f:+.1f}]. 개선의 증거가 아니라 '나빠지지 않음'이다")
            promoted = args.promote_as
            logger.info(f"  ⑧ 승격 완료. 서빙에 반영하려면 MODEL_DIR={promoted} (judgment_agent 기본값은 그대로다)")

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ids_file = HISTORY_PATH.parent / f"retrain_chunk_ids_{run_id}.json"
    ids_file.write_text(json.dumps(chunk_ids, ensure_ascii=False), encoding="utf-8")
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "run_id": run_id, "base": str(base_dir), "candidate": str(cand_dir),
            "snapshot": snap, "chunk_ids_file": str(ids_file), "new_records": new_records,
            "labeler_problems": problems, "drift_warnings": drift, "cumulative_warnings": cumulative,
            "eval": ev, "verdict": verdict, "promoted_as": promoted,
        }, ensure_ascii=False) + "\n")
    logger.info(f"  ⑦ 이력 기록: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
