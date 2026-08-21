# backend/eval/ensemble_compare.py
"""
KoELECTRA 시드 앙상블 — 추가 학습 없이 성능·안정성을 올릴 수 있는지 측정한다.

배경: `models/README.md`의 시드 변동성 실험에서 v4 세대는 **같은 데이터·같은 설정으로
시드만 바꿔 재학습해도 36.0~58.3%(std 8.9%p)로 흔들렸다.** 문서는 이를 "지금까지의 모든
버전 비교는 통계적으로 신뢰하기 어렵다"는 경고로 읽었는데, 여기서는 **개선 신호**로 읽는다 —
개별 모델이 서로 다른 실수를 한다는 뜻이고, 서로 다른 실수는 평균으로 상쇄된다.

v5~v9는 전부 "학습 데이터를 늘린다"는 **한 축만 여섯 번 반복**했고 전부 v4를 못 넘었다.
앙상블은 아직 시도된 적 없는 축이며, `models/_seedexp/`에 5세대×5시드 = **30개 체크포인트가
이미 존재**해서 추가 학습 없이 추론만으로 측정된다(GPU 시간만, 실패해도 잃는 것 없음).

평가 조건은 `compare_judgment.py`와 동일하게 맞춘다:
  - 평가셋: `data/eval/ground_truth_3class.jsonl` (860건, High 396/Medium 88/Low 376)
  - 입력: `evidence_span` (프로덕션 Judgment Agent가 실제로 받는 형태. v4는 원문을 그대로
    넣으면 정확도가 크게 떨어지는 게 알려진 특성) — 기존 캐시 재사용이라 **OpenAI 비용 0**
  - 따라서 기존에 기록된 수치와 직접 비교 가능

앙상블 방식은 soft voting(확률 평균)이다 — hard voting(다수결)은 동률이 잦고 확률 정보를
버린다. 확률 평균이 A-3(confidence) 조사와도 이어진다.

실행:
    .venv/bin/python -m backend.eval.ensemble_compare
    .venv/bin/python -m backend.eval.ensemble_compare --limit 50   # 스모크
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

from backend.model.electra import DualHeadElectra, INV_RISK_MAP, RISK_MAP
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("ensemble_compare.log")

GT_PATH = PROJECT_ROOT / "data/eval/ground_truth_3class.jsonl"
SPAN_CACHE_PATH = PROJECT_ROOT / "data/eval/evidence_span_cache.jsonl"
SEEDEXP_DIR = PROJECT_ROOT / "models/_seedexp"
OUT_PATH = PROJECT_ROOT / "data/eval/ensemble_report.json"

_LABELS = ["High", "Medium", "Low"]
_SEEDS = (1, 7, 42, 100, 123)
_GENERATIONS = ("v4", "v5", "v6", "v7", "v8")
_BATCH = 64
_MAX_LEN = 256


def _device() -> torch.device:
    # 프로젝트 규칙상 GPU는 cuda:1 고정(`Claude.md`).
    return torch.device(os.environ.get("EVAL_DEVICE", "cuda:1") if torch.cuda.is_available() else "cpu")


def predict_probs(model_dir: Path, texts: list[str], num_risk_labels: int | None = None) -> np.ndarray:
    """체크포인트 하나로 risk_level 확률을 배치 추론한다. 반환 shape: (N, num_risk_labels).

    `judgment_agent.electra_predict`와 동일한 토크나이즈 설정(max_length=256,
    padding="max_length")을 쓴다 — 다르면 기존 기록 수치와 비교가 성립하지 않는다.

    num_risk_labels를 안 주면 3class(기본)로 헤드를 만든다 — 2class 체크포인트를 읽을
    때는 2를 넘겨야 heads.pt 형상과 맞는다(`risk_scheme_compare.py`가 그렇게 쓴다).
    """
    device = _device()
    model = DualHeadElectra(str(model_dir), num_risk_labels=num_risk_labels)
    heads = torch.load(model_dir / "heads.pt", map_location=device, weights_only=True)
    model.domain_head.load_state_dict(heads["domain_head"])
    model.risk_head.load_state_dict(heads["risk_head"])
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    out = []
    for i in range(0, len(texts), _BATCH):
        enc = tokenizer(
            texts[i:i + _BATCH], max_length=_MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        tok_type = enc.get("token_type_ids", torch.zeros_like(ids)).to(device)
        with torch.no_grad():
            _, r_logits = model(ids, mask, tok_type)
        out.append(F.softmax(r_logits, dim=-1).cpu().numpy())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


def _score(probs: np.ndarray, y_true: list[int]) -> dict:
    y_pred = probs.argmax(axis=1)
    per_class = f1_score(y_true, y_pred, average=None, labels=list(range(len(_LABELS))), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_class": {lab: float(per_class[RISK_MAP[lab]]) for lab in _LABELS},
    }


def main(limit: int | None = None) -> None:
    gt = load_jsonl(GT_PATH)
    spans = {r["chunk_id"]: r["evidence_span"] for r in load_jsonl(SPAN_CACHE_PATH)}
    rows = [r for r in gt if spans.get(r["chunk_id"])]
    if limit:
        rows = rows[:limit]
    texts = [spans[r["chunk_id"]] for r in rows]
    y_true = [RISK_MAP[r["risk_level"]] for r in rows]
    logger.info(f"  평가 {len(rows)}건 (span 캐시 사용, OpenAI 호출 0)")

    # 개별 체크포인트 추론 — 세대×시드 전부
    probs: dict[str, np.ndarray] = {}
    for gen in _GENERATIONS:
        for seed in _SEEDS:
            d = SEEDEXP_DIR / f"{gen}_seed{seed}"
            if not d.exists():
                continue
            key = f"{gen}_seed{seed}"
            probs[key] = predict_probs(d, texts)
            logger.info(f"  {key}: acc={_score(probs[key], y_true)['accuracy'] * 100:.1f}%")

    prod_dir = PROJECT_ROOT / "models/v4"
    if prod_dir.exists():
        probs["v4_production"] = predict_probs(prod_dir, texts)
        logger.info(f"  v4_production: acc={_score(probs['v4_production'], y_true)['accuracy'] * 100:.1f}%")

    results = {k: _score(v, y_true) for k, v in probs.items()}

    # 앙상블 조합
    def ens(keys: list[str]) -> np.ndarray | None:
        avail = [probs[k] for k in keys if k in probs]
        return np.mean(avail, axis=0) if avail else None

    combos = {"ensemble_v4_5seeds": [f"v4_seed{s}" for s in _SEEDS],
              "ensemble_all_gens": [f"{g}_seed{s}" for g in _GENERATIONS for s in _SEEDS]}
    for gen in _GENERATIONS:
        combos[f"ensemble_{gen}_5seeds"] = [f"{gen}_seed{s}" for s in _SEEDS]

    for name, keys in combos.items():
        p = ens(keys)
        if p is not None:
            results[name] = _score(p, y_true)
            results[name]["n_models"] = sum(1 for k in keys if k in probs)

    v4_singles = [results[f"v4_seed{s}"]["accuracy"] for s in _SEEDS if f"v4_seed{s}" in results]
    summary = {
        "n_eval": len(rows),
        "v4_single_mean": float(np.mean(v4_singles)) if v4_singles else None,
        "v4_single_std": float(np.std(v4_singles)) if v4_singles else None,
        "v4_single_range": [float(min(v4_singles)), float(max(v4_singles))] if v4_singles else None,
    }
    # 샘플별 예측을 남긴다 — 집계만 저장하면 두 설정 간 McNemar 검정을 나중에 할 수 없다
    # (v6앙상블 48.8% vs v4 45.7% 같은 차이가 유의한지 판단하려면 쌍별 비교가 필요)
    predictions = {k: v.argmax(axis=1).astype(int).tolist() for k, v in probs.items()}
    for name, keys in combos.items():
        p = ens(keys)
        if p is not None:
            predictions[name] = p.argmax(axis=1).astype(int).tolist()

    save_json({"summary": summary, "results": results,
               "y_true": y_true, "predictions": predictions,
               "chunk_ids": [r["chunk_id"] for r in rows]}, OUT_PATH)

    logger.info("===== 결과 (정확도 / macro-F1 / Medium-F1) =====")
    for name in sorted(results, key=lambda k: -results[k]["accuracy"]):
        r = results[name]
        logger.info(f"  {name:<24} {r['accuracy'] * 100:>6.1f}%  {r['macro_f1']:>6.3f}  {r['f1_per_class']['Medium']:>6.3f}")
    if v4_singles:
        logger.info(f"  [v4 단일 5시드] 평균 {np.mean(v4_singles) * 100:.1f}% / 표준편차 {np.std(v4_singles) * 100:.1f}%p "
                    f"/ 범위 {min(v4_singles) * 100:.1f}~{max(v4_singles) * 100:.1f}%")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    main(limit=p.parse_args().limit)
