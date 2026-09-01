# backend/eval/fixedsplit_verify.py
"""
누수를 고친 분할로 재학습한 모델이 **지름길 없이도** 작동하는지 검증한다.

## 왜 필요한가

당시 프로덕션이던 `models/v4`는 두 가지로 오염돼 있다:
  ① 레코드 단위 분할 누수(증강 쌍·형제 조항·문서 간 텍스트 중복 55.4%)
  ② 학습 라벨의 16.4%를 정확도 45%짜리 이전 세대 KoELECTRA가 결정

`train.py::split_by_document()`로 ①을 고치고 재학습하면 내부 검증 정확도가 86.2%였는데,
이 값이 **출처 지름길**(`ftc_case→High, standard_contract→Low`)의 산물인지 실제 내용
이해인지 구분해야 한다. 그래서 검증셋을 둘로 갈라서 잰다:

  - 지름길이 **맞는** 구간: 출처 규칙만으로도 정답이 나오는 샘플
  - 지름길이 **틀리는** 구간: 출처 규칙이 오답을 내는 샘플 ← **여기 정확도가 진짜 실력**

지름길이 틀리는 구간에서 무작위(33%)를 크게 넘으면, 모델이 조항 내용을 읽고 있다는 뜻이다.

단일 시드 측정은 이 프로젝트에서 신뢰할 수 없다(시드만 바꿔도 22%p 흔들린 전례).
그래서 여러 시드의 평균·범위를 함께 낸다.

실행: .venv/bin/python -m backend.eval.fixedsplit_verify
"""

import argparse
import collections
import json
import logging

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from backend.model.electra import RISK_MAP, DualHeadElectra
from backend.training.train import load_records, split_by_document
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("fixedsplit_verify.log")
OUT_PATH = PROJECT_ROOT / "data/eval/fixedsplit_verify_report.json"
_SEEDS = (42, 7, 1, 100, 123)


def _source_map() -> dict[str, str]:
    """조항 텍스트 → 출처. clean/noise 원본에서 만든다(load_records는 source를 안 싣는다)."""
    m: dict[str, str] = {}
    for name in ("clean", "noise"):
        with open(PROJECT_ROOT / f"data/fb_check/{name}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                for key in ("evidence_span", "text"):
                    t = (r.get(key) or "").strip()
                    if t:
                        m[t] = r.get("source")
    return m


def predict(model_dir: str, texts: list[str], device: torch.device) -> np.ndarray:
    model = DualHeadElectra(str(model_dir))
    heads = torch.load(model_dir / "heads.pt", map_location=device, weights_only=True)
    model.domain_head.load_state_dict(heads["domain_head"])
    model.risk_head.load_state_dict(heads["risk_head"])
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    out = []
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], max_length=256, padding="max_length",
                  truncation=True, return_tensors="pt")
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        tt = enc.get("token_type_ids", torch.zeros_like(ids)).to(device)
        with torch.no_grad():
            _, r = model(ids, mask, tt)
        out.append(F.softmax(r, dim=-1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out).argmax(axis=1)


def main(model_root: str = "models/_fixedsplit", data_source: str = "clean") -> None:
    logging.disable(logging.INFO)          # split_by_document의 진단 로그는 여기선 소음
    src = _source_map()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    root = PROJECT_ROOT / model_root

    rows_out, agg = [], collections.defaultdict(list)
    for seed in _SEEDS:
        md = root / f"seed{seed}"
        if not (md / "heads.pt").exists():
            continue
        records = load_records(data_source, fulltext_augment=True)
        train, val = split_by_document(records, 0.2, seed)     # 학습과 동일한 분할 재현
        for group in (train, val):
            for r in group:
                r["source"] = src.get(r["text"].strip())

        # 지름길 규칙은 **학습셋에서만** 도출한다(검증셋을 보고 만들면 반칙)
        by = collections.defaultdict(collections.Counter)
        for r in train:
            by[r["source"]][r["risk_level"]] += 1
        rule = {s: c.most_common(1)[0][0] for s, c in by.items()}

        y = np.array([RISK_MAP[r["risk_level"]] for r in val])
        shortcut = np.array([RISK_MAP.get(rule.get(r["source"], "Low"), RISK_MAP["Low"]) for r in val])
        pred = predict(md, [r["text"] for r in val], device)

        hits = shortcut == y
        rec = {
            "seed": seed, "n_val": len(val),
            "majority": max(collections.Counter(r["risk_level"] for r in val).values()) / len(val),
            "shortcut": float(hits.mean()),
            "model": float((pred == y).mean()),
            "model_where_shortcut_wrong": float((pred[~hits] == y[~hits]).mean()) if (~hits).any() else float("nan"),
            "n_shortcut_wrong": int((~hits).sum()),
            "model_where_shortcut_right": float((pred[hits] == y[hits]).mean()) if hits.any() else float("nan"),
        }
        rows_out.append(rec)
        for k in ("majority", "shortcut", "model", "model_where_shortcut_wrong"):
            agg[k].append(rec[k])
        logging.disable(logging.NOTSET)
        logger.info(f"  seed{seed:<4} val={rec['n_val']:>3}  기준선 {rec['majority']*100:>5.1f}%  "
                    f"지름길 {rec['shortcut']*100:>5.1f}%  모델 {rec['model']*100:>5.1f}%  "
                    f"지름길틀린{rec['n_shortcut_wrong']:>3}건에서 {rec['model_where_shortcut_wrong']*100:>5.1f}%")
        logging.disable(logging.INFO)

    logging.disable(logging.NOTSET)
    if not rows_out:
        logger.warning(f"  {root} 에 체크포인트가 없다")
        return

    summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                   "min": float(min(v)), "max": float(max(v))} for k, v in agg.items()}
    save_json({"model_root": model_root, "seeds": rows_out, "summary": summary}, OUT_PATH)

    logger.info(f"===== {len(rows_out)}시드 요약 =====")
    for k, label in (("majority", "다수 클래스 기준선"), ("shortcut", "출처 지름길"),
                     ("model", "모델(전체)"), ("model_where_shortcut_wrong", "모델(지름길 틀린 구간)")):
        s = summary[k]
        logger.info(f"  {label:<24} {s['mean']*100:>5.1f}% ±{s['std']*100:.1f}  "
                    f"(범위 {s['min']*100:.1f}~{s['max']*100:.1f})")
    gain = summary["model_where_shortcut_wrong"]["mean"] - 1 / len(RISK_MAP)
    logger.info(f"  → 지름길 없는 구간에서 무작위(33.3%) 대비 {gain*100:+.1f}%p")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-root", default="models/_fixedsplit")
    p.add_argument("--data-source", default="clean", choices=["clean", "gt3", "seed"],
                   help="검증에 쓸 데이터(학습 때와 같아야 분할이 재현된다)")
    a = p.parse_args()
    main(model_root=a.model_root, data_source=a.data_source)
