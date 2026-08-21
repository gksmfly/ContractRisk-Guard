# backend/eval/confidence_calibration.py
"""
A-3 진단 — KoELECTRA가 계산해놓고 버리는 softmax 확률이 실제 정답률과 상관이 있는가.

현재 상태의 문제:
  `judgment_agent.electra_predict()`는 `F.softmax(...)`로 확률을 구한 뒤 `argmax`만 쓰고
  확률은 버린다. **softmax는 순서를 바꾸지 않으므로, 라벨만 쓸 거면 이 두 줄은 지워도
  결과가 완전히 동일하다** — 계산만 하고 안 쓰는 죽은 연산이다.
  그런데 `analyze.py`는 `confidence = 1.0 if verified else 0.7`이라는 **별도의 하드코딩**을
  사용자에게 "신뢰도"로 내보낸다(`verified`는 GPT와 KoELECTRA의 domain 판단 일치 여부이지
  신뢰도가 아니다). 즉 진짜 신호는 버리고 가짜 값을 노출하는 상태다.

그래서 "버린 확률을 살리면 되지 않나"가 자연스러운 결론이지만, v4의 실제 정확도는
43.4%(5시드 평균)이고 Medium F1은 0.10~0.14다. 딥러닝 분류기의 softmax는 실제 정확도보다
과하게 높게 나오는 것이 알려진 현상이라, 그대로 %로 노출하면 "97% 확신"인데 자주 틀리는
상황이 된다 — 지금보다 **더 나쁠 수 있다.**

따라서 추측하지 않고 잰다. 이 프로젝트는 이미 같은 방식으로 코사인 유사도 신호를
검증했다가 기각한 전례가 있다(`evidence_verification_agent.py`: hit 0.567 vs miss 0.559
→ 미채택). 동일한 기준을 적용한다:

  확률과 정답률이 **상관 있으면**  → 살려서 confidence로 사용(노출 형태는 보정도를 보고 결정)
  **상관 없으면**                  → softmax 두 줄을 삭제하고 confidence 필드도 제거

판정 지표:
  - 구간별 실제 정확도가 단조 증가하는가 (확률이 높을수록 정말 더 맞는가)
  - ECE(Expected Calibration Error) — 확률과 실제 정확도의 평균 괴리. 0에 가까울수록 잘 보정됨
  - 최고확률 구간과 최저확률 구간의 정확도 차이 (신호의 실질적 크기)

평가 조건은 `compare_judgment.py`·`ensemble_compare.py`와 동일하다(860건, evidence_span
입력, 기존 캐시 재사용이라 OpenAI 비용 0).

실행: .venv/bin/python -m backend.eval.confidence_calibration
"""

import argparse
from pathlib import Path

import numpy as np

from backend.eval.ensemble_compare import GT_PATH, SPAN_CACHE_PATH, predict_probs
from backend.model.electra import RISK_MAP
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("confidence_calibration.log")
OUT_PATH = PROJECT_ROOT / "data/eval/confidence_calibration_report.json"

_BINS = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def calibration_table(probs: np.ndarray, y_true: list[int]) -> tuple[list[dict], float]:
    """구간별 (건수, 평균 예측확률, 실제 정확도)와 ECE를 낸다."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.array(y_true)).astype(float)

    table, ece = [], 0.0
    for lo, hi in _BINS:
        m = (conf >= lo) & (conf < hi)
        n = int(m.sum())
        if n == 0:
            table.append({"bin": f"{lo:.1f}~{hi:.1f}", "n": 0, "mean_conf": None, "accuracy": None})
            continue
        mc, acc = float(conf[m].mean()), float(correct[m].mean())
        table.append({"bin": f"{lo:.1f}~{hi:.1f}", "n": n, "mean_conf": mc, "accuracy": acc})
        ece += n / len(conf) * abs(mc - acc)
    return table, ece


def main(model_dir: str | None = None) -> None:
    gt = load_jsonl(GT_PATH)
    spans = {r["chunk_id"]: r["evidence_span"] for r in load_jsonl(SPAN_CACHE_PATH)}
    rows = [r for r in gt if spans.get(r["chunk_id"])]
    texts = [spans[r["chunk_id"]] for r in rows]
    y_true = [RISK_MAP[r["risk_level"]] for r in rows]

    md = Path(model_dir) if model_dir else PROJECT_ROOT / "models/v4"
    logger.info(f"  평가 {len(rows)}건 / 체크포인트 {md.name} (OpenAI 호출 0)")

    probs = predict_probs(md, texts)
    table, ece = calibration_table(probs, y_true)

    filled = [t for t in table if t["n"] > 0]
    accs = [t["accuracy"] for t in filled]
    monotonic = all(a <= b + 1e-9 for a, b in zip(accs, accs[1:]))
    spread = (max(accs) - min(accs)) if accs else 0.0
    overall = float((probs.argmax(axis=1) == np.array(y_true)).mean())

    verdict = "상관 있음(살릴 가치)" if (monotonic and spread >= 0.10) else \
              "약함/없음(삭제 검토)" if spread < 0.10 else "부분적(단조 아님)"

    save_json({"model": md.name, "n_eval": len(rows), "overall_accuracy": overall,
               "ece": ece, "monotonic": monotonic, "accuracy_spread": spread,
               "verdict": verdict, "table": table}, OUT_PATH)

    logger.info(f"===== 확률 보정 진단 ({md.name}, n={len(rows)}) =====")
    logger.info(f"  전체 정확도: {overall * 100:.1f}%")
    logger.info(f"  {'확률 구간':<12}{'건수':>7}{'평균확률':>10}{'실제정확도':>12}{'괴리':>9}")
    for t in table:
        if t["n"] == 0:
            logger.info(f"  {t['bin']:<12}{0:>7}{'-':>10}{'-':>12}{'-':>9}")
        else:
            logger.info(f"  {t['bin']:<12}{t['n']:>7}{t['mean_conf'] * 100:>9.1f}%{t['accuracy'] * 100:>11.1f}%"
                        f"{(t['mean_conf'] - t['accuracy']) * 100:>+8.1f}%p")
    logger.info(f"  ECE(기대 보정 오차): {ece:.4f}  (0에 가까울수록 정확한 확률)")
    logger.info(f"  단조 증가: {monotonic} / 구간 간 정확도 차: {spread * 100:.1f}%p")
    logger.info(f"  → 판정: {verdict}")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=None)
    main(model_dir=p.parse_args().model_dir)
