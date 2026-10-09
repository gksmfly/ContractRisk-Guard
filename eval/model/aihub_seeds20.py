# eval/model/aihub_seeds20.py
"""시드 20개 — D(FB-Check) vs B(같은 수 무작위) 추가 10시드 학습 (사전 등록: eval/paper_plan.md §7-15, 2026-10-09)

디스크 여유가 3GB 남짓이라 한 모델씩: 학습 → AI Hub 예측 캐시 저장 → 가중치(model.safetensors) 삭제.
metrics.json·임계값은 남긴다. 판정은 캐시로 한다.

    .venv/bin/python -m eval.model.aihub_seeds20 --gpu 0        # 학습·캐시 (이어서 가능)
    .venv/bin/python -m eval.model.aihub_eval --gpu 0 --mode seeds20
"""

import argparse

from backend.utils import load_logger
from eval.model.aihub_eval import CACHE, MODES, load_probs, testset
from eval.model.e8_downstream_ablation import _model_dir, prepare_arms, train_arm

logger = load_logger("aihub_seeds20.log")
NEW_SEEDS = [s for s in MODES["seeds20"]["seeds"] if s not in MODES["seeds10"]["seeds"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    paths = prepare_arms()
    texts = [it["text"] for it in testset()]
    for s in NEW_SEEDS:
        for arm in ("D", "B"):
            if (CACHE / f"{arm}_s{s}.npz").exists():
                logger.info(f"  {arm} s{s}: 캐시 있음 — 건너뜀")
                continue
            d = train_arm(arm, s, paths[arm], args.gpu)
            load_probs(f"{arm}_s{s}", d, texts, args.gpu)
            w = d / "model.safetensors"
            if w.exists():
                w.unlink()
            logger.info(f"  {arm} s{s}: 학습·캐시 완료, 가중치 삭제")


if __name__ == "__main__":
    main()
