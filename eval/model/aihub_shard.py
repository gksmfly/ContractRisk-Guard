# eval/model/aihub_shard.py
"""§7-26(LoRA 어댑터)·§7-27(라벨 유형 oracle) 학습을 한 GPU에서 여러 프로세스로 나눠 돌린다 (사전 등록: eval/paper_plan.md §7-26·§7-27)

`aihub_seeds20.py`의 학습 루프와 같지만 (시드, 팔) 쌍을 `--shard i --of N`으로 나눈다. 쌍마다 캐시(`aihub_cache/{arm}_s{seed}.npz`)가
있으면 건너뛰므로 샤드끼리 겹치지 않고 중간에 멈춰도 이어서 돌 수 있다. 가중치는 학습 직후 지운다(디스크 여유 부족).

    HF_HUB_OFFLINE=1 .venv/bin/python -m eval.model.aihub_shard --mode adapter   --gpu 1 --shard 0 --of 1
    HF_HUB_OFFLINE=1 .venv/bin/python -m eval.model.aihub_shard --mode labeltype --gpu 1 --shard 0 --of 2
    HF_HUB_OFFLINE=1 .venv/bin/python -m eval.model.aihub_shard --repro --gpu 1        # §7-26 재현 확인

**재현 확인(--repro, 판정 기준을 실행 전에 고정):** 코드 변경(optimizer가 requires_grad 파라미터만 받도록) 뒤에도 전체 파인튜닝이
옛 결과를 내는지 보려고 D 팔 시드 42를 다시 학습해 저장된 예측(D_s42)과 비교한다. **일치 = 임계값을 적용한 예측의 일치율 ≥ 98% 이고
평균 |ΔP| < 0.02.** GPU 비결정성 때문에 비트 단위 일치는 기대하지 않는다. 기준을 못 넘으면 LoRA 비교(LA−A, LD−D)의 기준선이 달라졌을 수
있으므로 결과 해석을 보류한다.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import eval.model.e8_downstream_ablation as e8
from backend.utils import PROJECT_ROOT, load_logger, save_json
from eval.model.aihub_eval import CACHE, MODES, load_probs, testset
from eval.model.e8_downstream_ablation import prepare_arms, train_arm

logger = load_logger("aihub_shard.log")
ARMS = {"adapter": ("LA", "LD"), "labeltype": ("OQ", "OFP", "OFN", "OMX", "OG")}


def _pairs(mode: str) -> list[tuple[int, str]]:
    return [(s, a) for s in MODES[mode]["seeds"] for a in ARMS[mode]]


def run_shard(mode: str, gpu: int, shard: int, of: int) -> None:
    paths = prepare_arms()
    texts = [it["text"] for it in testset()]
    mine = [p for i, p in enumerate(_pairs(mode)) if i % of == shard]
    logger.info(f"=== {mode} shard {shard}/{of}: {len(mine)}쌍 ===")
    for s, arm in mine:
        if (CACHE / f"{arm}_s{s}.npz").exists():
            logger.info(f"  {arm} s{s}: 캐시 있음 — 건너뜀")
            continue
        d = train_arm(arm, s, paths[arm], gpu)
        load_probs(f"{arm}_s{s}", d, texts, gpu)
        w = Path(d) / "model.safetensors"
        if w.exists():
            w.unlink()
        logger.info(f"  {arm} s{s}: 학습·캐시 완료, 가중치 삭제")


def repro(gpu: int, tag: str = "") -> None:
    """D 팔 시드 42를 새 폴더에 다시 학습해 저장된 D_s42와 비교한다. tag가 있으면 추가 재현(run-to-run 변동 진단용, 판정 기준은 같다)."""
    orig = e8._model_dir
    e8._model_dir = lambda arm, seed: orig(arm, seed).with_name(orig(arm, seed).name + "_repro" + tag)
    try:
        paths = prepare_arms()
        d = train_arm("D", 42, paths["D"], gpu)
        texts = [it["text"] for it in testset()]
        P_new, thr_new = load_probs("D_s42_repro" + tag, d, texts, gpu)
    finally:
        e8._model_dir = orig
    z = np.load(CACHE / "D_s42.npz")
    P_old, thr_old = z["P"], z["thr"]
    pred_new, pred_old = P_new >= thr_new, P_old >= thr_old
    rep = {"rule": "eval/model/aihub_shard.py docstring (일치 = 예측 일치율 ≥ 98% 이고 평균 |ΔP| < 0.02, 실행 전 고정)",
           "pred_agreement_pct": round(float((pred_new == pred_old).mean()) * 100, 3),
           "mean_abs_dP": round(float(np.abs(P_new - P_old).mean()), 5), "max_abs_dP": round(float(np.abs(P_new - P_old).max()), 4),
           "same_shape": bool(P_new.shape == P_old.shape)}
    rep["reproduced"] = bool(rep["same_shape"] and rep["pred_agreement_pct"] >= 98 and rep["mean_abs_dP"] < 0.02)
    if tag:                                    # 첫 재현과의 차이 = 같은 시드·같은 코드의 run-to-run 변동
        z1 = np.load(CACHE / "D_s42_repro.npz")
        rep["vs_first_repro"] = {"pred_agreement_pct": round(float(((P_new >= thr_new) == (z1["P"] >= z1["thr"])).mean()) * 100, 3),
                                 "mean_abs_dP": round(float(np.abs(P_new - z1["P"]).mean()), 5)}
    save_json(rep, PROJECT_ROOT / f"data/eval/model/repro_fullft_d42{tag}.json")
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))
    w = Path(d) / "model.safetensors"
    if w.exists():
        w.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(ARMS))
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--repro-tag", default="")
    a = ap.parse_args()
    if a.repro:
        repro(a.gpu, a.repro_tag)
    if a.mode:
        run_shard(a.mode, a.gpu, a.shard, a.of)


if __name__ == "__main__":
    main()
