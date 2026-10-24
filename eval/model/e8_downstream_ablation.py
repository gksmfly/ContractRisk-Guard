# backend/eval/e8_downstream_ablation.py
"""E8 — 하류 학습 ablation (사전 등록, 미판정 예상)

## 설계 (paper_plan.md §E8)

팔: A (필터 없음, forward, CLEAN∪NOISE 2305건)
    B (무작위, forward, C와 같은 수 1473건)
    C (FB-Check, agreed, CLEAN 1473건) ← 현행
    (선택) D (CleanLab, forward, 1287건)
각 팔 3시드(42 · 7 · 123). 학습 설정은 article_v2와 동일.

## 2026-09-30 추가 팔 (P1 — E8 재분석, paper_plan.md §P1-1)

C는 A·B와 **두 가지가 동시에** 다르다 — 라벨 소스(agreed vs forward)와 음성 비율
(빈 라벨 59.2% vs 44~45%). C−B 차를 "필터 효과"로 귀속할 수 없다. 분해용 팔:

    D   CLEAN 레코드 + forward 라벨    A→D = 필터(레코드 선택) 효과, D→C = 라벨 소스 효과
                                       B↔D = 같은 수·같은 라벨 소스에서 무작위 vs FB-Check
    X        AI Hub 전문가 라벨 전량(평가셋 누수 제거본)   ← E_expert
    X3352    AI Hub에서 C와 같은 수(3,352)                ← 같은 N 비교
    X1000    AI Hub 1,000건                                ← 학습 곡선 한 점
    XC       AI Hub, 같은 시드 C의 실제 학습 수              ← 진짜 같은 N 비교

## 2026-10-04 추가 팔 — 필터 방법·구성요소의 하류 학습 (전부 forward 라벨, A에서 빼기만)

    G    Grounding만   A − snippet_not_found NOISE (33건)        ≈ A (구조상)
    K    Consistency만 A − label/article_mismatch NOISE (1,080건) ≈ D (구조상, D ∪ snippet 33건)
    CL   CleanLab      공정위 조항: A − CleanLab이 고른 832건 (`data/eval/label/e3_removed_ids.json`)
                       표준계약서·근거 없는 공정위 조항: D와 동일 (CleanLab OOF가 이 부분엔 없다)
                       → CL과 D는 **근거 있는 공정위 조항을 어느 규칙으로 고르느냐**만 다르다

## 2026-10-05 추가 — 백본 비교

    RB   C와 같은 데이터·설정, 백본만 klue/roberta-base(110M, KoELECTRA-base와 같은 크기)
         08월 백본 8종 비교는 누수 있던 옛 분할·옛 라벨 체계에서 잰 것이라 현행 설정으로 다시 잰다

판정: C−A · C−B 각각. CI가 0을 포함하면 "미판정, MDE ±4%p"로 보고.

## 실행

    # 데이터 파일 준비만:
    python -m eval.e8_downstream_ablation --prep-only

    # 학습 + 평가 (각 팔·시드 순차 실행, ~2-3일 소요):
    python -m eval.e8_downstream_ablation --arms A B C --gpu 0

    # 평가만 (모델 디렉터리 이미 있을 때):
    python -m eval.e8_downstream_ablation --eval-only --gpu 0

    # 결과 집계:
    python -m eval.e8_downstream_ablation --analyze
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json, save_jsonl

logger = load_logger("e8_downstream_ablation.log")

# ── 설정 ────────────────────────────────────────────────────────────────────

SEEDS = [42, 7, 123]
N_C   = 0     # 실행 시 clean.jsonl 크기로 동적 결정 (prepare_arms 참고)

CLEAN_PATH  = PROJECT_ROOT / "data/fb_check/clean.jsonl"
NOISE_PATH  = PROJECT_ROOT / "data/fb_check/noise.jsonl"
BASE_DIR    = PROJECT_ROOT / "models/article_v2"
HOLDOUT_PATH = PROJECT_ROOT / "backend/training/negative_holdout_docs.json"
ARM_DATA_DIR = PROJECT_ROOT / "data/eval/model/e8_arms"
MODEL_OUT    = PROJECT_ROOT / "models"
OUT_PATH     = PROJECT_ROOT / "data/eval/model/e8_downstream_ablation.json"

# arm B 무작위 선택 시드 (학습 시드와 분리)
B_SUBSET_SEED = 42

AIHUB_PATH = PROJECT_ROOT / "data/aihub_tos/train.jsonl"
# 백본만 바꾸는 팔 → 사전학습 모델 id (데이터·라벨은 C와 같다)
BACKBONE_ARMS = {"RB": "klue/roberta-base"}

# 말투 편향 줄이기 팔 (paper_plan §7-2) → (라벨 파일, 백본). 라벨 = agreed
STYLE_DIR = PROJECT_ROOT / "data/eval/style_aug"
STYLE_ARMS = {
    "KSN": (STYLE_DIR / "labels_SN.jsonl", None),
    "KSS": (STYLE_DIR / "labels_SS.jsonl", None),
    "RSN": (STYLE_DIR / "labels_SN.jsonl", "klue/roberta-base"),
    "RSS": (STYLE_DIR / "labels_SS.jsonl", "klue/roberta-base"),
    # 새 시험 데이터 재검증 (paper_plan §7-3): 새 시험 문서를 뺀 라벨
    "NC":   (PROJECT_ROOT / "data/eval/fresh/labels_NC.jsonl", None),
    "NRSN": (PROJECT_ROOT / "data/eval/fresh/labels_NRSN.jsonl", "klue/roberta-base"),
    # AI Hub 외부 평가 (paper_plan §7-8): 다중 검증자 FB-Check와 같은 수 무작위 제거 대조군
    "MV":   (ARM_DATA_DIR / "arm_MV_agreed.jsonl", None),
    "RMV":  (ARM_DATA_DIR / "arm_RMV_agreed.jsonl", None),
    # 외부 오경보 줄이기 (paper_plan §7-9 나): C + AI Hub 학습 절반 공정 조항 1,500건
    "CN":   (ARM_DATA_DIR / "arm_CN_agreed.jsonl", None),
}

# 2진 FB-Check 하류 학습 (paper_plan §7-16 ②): forward 라벨 팔, 파일은 prepare_arms가 만든다
#   D2 = 2진 규칙이 남긴 레코드 · B2 = A에서 D2와 같은 수 무작위(B와 같은 방식)
BINARY_ARMS = {"D2": ARM_DATA_DIR / "arm_D2_forward_binary.jsonl", "B2": ARM_DATA_DIR / "arm_B2_forward_random.jsonl"}
# FB-Check v2 (paper_plan §7-17): 버리지 않고 확신도 목표값(1.0/0.5)으로 학습. label_source=soft
SOFT_PATH = ARM_DATA_DIR / "arm_S_soft.jsonl"
# 인과 확인 (paper_plan §7-18): 표준계약서 출처의 NOISE만 뺀다(공정위 조항은 전부 남김). forward 라벨
DS_PATH = ARM_DATA_DIR / "arm_DS_forward_stdonly.jsonl"

# 어댑터 팔 (paper_plan §7-26): 같은 데이터에서 인코더를 LoRA로만 학습한다. LA = 필터 없음(A) 데이터, LD = FB-Check(D) 데이터
ADAPTER_ARMS = {"LA": "A", "LD": "D"}
ADAPTER_LR = 3e-4      # 사전 등록 고정값. LoRA는 전체 파인튜닝보다 큰 학습률을 쓰는 것이 일반적이다. 결과를 보고 바꾸지 않는다

# 라벨 문제 유형별 영향 (paper_plan §7-27): AI Hub 학습 절반의 같은 조항에서 라벨만 유형별로 정답으로 바꾼다
LABEL_TYPE_ARMS = ("OQ", "OFP", "OFN", "OMX", "OG")

# AI Hub 팔 → 학습 부분표본 크기(0 = 전량)
AIHUB_ARMS = {"X": 0, "X3352": 3352, "X1000": 1000, "XC": -1}
# XC: 같은 시드 C 팔의 **실제 학습 수**(gold 배제·홀드아웃·dev 분리 후)에 맞춘다. C의 3,352는 분리 전
# 파일 건수라 실제 학습은 1,813~2,074건이다 — X3352는 C보다 1.7배 많이 학습한 것이었다(10-03 발견)


# ── 데이터 준비 ──────────────────────────────────────────────────────────────

def _doc_of(chunk_id: str) -> str:
    parts = str(chunk_id).split(":")
    return ":".join(parts[:2]) if len(parts) >= 3 else str(chunk_id)


def prepare_arms() -> dict[str, Path]:
    """각 팔의 라벨 JSONL 파일 생성. 이미 있으면 덮어쓰지 않는다."""
    ARM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    holdout_docs = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))

    clean  = load_jsonl(CLEAN_PATH)
    noise  = load_jsonl(NOISE_PATH)
    all_records = clean + noise

    logger.info(f"  원본: CLEAN {len(clean)} + NOISE {len(noise)} = {len(all_records)}")

    paths: dict[str, Path] = {}

    # ── Arm C: CLEAN 현행 (agreed labels) ────────────────────────────────────
    # 기존 clean.jsonl을 그대로 씀 — 재생성 불필요
    paths["C"] = CLEAN_PATH
    logger.info(f"  Arm C: {CLEAN_PATH} (기존, agreed, {len(clean)}건)")

    # ── Arm A: CLEAN∪NOISE (forward labels) ──────────────────────────────────
    arm_a_path = ARM_DATA_DIR / "arm_A_forward_all.jsonl"
    if not arm_a_path.exists():
        # forward_articles가 있는 레코드만 (CLEAN은 항상, NOISE도 있어야 함)
        a_recs = []
        for r in all_records:
            arts = r.get("forward_articles")
            if arts is None:
                continue
            # agreed_articles를 forward_articles 값으로 덮어써서 train_article이 forward를 쓰게 함
            # label_source=forward를 쓰면 r["forward_articles"]를 직접 읽으므로 원본을 그대로 저장
            a_recs.append(r)
        save_jsonl(a_recs, arm_a_path)
        logger.info(f"  Arm A: {arm_a_path} 저장 (forward, {len(a_recs)}건)")
    else:
        logger.info(f"  Arm A: {arm_a_path} 이미 있음")
    paths["A"] = arm_a_path

    # ── Arm B: 무작위 len(clean)건 (forward labels) ──────────────────────────
    n_c = len(clean)  # C와 같은 크기
    arm_b_path = ARM_DATA_DIR / "arm_B_forward_random.jsonl"
    if not arm_b_path.exists():
        a_recs = load_jsonl(arm_a_path)
        holdout_set = set(holdout_docs)

        # holdout docs는 반드시 포함 (chunk_id "source:doc" 단위)
        must_include = [r for r in a_recs
                       if _doc_of(r.get("chunk_id","")) in holdout_set]
        rest         = [r for r in a_recs
                       if _doc_of(r.get("chunk_id","")) not in holdout_set]

        rng = np.random.default_rng(B_SUBSET_SEED)
        n_random = n_c - len(must_include)
        if n_random < 0:
            logger.warning(f"  holdout({len(must_include)}) > n_c({n_c}) — 그냥 holdout만")
            b_recs = must_include
        else:
            selected = rng.choice(len(rest), size=min(n_random, len(rest)), replace=False)
            b_recs = must_include + [rest[i] for i in sorted(selected)]

        save_jsonl(b_recs, arm_b_path)
        logger.info(f"  Arm B: {arm_b_path} 저장 (forward, random {len(b_recs)}건, "
                    f"holdout docs {len(set(_doc_of(r.get('chunk_id','')) for r in must_include))}개 보장)")
    else:
        logger.info(f"  Arm B: {arm_b_path} 이미 있음")
    paths["B"] = arm_b_path

    # ── Arm D: CLEAN 레코드 + forward 라벨 (C와 레코드 동일, 라벨 소스만 A·B와 같게) ──
    paths["D"] = CLEAN_PATH
    for arm in BACKBONE_ARMS:
        paths[arm] = CLEAN_PATH
    for arm, (labels, _) in STYLE_ARMS.items():
        paths[arm] = labels
    for arm in AIHUB_ARMS:
        paths[arm] = AIHUB_PATH

    # ── G / K / CL: A(forward 전체)에서 규칙별로 빼기 ──
    a_recs = load_jsonl(arm_a_path)
    reason = lambda r: str(r.get("noise_reason") or "").split(":")[0]  # noqa: E731
    rules = {
        "G": lambda r: not (r.get("status") == "NOISE" and reason(r) == "snippet_not_found"),
        "K": lambda r: not (r.get("status") == "NOISE" and reason(r) in ("label_mismatch", "article_mismatch")),
    }
    removed_path = PROJECT_ROOT / "data/eval/label/e3_removed_ids.json"
    if removed_path.exists():
        from eval.label.label_ablation import _ref_by_doc
        cl_removed = set(json.loads(removed_path.read_text(encoding="utf-8"))["cleanlab_removed"])
        ref = _ref_by_doc()

        def _keep_cl(r: dict) -> bool:
            if r.get("source") == "ftc_case" and _doc_of(r.get("chunk_id", "")) in ref:
                return r["chunk_id"] not in cl_removed      # E3 모집단: CleanLab 규칙
            return r.get("status") == "CLEAN"               # 그 밖: D와 동일
        rules["CL"] = _keep_cl
    for arm, keep in rules.items():
        out = ARM_DATA_DIR / f"arm_{arm}_forward.jsonl"
        if not out.exists():
            recs = [r for r in a_recs if keep(r)]
            save_jsonl(recs, out)
            logger.info(f"  Arm {arm}: {out} 저장 ({len(recs)}건, A 대비 −{len(a_recs) - len(recs)})")
        paths[arm] = out

    # ── D2 / B2: 2진 FB-Check와 같은 수 무작위 (§7-16 ②) ──
    from eval.label.fbcheck_binary import binary_keep
    d2 = BINARY_ARMS["D2"]
    if not d2.exists():
        recs = [r for r in a_recs if binary_keep(r)]
        save_jsonl(recs, d2)
        logger.info(f"  Arm D2: {d2} 저장 ({len(recs)}건)")
    n_d2 = len(load_jsonl(d2))
    b2 = BINARY_ARMS["B2"]
    if not b2.exists():
        holdout_set = set(holdout_docs)
        must = [r for r in a_recs if _doc_of(r.get("chunk_id", "")) in holdout_set]
        rest = [r for r in a_recs if _doc_of(r.get("chunk_id", "")) not in holdout_set]
        sel = np.random.default_rng(B_SUBSET_SEED).choice(len(rest), size=min(n_d2 - len(must), len(rest)), replace=False)
        recs = must + [rest[i] for i in sorted(sel)]
        save_jsonl(recs, b2)
        logger.info(f"  Arm B2: {b2} 저장 (random {len(recs)}건)")
    paths.update(BINARY_ARMS)

    # ── S: FB-Check v2 — 근거 없음·verify 위반 아님만 버린다 (§7-17) ──
    if not SOFT_PATH.exists():
        def _keep_s(r: dict) -> bool:
            f, v = r.get("forward_articles") or [], r.get("verify_articles") or []
            return not (f and (not r.get("snippet_exists", True) or not v))
        recs = [r for r in a_recs if _keep_s(r)]
        save_jsonl(recs, SOFT_PATH)
        logger.info(f"  Arm S: {SOFT_PATH} 저장 ({len(recs)}건)")
    paths["S"] = SOFT_PATH
    paths["LA"] = paths["A"]
    paths["LD"] = CLEAN_PATH
    paths.update(_prepare_label_type_arms())

    # ── DS: 표준계약서 NOISE만 제외 (§7-18) ──
    if not DS_PATH.exists():
        recs = [r for r in a_recs if not (r.get("source") == "standard_contract" and r.get("status") == "NOISE")]
        save_jsonl(recs, DS_PATH)
        logger.info(f"  Arm DS: {DS_PATH} 저장 ({len(recs)}건)")
    paths["DS"] = DS_PATH

    return paths


def _prepare_label_type_arms() -> dict[str, Path]:
    """§7-27: AI Hub **학습 절반** 조항(2,579건, §7-14 행)에 Qwen 라벨·전문가 정답을 붙여 팔별 라벨 파일을 만든다. 이미 있으면 그대로."""
    from eval.label.label_type_fix import ARMS, fixed_labels
    out = {a: ARM_DATA_DIR / f"arm_{a}_labeltype.jsonl" for a in LABEL_TYPE_ARMS}
    if all(f.exists() for f in out.values()):
        return out
    from eval.label.fbcheck_aihub import RAW, population
    train_files = set(json.loads((PROJECT_ROOT / "data/eval/model/aihub_split.json").read_text(encoding="utf-8"))["train_files"])
    raw = {r["key"]: r for r in load_jsonl(RAW)}
    rows = [(p, raw[p["key"]]) for p in population() if p["group"] in train_files]
    for arm, fix in ARMS.items():
        recs = []
        for p, r in rows:
            labels = fixed_labels(set(r["forward"]), set(p["articles"]), p["label"] == 1, fix)
            recs.append({"text": p["text"], "articles": labels, "forward_articles": labels, "agreed_articles": labels,
                         "source": "aihub", "chunk_id": p["group"]})
        save_jsonl(recs, out[arm])
        logger.info(f"  Arm {arm}: {out[arm]} 저장 ({len(recs)}건, 위반 라벨 있는 조항 {sum(bool(x['articles']) for x in recs)})")
    return out


# ── 학습 ─────────────────────────────────────────────────────────────────────

def _arm_label_source(arm: str) -> str:
    if arm == "S":
        return "soft"
    # AI Hub는 agreed == forward == 전문가 라벨이라 어느 쪽이든 같다
    return "agreed" if arm == "C" or arm in AIHUB_ARMS or arm in BACKBONE_ARMS or arm in STYLE_ARMS else "forward"


def _model_dir(arm: str, seed: int) -> Path:
    return MODEL_OUT / f"_e8_{arm}_s{seed}"


def train_arm(arm: str, seed: int, labels_path: Path, gpu: int) -> Path:
    """하나의 팔·시드 학습. 이미 완료됐으면 건너뜀."""
    out_dir = _model_dir(arm, seed)
    done_flag = out_dir / "metrics.json"
    if done_flag.exists():
        logger.info(f"  Arm {arm} seed {seed}: {out_dir} 이미 있음 — 건너뜀")
        return out_dir

    metrics_v2 = json.loads((BASE_DIR / "metrics.json").read_text(encoding="utf-8"))
    tc = metrics_v2["train_config"]

    argv = [
        sys.executable, "-m", "backend.training.train_article",
        "--labels",   str(labels_path),
        "--model-dir", str(out_dir),
        "--epochs",    str(tc["epochs"]),
        "--batch-size", str(tc["batch_size"]),
        "--lr",         str(ADAPTER_LR if arm in ADAPTER_ARMS else tc["lr"]),
        "--max-len",    str(tc["max_len"]),
        "--min-support", str(tc["min_support"]),
        "--seed",        str(seed),
        "--label-source", _arm_label_source(arm),
        "--gpu", str(gpu),
    ]
    # 음성 홀드아웃은 표준계약서 문서 id라 AI Hub 데이터에는 없다(학습기가 "문서가 없다"로 멈춘다).
    # AI Hub 팔은 그 문서를 애초에 못 보므로 떼어낼 것도 없다 — 평가셋은 모든 팔이 같다.
    if arm not in AIHUB_ARMS and arm not in LABEL_TYPE_ARMS:
        argv += ["--negative-holdout", str(metrics_v2["negative_holdout_docs"]),
                 "--negative-holdout-docs", str(HOLDOUT_PATH)]
    n_sub = AIHUB_ARMS.get(arm, 0)
    if n_sub == -1:
        n_sub = json.loads((_model_dir("C", seed) / "metrics.json").read_text(encoding="utf-8"))["train_samples"]
    if n_sub:
        argv += ["--train-subsample", str(n_sub)]
    if metrics_v2.get("negative_ratio") is not None:
        argv += ["--negative-ratio", str(metrics_v2["negative_ratio"])]
    if arm in ADAPTER_ARMS:
        argv += ["--adapter", "lora", "--lora-r", "8", "--lora-alpha", "16"]

    logger.info(f"  Arm {arm} seed {seed}: 학습 시작 → {out_dir}")
    logger.info(f"    명령: {' '.join(argv[2:])}")
    env = None
    backbone = BACKBONE_ARMS.get(arm) or (STYLE_ARMS[arm][1] if arm in STYLE_ARMS else None)
    if backbone:
        import os
        env = {**os.environ, "BASE_MODEL": backbone}
    result = subprocess.run(argv, capture_output=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Arm {arm} seed {seed} 학습 실패 (returncode={result.returncode})")

    logger.info(f"  Arm {arm} seed {seed}: 완료")
    return out_dir


# ── 평가 ─────────────────────────────────────────────────────────────────────

def eval_model(model_dir: Path, gpu: int) -> dict:
    """단일 모델 평가 — gold 재현·홀드아웃 FPR·조 F1."""
    from backend.training.retrain import eval_sets, f1_rows, predict

    holdout_docs = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
    gold, neg    = eval_sets(holdout_docs)
    neg_groups   = [":".join(str(r["chunk_id"]).split(":")[:2]) for r in neg]

    P, thr, names = predict(model_dir, [g["text"] for g in gold], gpu)
    Pn, *_        = predict(model_dir, [r["text"] for r in neg], gpu)
    pred   = P >= thr
    labels = np.array([[a in g.get("articles", []) for a in names] for g in gold])

    recall = pred.any(1).astype(float)
    fpr    = (Pn >= thr).any(1).astype(float)
    f1     = f1_rows(pred, labels)

    return {
        "dir":          str(model_dir),
        "n_gold":       int(len(gold)),
        "n_holdout":    int(len(neg)),
        "gold_recall":  float(recall.mean()),
        "holdout_fpr":  float(fpr.mean()),
        "gold_f1":      float(f1.mean()),
        "_arrays": {"recall": recall, "fpr": fpr, "f1": f1,
                    "neg_groups": neg_groups},
    }


# ── 집계 및 판정 ─────────────────────────────────────────────────────────────

def _paired_ci(diff: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float, float]:
    """페어드 부트스트랩 CI (mean, lo, hi). 단위: %p."""
    rng = np.random.default_rng(seed)
    point = float(diff.mean() * 100)
    if len(diff) < 2:
        return point, float("nan"), float("nan")
    boots = np.array([
        rng.choice(diff, size=len(diff), replace=True).mean() * 100
        for _ in range(n_boot)
    ])
    boots = np.sort(boots)
    return point, float(boots[int(n_boot * 0.025)]), float(boots[int(n_boot * 0.975)])


def _arm_summary(arm: str, seed_evals: list[dict]) -> dict:
    """3시드 평균 + CI."""
    recalls = np.array([e["gold_recall"] for e in seed_evals])
    fprs    = np.array([e["holdout_fpr"] for e in seed_evals])
    f1s     = np.array([e["gold_f1"]     for e in seed_evals])
    return {
        "arm":           arm,
        "seeds":         SEEDS[:len(seed_evals)],
        "gold_recall_pp": round(float(recalls.mean() * 100), 2),
        "holdout_fpr_pp": round(float(fprs.mean() * 100), 2),
        "gold_f1_pp":     round(float(f1s.mean() * 100), 2),
        "recall_std_pp":  round(float(recalls.std() * 100), 2),
        "fpr_std_pp":     round(float(fprs.std() * 100), 2),
    }


def _compare_arms(arm_c_evals: list[dict], arm_x_evals: list[dict],
                  label: str) -> dict:
    """C와 X를 개별 레코드 단위 페어드 비교."""
    c_arrs = [e["_arrays"] for e in arm_c_evals]
    x_arrs = [e["_arrays"] for e in arm_x_evals]

    # 각 시드의 diff를 concat해 분포로
    recall_diffs = np.concatenate([c["recall"] - x["recall"]
                                   for c, x in zip(c_arrs, x_arrs)])
    fpr_diffs    = np.concatenate([c["fpr"] - x["fpr"]
                                   for c, x in zip(c_arrs, x_arrs)])
    f1_diffs     = np.concatenate([c["f1"] - x["f1"]
                                   for c, x in zip(c_arrs, x_arrs)])

    def _ci(diff: np.ndarray) -> dict:
        mean_pp, lo, hi = _paired_ci(diff)
        zero_included = lo <= 0 <= hi
        verdict = ("미판정, MDE ±4%p" if zero_included
                   else ("C > " + label if mean_pp > 0 else label + " > C"))
        return {"mean_pp": round(mean_pp, 2), "ci95": [round(lo, 2), round(hi, 2)],
                "verdict": verdict}

    return {
        "comparison": f"C vs {label}",
        "n_per_seed": int(len(recall_diffs) // max(len(c_arrs), 1)),
        "recall": _ci(recall_diffs),
        "fpr":    _ci(fpr_diffs),
        "f1":     _ci(f1_diffs),
        "mde_pp": 4.3,
        "note":   "article_v1→v2 페어드 CI 반폭 ±4.3%p 기준 (paper_plan §E8)",
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms",      nargs="+", choices=["A", "B", "C", "D", "X", "X3352", "X1000", "XC", "CL", "G", "K", "RB", "KSN", "KSS", "RSN", "RSS", "NC", "NRSN", "MV", "RMV", "CN", "D2", "B2", "S", "DS", "LA", "LD", *LABEL_TYPE_ARMS],
                        default=["A", "B", "C"])
    parser.add_argument("--seeds",     nargs="+", type=int, default=SEEDS)
    parser.add_argument("--gpu",       type=int,  default=0)
    parser.add_argument("--prep-only", action="store_true", help="데이터 파일만 만들고 멈춤")
    parser.add_argument("--eval-only", action="store_true", help="이미 학습된 모델만 평가")
    parser.add_argument("--analyze",   action="store_true", help="결과 집계만")
    args = parser.parse_args()

    if args.analyze:
        _analyze(args.gpu, args.arms)
        return

    logger.info("=== E8 하류 학습 ablation ===")
    arm_paths = prepare_arms()
    logger.info("  데이터 파일 준비 완료")

    if args.prep_only:
        for arm, p in arm_paths.items():
            records = load_jsonl(p) if p.exists() else []
            logger.info(f"  {arm}: {p} ({len(records)}건)")
        return

    for arm in args.arms:
        labels_path = arm_paths[arm]
        for seed in args.seeds:
            if not args.eval_only:
                train_arm(arm, seed, labels_path, args.gpu)
            out_dir = _model_dir(arm, seed)
            if not (out_dir / "metrics.json").exists():
                logger.warning(f"  {out_dir} 없음 — 건너뜀")
                continue
            logger.info(f"  평가: Arm {arm} seed {seed}")
            ev = eval_model(out_dir, args.gpu)
            logger.info(
                f"    recall={ev['gold_recall']*100:.1f}%  "
                f"fpr={ev['holdout_fpr']*100:.1f}%  "
                f"f1={ev['gold_f1']*100:.1f}%"
            )

    _analyze(args.gpu, args.arms)


def _analyze(gpu: int, arms: list[str]) -> None:
    logger.info("=== E8 분석 ===")
    arm_evals: dict[str, list[dict]] = {}

    for arm in ["A", "B", "C"]:
        seed_evals = []
        for seed in SEEDS:
            d = _model_dir(arm, seed)
            if not (d / "metrics.json").exists():
                logger.warning(f"  {d} 없음 — 건너뜀")
                continue
            ev = eval_model(d, gpu)
            seed_evals.append(ev)
        if seed_evals:
            arm_evals[arm] = seed_evals

    if not arm_evals:
        logger.error("  평가할 모델이 없다 — 먼저 학습을 실행하세요")
        return

    summaries = {arm: _arm_summary(arm, evs) for arm, evs in arm_evals.items()}
    comparisons: dict[str, Any] = {}
    if "C" in arm_evals and "A" in arm_evals:
        comparisons["C_vs_A"] = _compare_arms(arm_evals["C"], arm_evals["A"], "A")
    if "C" in arm_evals and "B" in arm_evals:
        comparisons["C_vs_B"] = _compare_arms(arm_evals["C"], arm_evals["B"], "B")

    report = {
        # 이 함수의 판정은 철회됐다 — 학습 팔을 추가로 돌릴 때마다 이 파일이 다시 써지므로 표시를 코드에 둔다
        "RETRACTED":   ("철회(2026-10-03): 제13조 헤드 인공물 + 팔별 동작점 비교. "
                        "대체 data/eval/model/e8b_matched.json, 근거 eval/reported_numbers.md §8"),
        "experiment":  "E8",
        "description": "하류 학습 ablation (사전 등록, 미판정 예상)",
        "arms":        {"A": "필터 없음(forward, 2305건)", "B": "무작위(forward, 1473건)",
                        "C": "FB-Check(agreed, 1473건)"},
        "seeds":       SEEDS,
        "mde_pp":      4.3,
        "judgment_rule": "CI 하한 > 0 → C > X, CI 상한 < 0 → X > C, 포함 → 미판정 MDE±4%p",
        "summaries":   summaries,
        "comparisons": comparisons,
    }

    save_json(report, OUT_PATH)
    logger.info(f"저장: {OUT_PATH}")

    for arm, s in summaries.items():
        logger.info(f"  Arm {arm}: recall={s['gold_recall_pp']}% fpr={s['holdout_fpr_pp']}% "
                    f"f1={s['gold_f1_pp']}% (±{s['recall_std_pp']}%p std)")
    for name, comp in comparisons.items():
        r = comp["recall"]
        f = comp["fpr"]
        logger.info(f"  {comp['comparison']}: recall Δ={r['mean_pp']:+.1f}%p {r['ci95']} → {r['verdict']} | "
                    f"fpr Δ={f['mean_pp']:+.1f}%p → {f['verdict']}")


if __name__ == "__main__":
    main()
