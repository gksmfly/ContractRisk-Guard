# backend/fb_check/__main__.py
"""
FB-Check 오케스트레이터 (Step 6)

3단계 파이프라인을 조합하여 seed 라벨의 CLEAN/NOISE를 판정한다.

  forward_labeling.py        : Forward Labeling   C → L + evidence_span
  backward_grounding.py      : Backward Grounding E ⊂ C 검증 + KoELECTRA 예측
  consistency_verification.py: Consistency Verify E → L' 재라벨링

  Decision: forward_label(GPT) / verify_label(GPT) / backward_risk(KoELECTRA, 독립 모델)
  세 신호 중 2개 이상 일치하면 CLEAN (2/3 다수결). snippet_exists(E⊂C)는 사전 조건.

  예전엔 backward_risk가 계산만 되고 판정엔 안 쓰여서(forward==verify만 봄) GPT 혼자
  일관되게 틀려도(systematic bias) 못 걸렀다 — KoELECTRA를 실제로 반영해 교차검증한다.

출력:
    data/fb_check/fb_check_results.jsonl
    data/fb_check/clean.jsonl
    data/fb_check/noise.jsonl
    data/fb_check/fb_check_report.json

사용법:
    python -m backend.fb_check
    python -m backend.labeling.fb_check --sample 200 --gpu 1
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from openai import OpenAI
from transformers import ElectraTokenizerFast

from backend.fb_check.forward_labeling import run_forward
from backend.fb_check.backward_grounding import load_model, snippet_exists, predict
from backend.fb_check.consistency_verification import run_verify
from backend.model.electra import DualHeadElectra
from backend.utils import load_jsonl, load_logger, save_json, save_jsonl, PROJECT_ROOT

logger = load_logger("fb_check.log")

SEED_PATH = Path(os.environ.get("SEED_PATH",  str(PROJECT_ROOT / "data/labels/seed_labeled.jsonl")))
MODEL_DIR = Path(os.environ.get("MODEL_DIR",  str(PROJECT_ROOT / "models/v1")))
OUT_DIR   = Path(os.environ.get("FB_OUT_DIR", str(PROJECT_ROOT / "data/fb_check")))


def run_fb_check(
    record: dict,
    client: OpenAI,
    model: DualHeadElectra,
    tokenizer: ElectraTokenizerFast,
    device: torch.device,
) -> dict[str, Any]:
    clause_text = record["text"]
    result: dict[str, Any] = {
        "fb_id":       f"fb:{record['chunk_id']}",
        "chunk_id":    record["chunk_id"],
        "source":      record["source"],
        "text":        clause_text,
        "seed_domain": record["domain"],
        "seed_risk":   record["risk_level"],
    }

    # --- Forward Labeling: C → L ---
    forward = run_forward(client, clause_text)
    if not forward:
        result.update({"status": "ERROR", "error": "forward_failed"})
        return result

    forward_label  = forward.get("risk_level", "")
    forward_domain = forward.get("domain", "")
    evidence_span  = forward.get("evidence_span", "")
    reasoning      = forward.get("reasoning", "")
    result.update({
        "forward_label":  forward_label,
        "forward_domain": forward_domain,
        "evidence_span":  evidence_span,
        "reasoning":      reasoning,
    })

    # --- Backward Grounding: E ⊂ C 검증 + KoELECTRA ---
    span_exists = snippet_exists(clause_text, evidence_span)
    backward_domain, backward_risk = predict(clause_text, model, tokenizer, device)
    result.update({
        "snippet_exists":  span_exists,
        "backward_domain": backward_domain,
        "backward_risk":   backward_risk,
    })

    if forward_domain == "해당없음":
        result.update({"status": "NOISE", "noise_reason": "domain_none"})
        return result

    if not span_exists:
        result.update({"status": "NOISE", "noise_reason": "snippet_not_found"})
        return result

    # --- Consistency Verify: E → L' ---
    verify = run_verify(client, evidence_span)
    if not verify:
        result.update({"status": "ERROR", "error": "verify_failed"})
        return result

    verify_label  = verify.get("risk_level", "")
    verify_domain = verify.get("domain", "")
    result.update({"verify_label": verify_label, "verify_domain": verify_domain})

    # --- Decision: forward_label(GPT)·verify_label(GPT)·backward_risk(KoELECTRA, 독립 모델)
    # 세 신호 중 2개 이상 일치하면 CLEAN. 기존엔 forward==verify(GPT vs GPT)만 봐서
    # backward_risk가 계산은 되지만 판정에 전혀 안 쓰였다 — 이러면 GPT 혼자 일관되게
    # 틀려도(systematic bias) 걸러내지 못한다. KoELECTRA는 별도 모델이라 GPT와 같은
    # 편향을 공유할 가능성이 낮아, 진짜 교차검증(Forward-*Backward*-Verify)이 되게 한다.
    votes = [v for v in (forward_label, verify_label, backward_risk) if v]
    label, count = Counter(votes).most_common(1)[0] if votes else (None, 0)

    if count >= 2:
        result["status"] = "CLEAN"
        result["final_label"] = label
    else:
        result["status"] = "NOISE"
        result["noise_reason"] = (
            f"no_majority: forward={forward_label} verify={verify_label} backward={backward_risk}"
        )

    return result


def build_report(results: list[dict]) -> tuple[dict[str, Any], list[dict], list[dict]]:
    total = len(results)
    clean_n = noise_n = error_n = noise_snippet = noise_mismatch = noise_domain_none = 0
    clean_risk: dict[str, int] = {}
    clean_domain: dict[str, int] = {}
    clean: list[dict] = []
    noise: list[dict] = []

    for r in results:
        status = r.get("status")
        if status == "CLEAN":
            clean_n += 1
            clean.append(r)
            # 2/3 다수결로 나온 final_label이 forward_label과 다를 수 있다
            # (예: verify+backward가 forward와 다른 라벨에 동의한 경우)
            v = r.get("final_label", r.get("forward_label", "unknown"))
            clean_risk[v] = clean_risk.get(v, 0) + 1
            v = r.get("forward_domain", "unknown")
            clean_domain[v] = clean_domain.get(v, 0) + 1
        elif status == "NOISE":
            noise_n += 1
            noise.append(r)
            reason = r.get("noise_reason", "")
            if reason == "snippet_not_found":
                noise_snippet += 1
            elif reason.startswith("no_majority") or reason.startswith("label_mismatch"):
                noise_mismatch += 1
            elif reason == "domain_none":
                noise_domain_none += 1
        elif status == "ERROR":
            error_n += 1

    report = {
        "총_입력":           total,
        "CLEAN":             clean_n,
        "NOISE":             noise_n,
        "ERROR":             error_n,
        "CLEAN_비율":        round(clean_n / total, 4) if total else 0,
        "NOISE_비율":        round(noise_n / total, 4) if total else 0,
        "노이즈_원인": {
            "domain_none":       noise_domain_none,
            "snippet_not_found": noise_snippet,
            "no_majority":       noise_mismatch,
        },
        "CLEAN_risk_분포":   clean_risk,
        "CLEAN_domain_분포": clean_domain,
    }
    return report, clean, noise


def _load_checkpoint(results_path: Path) -> set[str]:
    """이미 처리된 chunk_id 집합을 반환한다."""
    if not results_path.exists():
        return set()
    done: set[str] = set()
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line).get("chunk_id", ""))
    return done


def _append_result(results_path: Path, result: dict) -> None:
    """결과 1건을 파일에 즉시 추가한다."""
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FB-Check: Forward-Backward Consistency Check")
    parser.add_argument("--input",     default=str(SEED_PATH), help="입력 JSONL 경로")
    parser.add_argument("--sample",    type=int, default=0,    help="샘플 수 (0=전체)")
    parser.add_argument("--gpu",       type=int, default=0,    help="GPU 인덱스")
    parser.add_argument("--save-every",type=int, default=50,   help="clean/noise 중간 저장 주기")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"========== FB-Check 시작 | device={device} ==========")

    records = load_jsonl(Path(args.input))
    if args.sample > 0:
        records = records[:args.sample]
    logger.info(f"  입력: {len(records)}건 ({args.input})")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model, tokenizer = load_model(MODEL_DIR, device)
    logger.info(f"  KoELECTRA 로드 완료: {MODEL_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "fb_check_results.jsonl"

    # 체크포인트: 이미 처리된 건 건너뜀
    done_ids = _load_checkpoint(results_path)
    if done_ids:
        logger.info(f"  체크포인트 복원: {len(done_ids)}건 건너뜀")
    pending = [r for r in records if r["chunk_id"] not in done_ids]
    logger.info(f"  처리 대상: {len(pending)}건")

    clean_n = noise_n = 0

    for i, record in enumerate(pending, 1):
        result = run_fb_check(record, client, model, tokenizer, device)
        _append_result(results_path, result)          # 즉시 디스크에 기록

        if result.get("status") == "CLEAN":
            clean_n += 1
        elif result.get("status") == "NOISE":
            noise_n += 1

        if i % 10 == 0 or i == len(pending):
            logger.info(f"  [{i}/{len(pending)}] CLEAN={clean_n} NOISE={noise_n}")

        # 주기적으로 clean/noise 파일 갱신
        if i % args.save_every == 0:
            all_results = load_jsonl(results_path)
            report, clean, noise = build_report(all_results)
            save_jsonl(clean, OUT_DIR / "clean.jsonl")
            save_jsonl(noise, OUT_DIR / "noise.jsonl")
            save_json(report, OUT_DIR / "fb_check_report.json")
            logger.info(f"  중간 저장 완료 ({i}건)")

    # 최종 저장
    all_results = load_jsonl(results_path)
    report, clean, noise = build_report(all_results)
    save_jsonl(clean, OUT_DIR / "clean.jsonl")
    save_jsonl(noise, OUT_DIR / "noise.jsonl")
    save_json(report, OUT_DIR / "fb_check_report.json")

    logger.info(f"  결과: CLEAN={report['CLEAN']} / NOISE={report['NOISE']} / ERROR={report['ERROR']}")
    logger.info(f"  CLEAN 비율: {report['CLEAN_비율']:.1%}")
    logger.info(f"  노이즈 원인: {report['노이즈_원인']}")
    logger.info("========== FB-Check 완료 ==========")


if __name__ == "__main__":
    main()
