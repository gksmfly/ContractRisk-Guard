# backend/fb_check/__main__.py
"""
FB-Check 오케스트레이터 (Step 6)

3단계 파이프라인을 조합하여 seed 라벨의 CLEAN/NOISE를 판정한다.

  forward_labeling.py        : Forward Labeling   C → 위반 유형 A + 라벨 L + 근거 문구 E
  backward_grounding.py      : Backward Grounding E ⊂ C 인덱스 검증 (+ KoELECTRA 예측은 기록만)
  consistency_verification.py: Consistency Verify E → A' + L' 재라벨링

  Decision: **L == L' → CLEAN** (논문 정의, KAICTS 2025). E⊂C는 사전 조건.
  위험 판정인데 A ∩ A'가 비면 NOISE(article_mismatch) — 조 라벨을 학습에 쓰기 위한 추가 조건.

  KoELECTRA는 투표하지 않는다. 자세한 근거는 run_fb_check()의 Decision 주석 참고.

세 단계 모두의 원시 신호를 결과 레코드에 남긴다(`backward_risk`, `forward_articles`,
`verify_articles`, `verify_mode`, 각 단계의 `*_model`/`*_prompt`). 판정 규칙을 바꿔
비교할 때 LLM을 다시 호출하지 않기 위해서다 — `backend/eval/fbcheck_variant_compare.py`가
이 기록으로 규칙 변형을 오프라인 비교한다.

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

    forward_label    = forward.get("risk_level", "")
    forward_articles = forward.get("articles", [])
    forward_domain   = forward.get("domain", "")
    evidence_span    = forward.get("evidence_span", "")
    reasoning        = forward.get("reasoning", "")
    result.update({
        "forward_label":     forward_label,
        "forward_articles":  forward_articles,
        "forward_domain":    forward_domain,
        "evidence_span":     evidence_span,
        "reasoning":         reasoning,
        # 재현성: 어느 모델·프롬프트가 이 라벨을 만들었는지 레코드에 남긴다.
        # 이전 산출물(clean.jsonl)에는 이 정보가 없어 사후 확인이 불가능했다.
        "forward_model":     forward.get("model"),
        "forward_prompt":    forward.get("prompt_version"),
    })

    # --- Backward Grounding: E ⊂ C 검증 + KoELECTRA ---
    span_exists = snippet_exists(clause_text, evidence_span)
    backward_domain, backward_risk = predict(clause_text, model, tokenizer, device)
    result.update({
        "snippet_exists":  span_exists,
        "backward_domain": backward_domain,
        "backward_risk":   backward_risk,
    })

    # --- Consistency Verify ---
    # 위반 유형을 찾았으면 근거 문구만 주고(span), "위반 없음"이면 조항 전문을 준다(clause).
    #
    # 예전에는 여기서 `forward_domain == "해당없음"`이면 곧장 NOISE로 버렸다. 그 규칙이
    # 입력 2,218건 중 1,257건(56.7%)을 탈락시켰고, 그중 349건은 공정위가 불공정으로
    # 확정한 ftc_case 조항이었다. 유형이 2개뿐이라 나머지 유형(제10·11·12·14조 등)이
    # 전부 "해당없음"으로 밀려난 것이 원인이다.
    #
    # 유형을 약관규제법 9개로 늘린 지금, 빈 `articles`는 "우리 taxonomy 밖"이 아니라
    # "위반 없음"을 뜻한다 — 즉 Low 클래스의 정당한 학습 표본이므로 버리지 않고 검증한다.
    if forward_articles:
        if not span_exists:
            result.update({"status": "NOISE", "noise_reason": "snippet_not_found"})
            return result
        verify_mode = "span"
        verify_input = evidence_span
    else:
        verify_mode = "clause"
        verify_input = clause_text

    verify = run_verify(client, verify_input, mode=verify_mode)
    if not verify:
        result.update({"status": "ERROR", "error": "verify_failed"})
        return result

    verify_label    = verify.get("risk_level", "")
    verify_articles = verify.get("articles", [])
    verify_domain   = verify.get("domain", "")
    result.update({
        "verify_label":    verify_label,
        "verify_articles": verify_articles,
        "verify_domain":   verify_domain,
        "verify_mode":     verify_mode,
        "verify_model":    verify.get("model"),
        "verify_prompt":   verify.get("prompt_version"),
    })

    # --- Decision: L == L' → CLEAN (논문 정의) ---
    #
    # KoELECTRA(backward)는 **투표하지 않는다.** 논문(KAICTS 2025)이 정의한 역할은
    # `E ⊂ C` 인덱스 검증이고, FB-Check 최초 구현도 `forward_label == verify_label`만
    # 봤다(커밋 edb832d, 주석 "논문 정의"). 3표 다수결은 그 뒤에 추가된 것으로,
    # 논문과 구현이 어긋난 상태였다 — 지금 학습에 쓰는 clean.jsonl 694건은 논문
    # 방법으로 만든 데이터가 아니다.
    #
    # 되돌리는 근거(`backend/eval/fbcheck_variant_compare.py` 실측):
    #   - KoELECTRA 표는 상수 투표자 대비 순 기여 +6.1%p에 그친다
    #   - 현행 CLEAN High 184건 중 82건이 이 표의 캐스팅보트 산물이다
    #   - 거부권으로 쓰면 출처 교락이 75.5% → 96.9%로 악화된다(교락을 깨는 샘플만 골라 거부)
    #   - 자기 산출물(CLEAN)로 학습되므로 검증자가 될 수 없다(README의 "Data Flywheel")
    #   - 새 taxonomy에서는 약관규제법 조 개념이 없어 `articles`에 투표할 수조차 없다
    #
    # forward/verify가 같은 모델인데도 이 검증이 성립하는 이유는 **입력이 다르기**
    # 때문이다 — forward는 조항 전문, verify는 근거 문구만 본다. 실측상 744건 중
    # 22.0%가 불일치하고, 그 차이가 공정위 위반 확정 조항에서 선택적으로 나타난다
    # (High: forward 108 → verify 192, 표준계약서는 358 → 349로 거의 불변).
    # 즉 재는 것은 self-consistency가 아니라 grounding이다.
    #
    # **판정에 쓰지 않는 신호도 전부 레코드에 남긴다**(backward_risk, 양쪽 articles) —
    # 다른 판정 규칙을 나중에 비교할 때 LLM을 다시 호출하지 않아도 되게 하기 위함.
    agreed_articles = [a for a in forward_articles if a in set(verify_articles)]
    result["agreed_articles"] = agreed_articles

    if not forward_label or forward_label != verify_label:
        result["status"] = "NOISE"
        result["noise_reason"] = f"label_mismatch: {forward_label} != {verify_label}"
        return result

    # 위험하다고 판정했는데 두 단계가 지목한 조가 하나도 겹치지 않으면, 라벨은 같아도
    # 근거가 다른 것이다. 조 라벨을 학습에 쓰려면 이건 걸러야 한다 — 논문 정의에 없는
    # 추가 조건이므로 사유를 따로 남겨 수율 영향을 측정할 수 있게 한다.
    if forward_label != "Low" and not agreed_articles:
        result["status"] = "NOISE"
        result["noise_reason"] = (
            f"article_mismatch: forward={forward_articles} verify={verify_articles}"
        )
        return result

    result["status"] = "CLEAN"
    result["final_label"] = forward_label
    return result


def build_report(results: list[dict]) -> tuple[dict[str, Any], list[dict], list[dict]]:
    total = len(results)
    clean_n = noise_n = error_n = noise_snippet = noise_mismatch = noise_article = 0
    clean_risk: dict[str, int] = {}
    clean_domain: dict[str, int] = {}
    clean_articles: dict[str, int] = {}   # 조별 CLEAN 건수 — 유형이 실제로 고르게 나오는지 확인용
    clean_no_violation = 0                 # articles가 빈 CLEAN(= 위반 없음 표본) 수
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
            arts = r.get("agreed_articles") or []
            if arts:
                for a in arts:
                    clean_articles[a] = clean_articles.get(a, 0) + 1
            else:
                clean_no_violation += 1
        elif status == "NOISE":
            noise_n += 1
            noise.append(r)
            reason = r.get("noise_reason", "")
            if reason == "snippet_not_found":
                noise_snippet += 1
            elif reason.startswith("label_mismatch"):
                noise_mismatch += 1
            elif reason.startswith("article_mismatch"):
                noise_article += 1
        elif status == "ERROR":
            error_n += 1

    report = {
        "총_입력":           total,
        "CLEAN":             clean_n,
        "NOISE":             noise_n,
        "ERROR":             error_n,
        "CLEAN_비율":        round(clean_n / total, 4) if total else 0,
        "NOISE_비율":        round(noise_n / total, 4) if total else 0,
        # domain_none은 더 이상 탈락 사유가 아니다 — 빈 articles는 "위반 없음"으로
        # CLEAN 후보가 되며 `CLEAN_위반없음`으로 따로 센다.
        "노이즈_원인": {
            "snippet_not_found": noise_snippet,   # evidence_span이 원문에 없음 (E⊂C 실패)
            "label_mismatch":    noise_mismatch,  # L != L' (논문 정의의 탈락 사유)
            "article_mismatch":  noise_article,   # L == L'이나 지목한 조가 안 겹침
        },
        "CLEAN_risk_분포":   clean_risk,
        "CLEAN_domain_분포": clean_domain,
        "CLEAN_조별_분포":   dict(sorted(clean_articles.items(), key=lambda kv: -kv[1])),
        "CLEAN_위반없음":    clean_no_violation,
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
