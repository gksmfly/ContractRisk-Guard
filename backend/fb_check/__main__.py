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
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from openai import OpenAI
from transformers import ElectraTokenizerFast

from backend.fb_check.forward_labeling import run_forward
from backend.fb_check.api_errors import FatalAPIError
from backend.fb_check.backward_grounding import (MATCHER_VERSION, load_model,
                                                 snippet_exists, predict)
from backend.fb_check.consistency_verification import run_verify
from backend.model.electra import DualHeadElectra
from backend.utils import load_jsonl, load_logger, save_json, save_jsonl, PROJECT_ROOT

logger = load_logger("fb_check.log")

_SEGMENT_EVERY = 500
# 연속 이만큼 실패하면 실행을 끊는다. 50건이면 정상 구간에서 우연히 걸릴 일이 없고
# (실측 ERROR율 0%대), 사고가 나면 8.4시간이 아니라 8분 안에 멈춘다.
_ERROR_STREAK_ABORT = 50
# 구간 판정 문턱. 스트릭만으로는 **간헐적** 붕괴를 못 잡는다 — 40%가 실패하는데 50연속은
# 안 나는 상황이면 스트릭이 계속 리셋되면서 몇 시간을 돈다.
_SEGMENT_ERROR_ABORT = 0.20    # 구간 ERROR 20% 초과 → 중단

# CLEAN 붕괴 문턱은 **관측에서 도출한다** — 상수로 박지 않는다.
#
# 처음엔 "첫 구간 × 0.50"으로 잡았다. 08-23 데이터로 재생해보니 문턱 35.7% vs 실제 붕괴
# 구간 35.8%로 **0.1%p 차이로 통과**했다. 0.50이 아무 근거 없는 숫자라 그렇다.
#
# [측정] 08-23 정상 구간 4개(ERROR<1%)의 CLEAN 비율:
#     [1~500] 71.4% · [501~1000] 76.6% · [1001~1500] 72.2% · [1501~2000] 71.0%
#     μ = 72.80%   관측 σ = 2.58%p   이항 표집 SE(n=500, p=.73) = 1.99%p
#     μ − 4σ = 62.5%  →  붕괴 구간 35.8%를 잡고, 정상 최저 71.0%와 8.5%p 여유
#
# σ는 max(관측, 이항 SE)를 쓴다. 구간들이 우연히 촘촘하게 모이면 관측 σ가 0에 가까워져
# 문턱이 정상 변동에도 걸릴 만큼 조여지는데, 이항 SE가 그 하한이다 — 완벽히 안정적인
# 과정이라도 500건 표집만으로 이만큼은 흔들린다.
_SEGMENT_CLEAN_SIGMA = 4.0     # μ − kσ 의 k
_SEGMENT_BASELINE_MIN = 3      # 이만큼 정상 구간이 쌓여야 CLEAN 게이트를 켠다
_SEGMENT_BASELINE_MAX = 6      # 이후 구간은 기준에 넣지 않는다 — 서서히 나빠지면 기준까지 끌려간다   # 구간 지표 주기 — 8.4시간 실행에서 중간 상태를 알기 위해

SEED_PATH = Path(os.environ.get("SEED_PATH",  str(PROJECT_ROOT / "data/labels/seed_labeled.jsonl")))
MODEL_DIR = Path(os.environ.get("MODEL_DIR",  str(PROJECT_ROOT / "models/v1")))
OUT_DIR   = Path(os.environ.get("FB_OUT_DIR", str(PROJECT_ROOT / "data/fb_check")))


def stratified_limit(records: list[dict], limit: int, seed: int = 42) -> list[dict]:
    """스모크용 표본 — **단일조항 문서와 다조항 문서를 반반** 뽑는다.

    앞선 라벨링 파일럿은 `조항이 1개인 사건`만 봤다. 그건 파서가 깨끗하게 동작한 구간이라,
    거기서 나온 F1이 전량에서 유지된다는 보장이 없다. 다조항 사건에는 좌우 대조표가
    뒤엉킨 텍스트가 섞여 있다:

        제9조(이용계약의 중지 및 해지) 제9조(이용계약의 중지 및 해지) ② 회사는 회원이
        다음 각 호의 어느 하나 ② 회사는 회원이 다음 각 호의 하나에 해 에 해당하는...

    두 구간을 반반 뽑아 **따로 채점해야** "전량을 돌려도 되는가"에 답할 수 있다.
    문서 단위로 뽑으므로(조항 단위 아님) 같은 사건의 조항이 쪼개지지 않는다 —
    다조항 구간을 사건 단위 union으로 채점하려면 사건이 통째로 들어와야 한다.
    """
    by_doc: dict[str, list[dict]] = {}
    for r in records:
        parts = str(r.get("chunk_id", "")).split(":")
        doc = ":".join(parts[:2]) if len(parts) >= 3 else str(r.get("chunk_id"))
        by_doc.setdefault(doc, []).append(r)

    single = sorted(d for d, rs in by_doc.items() if len(rs) == 1)
    multi  = sorted(d for d, rs in by_doc.items() if len(rs) > 1)
    rng = random.Random(seed)
    rng.shuffle(single); rng.shuffle(multi)

    picked: list[dict] = []
    half = limit // 2
    for docs, want, label in ((single, half, "단일조항"), (multi, limit - half, "다조항")):
        got = 0
        for d in docs:
            if got >= want:
                break
            picked.extend(by_doc[d])
            got += len(by_doc[d])
        logger.info(f"  층화 표본 [{label}] 문서에서 조항 {got}건")
    return picked[:limit] if len(picked) > limit else picked


def run_fb_check(
    record: dict,
    client: OpenAI,
    model: DualHeadElectra | None = None,
    tokenizer: ElectraTokenizerFast | None = None,
    device: torch.device | None = None,
    llm_model: str | None = None,
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
    forward = run_forward(client, clause_text, **({'model': llm_model} if llm_model else {}))
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
    # `snippet_exists`(E⊂C)는 순수 문자열 검사라 모델이 필요 없다 — 논문이 정의한
    # Backward Grounding의 역할이 바로 이 인덱스 검증이다.
    span_exists = snippet_exists(clause_text, evidence_span)
    result["snippet_exists"] = span_exists
    result["snippet_matcher"] = MATCHER_VERSION

    # KoELECTRA 독립 예측은 **판정에 쓰이지 않는다**(2-way = L == L'). 기록용일 뿐인데
    # 예전에는 조항마다 무조건 돌렸다 — 전량 라벨링(2,400건+)이 폐기 대상 체크포인트
    # `models/v4`에 묶이고, 헤드 구조를 바꾸면 라벨링이 먼저 깨지는 구조였다.
    # 지금은 `--record-backward`로 명시할 때만 로드·실행한다.
    if model is not None:
        backward_domain, backward_risk = predict(clause_text, model, tokenizer, device)
        result.update({"backward_domain": backward_domain, "backward_risk": backward_risk})

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

    verify = run_verify(client, verify_input, mode=verify_mode,
                        **({'model': llm_model} if llm_model else {}))
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


# 재처리 판정에 쓰는 필드. **forward만 보면 절반짜리다** — `--model`이 verify까지 덮는지는
# 별개 문제이고, .env의 VERIFY_MODEL이 다르면 판정(L == L')의 절반이 다른 모델 산출이
# 되면서도 가드를 통과한다. 오늘 하루 쫓은 것과 같은 종류의 조용한 오염이 verify 쪽으로
# 재발하는 경로다. 네 필드를 다 본다.
#
# `verify_*`가 None인 레코드는 검사에서 제외한다 — `snippet_not_found`(E⊂C 실패)로
# verify 호출 전에 탈락한 정상 케이스다.
#
# ⚠️ 남은 구멍: None이 "verify를 안 함"인지 "옛 포맷이라 필드 자체가 없음"인지 구분하지
#    못한다. 필드를 도입하기 전(2026-08-21 이전) 산출물이 섞이면 조용히 통과한다.
#    지금 데이터에는 옛 포맷이 없어 문제되지 않지만, 옛 결과 파일을 다시 쓰게 되면
#    `noise_reason == "snippet_not_found"`인 경우만 면제하도록 좁혀야 한다.
_STALE_FIELDS = ("forward_model", "forward_prompt", "verify_model", "verify_prompt")


# 레코드당 예상 비용(USD). 실측 토큰 기준 — system 1,459 + few-shot 1,048 + 조항 평균 173,
# forward/verify 2회, 프롬프트 캐싱(1,024토큰 이상 접두사 자동 50% 할인) 반영.
# 정확한 청구액이 아니라 **자릿수를 틀리지 않기 위한** 값이다.
_USD_PER_RECORD = 0.0075


def log_provenance() -> None:
    """**누가·어디서·어떤 명령으로 이 실행을 시작했는지** 로그 첫 줄에 남긴다.

    08-25 05:44에 출처 불명의 `--dry-scope` 실행이 로그에 찍혔다. API 호출이 0이라
    무해했지만, **비용이 나가는 파이프라인에서 출처를 모르는 실행이 있다는 것 자체가
    문제다.** 그리고 오늘 하루가 "누가 뭘 어떤 설정으로 돌렸는지 몰라서" 생긴 문제의
    연속이었다 — mini 산출 318건 혼입, .env와 실험 경로의 모델 불일치, 크레딧 소진 시점
    추적. 전부 실행 메타데이터가 없어서 사후에 추론해야 했다.

    argv를 통째로 남기는 게 핵심이다. 범위를 바꾸는 플래그(`--limit`/`--sample`/
    `--redo-reason`/`--only-redo`)가 무엇이었는지가 그대로 보인다.
    """
    import getpass
    import socket
    import sys
    try:
        ppid = os.getppid()
        parent = Path(f"/proc/{ppid}/cmdline").read_text(errors="replace").replace("\0", " ").strip()
    except OSError:
        parent = "?"
    logger.info(f"  실행 주체: {getpass.getuser()}@{socket.gethostname()} "
                f"pid={os.getpid()} ppid={os.getppid()}")
    logger.info(f"  명령: {' '.join(sys.argv)}")
    logger.info(f"  부모: {parent[:120] or '(detached)'}")
    logger.info(f"  작업 디렉터리: {os.getcwd()}")


def log_scope(pending: list[dict], records: list[dict], done: int, redo: frozenset[str],
              only_redo: bool) -> None:
    """**돈을 쓰기 전에 무엇을 얼마나 건드리는지 찍는다.**

    08-24에 `--redo-reason snippet_not_found`가 의도한 146건이 아니라 2,277건을 대상으로
    잡았다. 버그가 아니라 **옳은 기능 둘이 합쳐진 결과**다:

        _load_checkpoint    ERROR 레코드를 재시도 대상으로 둔다   ← 옳다(일시적 실패의 영구화 방지)
        --redo-reason       해당 사유의 NOISE를 재시도 대상으로 둔다 ← 옳다
        합쳐지면            미처리 2,131건 + 재처리 146건 = 15배 초과

    단위 테스트로는 안 잡힌다. 두 기능 다 명세대로 동작하기 때문이다. **실행 직전에
    건수를 세야만 보인다.** 그래서 세는 것을 코드에 넣는다.

    `--limit` · `--sample` · `--redo-reason` · `--only-redo`는 전부 "무엇을 얼마나
    건드리는가"를 바꾸는 플래그다. 건수가 먼저 보이면 이런 조합 사고가 실행 전에 걸린다.
    """
    logger.info("  ----- 실행 범위 -----")
    logger.info(f"    입력            {len(records):>6}건")
    logger.info(f"    체크포인트 건너뜀 {done:>6}건")
    if redo:
        logger.info(f"    재처리 사유      {sorted(redo)}" + ("  (--only-redo)" if only_redo else ""))
    logger.info(f"    처리 대상        {len(pending):>6}건")
    by_src = Counter(r.get("source", "?") for r in pending)
    if by_src:
        logger.info(f"    출처            {dict(by_src)}")
    logger.info(f"    예상 비용        ${len(pending) * _USD_PER_RECORD:>6.2f}  "
                f"(API 호출 약 {len(pending) * 2:,}회) — 추정치")
    if redo and not only_redo:
        logger.warning("    ★ --redo-reason만 줬다. 미처리(ERROR) 레코드도 함께 돈다 — "
                       "재처리 대상만 원하면 --only-redo를 붙일 것")


def _redo_ids(results_path: Path, redo_reasons: frozenset[str]) -> set[str]:
    """재처리 대상 chunk_id만 모은다 — `--only-redo`용.

    `--redo-reason`만으로는 부족하다. `_load_checkpoint`가 ERROR 레코드를 **재시도 대상**
    으로 빼두기 때문에(일시적 실패의 영구화를 막으려고 그렇게 만들었다) 미처리분이 전부
    pending에 함께 들어온다.

    08-24 실측: `--redo-reason snippet_not_found`가 의도한 146건이 아니라
    **2,277건(= ERROR 2,131 + 146)** 을 처리 대상으로 잡았다. $1.09짜리 작업이 $17이 된다.
    규칙 하나를 고치고 그 부분집합만 다시 태우려는 것이므로, 미처리분과는 분리돼야 한다.
    """
    ids: set[str] = set()
    if not results_path.exists() or not redo_reasons:
        return ids
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "NOISE" and \
                    str(row.get("noise_reason", "")).split(":")[0] in redo_reasons:
                ids.add(row.get("chunk_id"))
    return ids


def _clean_floor(baseline: list[float]) -> float | None:
    """정상 구간들의 CLEAN 비율에서 붕괴 문턱 μ − kσ 를 도출한다.

    σ = max(관측 표준편차, 이항 표집 SE). 후자가 하한인 이유는 `_SEGMENT_CLEAN_SIGMA`
    주석 참고 — 구간들이 우연히 촘촘하면 관측 σ가 0에 가까워져 문턱이 정상 변동에도
    걸릴 만큼 조여진다.
    """
    if len(baseline) < _SEGMENT_BASELINE_MIN:
        return None                       # 표본이 모자라면 게이트를 켜지 않는다
    mu = statistics.mean(baseline)
    binom_se = (mu * (1 - mu) / _SEGMENT_EVERY) ** 0.5
    sigma = max(statistics.stdev(baseline), binom_se)
    return max(0.0, mu - _SEGMENT_CLEAN_SIGMA * sigma)


def _load_checkpoint(results_path: Path, expect: dict[str, str] | None = None,
                     redo_reasons: frozenset[str] = frozenset()) -> set[str]:
    """이미 처리된 chunk_id 집합. **ERROR로 끝난 건은 제외해 재시도 대상으로 둔다.**

    두 가지를 견뎌야 한다 — 전량 실행이 12시간짜리라 중간에 죽는 것이 예외가 아니라 정상이다:

    1. **잘린 마지막 줄.** 프로세스가 쓰는 도중 죽으면 마지막 줄이 불완전한 JSON으로 남는다.
       예전에는 `json.loads`가 그대로 터져서 **재개 자체가 불가능**했다(10시간 진행분을
       읽지 못하고 죽는다). 깨진 줄은 건너뛰고 경고만 남긴다 — 그 한 건은 어차피 미처리라
       `pending`에 다시 들어온다.
    2. **일시적 실패의 영구화.** ERROR 레코드도 chunk_id가 있어 "처리됨"으로 잡혔다.
       429·네트워크 오류로 실패한 건이 재개해도 영원히 재시도되지 않는다.
    """
    if not results_path.exists():
        return set()
    done: set[str] = set()
    broken = stale = 0
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                broken += 1        # 중단 시점의 잘린 줄 — 재처리 대상으로 두고 넘어간다
                continue
            if row.get("status") == "ERROR":
                continue           # 재시도 대상
            # 판정 **규칙**이 고쳐졌을 때 해당 사유로 버려진 건만 다시 태운다.
            # 실제 사례: `snippet_exists`가 공백을 압축만 하고 제거하지 않아 PDF 줄바꿈이
            # 단어를 쪼갠 건(`영업정 지`)을 전부 놓쳤다 — E⊂C 실패율 12.5% 중 9.8%p가
            # 매칭 버그였다. 그 건들은 게이트에서 조기 반환돼 verify_*가 비어 있으므로
            # 오프라인 재계산으로는 못 살리고 verify를 다시 불러야 한다.
            if redo_reasons and row.get("status") == "NOISE" and \
                    str(row.get("noise_reason", "")).split(":")[0] in redo_reasons:
                # 매처 버전이 지금과 같으면 다시 태워도 **같은 결과**다. 규칙이 바뀐 뒤에만
                # 재처리한다 — 그러지 않으면 `--redo-reason`을 줄 때마다 "현재 매처 기준으로
                # 진짜 불일치"인 건들(08-25 기준 13건)을 영원히 다시 돈다.
                if (str(row.get("noise_reason", "")).split(":")[0] == "snippet_not_found"
                        and row.get("snippet_matcher") == MATCHER_VERSION):
                    done.add(row.get("chunk_id"))
                    continue
                continue
            # **모델·프롬프트가 다르면 재처리한다.** 예전에는 chunk_id만 보고 건너뛰어서,
            # 잘못된 모델로 만든 레코드가 그대로 학습 데이터에 섞였다 — 조용한 오염이라
            # 나중에 "왜 일부만 이상하지"를 추적하게 된다. 레코드에 forward_model /
            # forward_prompt를 남기고 있으므로 그 두 필드로 판정하면 구조적으로 막힌다.
            if expect and any(
                row.get(f) not in (None, expect[f]) for f in _STALE_FIELDS if f in expect
            ):
                stale += 1
                continue
            done.add(row.get("chunk_id", ""))
    if broken:
        logger.warning(f"  체크포인트에서 깨진 줄 {broken}개 건너뜀(중단 시점의 잘린 기록) — 해당 건은 재처리된다")
    if stale:
        logger.warning(f"  모델·프롬프트가 다른 레코드 {stale}건 → **재처리 대상** (기대: {expect})")
    return done


def _save_atomic(rows: list[dict], path: Path) -> None:
    """최종 산출물을 **원자적으로** 쓴다 — 임시 파일에 다 쓴 뒤 한 번에 갈아 끼운다.

    `clean.jsonl`/`noise.jsonl`은 학습 데이터다. 쓰는 도중에 죽으면 반쪽짜리 파일이
    남고, `train_article.py`가 `load_jsonl`로 읽다가 그때서야 "라벨링이 안 끝났다"를
    알게 된다. `os.replace`는 같은 파일시스템에서 원자적이라, 크래시 후 남는 것은
    **완전한 이전 파일이거나 완전한 새 파일**이다. 반쪽은 절대 안 나온다.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_jsonl(rows, tmp)
    os.replace(tmp, path)


def _save_json_atomic(data: Any, path: Path) -> None:
    """리포트도 같은 이유로 원자적으로 쓴다."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_json(data, tmp)
    os.replace(tmp, path)


def _read_results(path: Path) -> list[dict]:
    """결과 파일을 **깨진 줄을 견디며** 읽는다.

    ## 파일 종류에 따라 규칙이 다르다

        체크포인트(fb_check_results.jsonl)  append-only · 깨진 줄 **관용**
                                            중단이 정상 동작이라 잘린 줄은 흔적일 뿐이다
        최종 산출물(clean/noise.jsonl)      원자적 쓰기 · 깨진 줄 **불관용**
                                            반쪽이면 잘못된 것이므로 시끄럽게 죽어야 한다

    그래서 `backend.utils.load_jsonl`을 전역으로 관용하게 만들지 않았다 —
    `data/processed/*.jsonl`의 잘린 줄은 진짜 손상이라 조용히 삼키면 안 된다.


    `backend.utils.load_jsonl`은 `json.loads`를 그대로 부르므로 중단 시점의 잘린 줄에서
    터진다. `_load_checkpoint`만 고치고 이 경로를 안 고쳤다가 실제로 크래시했다 —
    resume은 정상 복구했는데 `--save-every` 시점의 중간 저장에서 같은 파일을 다시 읽다가
    죽었다. 결과 파일을 읽는 곳은 **전부** 이 함수를 거쳐야 한다.
    """
    rows, broken = [], 0
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                broken += 1
    if broken:
        logger.warning(f"  결과 파일에서 깨진 줄 {broken}개 건너뜀(중단 시점의 잘린 기록)")
    return rows


def _dedup_last(rows: list[dict]) -> list[dict]:
    """chunk_id마다 **마지막 기록만** 남긴다.

    ERROR 재시도로 같은 chunk_id가 두 번 이상 append될 수 있는데, 그대로 집계하면
    CLEAN/NOISE가 중복으로 세어진다. 나중 기록이 재시도 결과이므로 그쪽을 취한다.
    """
    by: dict[str, dict] = {}
    for r in rows:
        by[r.get("chunk_id", "")] = r
    return list(by.values())


def _append_result(results_path: Path, result: dict) -> None:
    """결과 1건을 파일에 즉시 추가한다."""
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FB-Check: Forward-Backward Consistency Check")
    parser.add_argument("--input",     default=str(SEED_PATH), help="입력 JSONL 경로")
    parser.add_argument("--sample",    type=int, default=0,    help="샘플 수 (0=전체) — 앞에서부터 자른다")
    parser.add_argument("--limit",     type=int, default=0,
                        help="스모크용 층화 표본 수(0=전체). 단일조항/다조항 문서를 반반 뽑는다. "
                             "**전량과 같은 코드 경로**로 돌므로 여기서 만든 결과는 버려지지 않고 "
                             "이어서 전량을 돌리면 resume이 그 뒤부터 이어받는다")
    parser.add_argument("--gpu",       type=int, default=0,    help="GPU 인덱스")
    parser.add_argument("--save-every",type=int, default=50,   help="clean/noise 중간 저장 주기")
    parser.add_argument("--model", default=None,
                        help="forward/verify에 쓸 모델. 기본은 .env의 FORWARD_MODEL/VERIFY_MODEL. "
                             "**명시하는 편이 안전하다** — 스모크 300건이 .env의 gpt-4o-mini로 돌아 "
                             "빈 배열 63%%가 나온 적이 있다(mini는 9유형 분류에 부적합 확정)")
    parser.add_argument("--dry-scope", action="store_true",
                        help="처리 대상 건수·출처·예상 비용만 찍고 종료한다(API 호출 없음). "
                             "범위를 바꾸는 플래그(--limit/--sample/--redo-reason)를 쓸 때 "
                             "**먼저 이걸로 확인하고** 실행할 것")
    parser.add_argument("--only-redo", action="store_true",
                        help="--redo-reason 대상만 처리하고 미처리분은 건드리지 않는다. "
                             "이걸 안 주면 ERROR로 남은 미처리분까지 함께 돈다 "
                             "(_redo_ids 참고 — 146건 작업이 2,277건이 된다)")
    parser.add_argument("--redo-reason", default="", metavar="사유[,사유…]",
                        help="해당 noise_reason으로 버려진 건만 재처리한다 "
                             "(판정 규칙을 고친 뒤 그 부분집합만 다시 태울 때). "
                             "예: --redo-reason snippet_not_found")
    parser.add_argument("--record-backward", action="store_true",
                        help="KoELECTRA 독립 예측을 결과에 기록한다(판정에는 안 쓰임). "
                             "모델을 로드하므로 전량 실행 시 느려지고 체크포인트에 의존하게 된다")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"========== FB-Check 시작 | device={device} ==========")
    log_provenance()

    records = load_jsonl(Path(args.input))
    if args.sample > 0:
        records = records[:args.sample]
    if args.limit > 0:
        records = stratified_limit(records, args.limit)
    else:
        # **전량도 섞는다.** seed_labeled.jsonl은 FTC 전부 → 표준계약서 전부 순으로 쓰여 있어,
        # 파일 순서대로 돌리면 두 가지가 깨진다:
        #
        #   ① 중간에 죽으면 음성 표본이 없다 — 5시간 지점에서 멈추면 FTC만 끝나 있고
        #      표준계약서(빈 배열 표본)가 거의 없어 그 상태로는 학습을 시작할 수 없다.
        #      섞여 있으면 절반만 돼도 양쪽이 균형 있게 들어와 바로 학습할 수 있다.
        #   ② 진행 중 지표가 해석 불가능하다 — CLEAN 비율이 블록 경계에서 급변하는데,
        #      품질 저하인지 코퍼스가 바뀐 건지 구분이 안 된다.
        #
        # resume은 안 깨진다 — `chunk_id not in done_ids`로 거르므로 순서는 정확성에 무관하고,
        # 스모크로 끝낸 300건도 그대로 건너뛴다. 시드 고정이라 재현성도 유지된다.
        random.Random(42).shuffle(records)
        logger.info("  전량 실행 — 레코드를 섞는다(seed=42). 중간에 멈춰도 부분 결과가 대표성을 갖는다")
    logger.info(f"  입력: {len(records)}건 ({args.input})")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model = tokenizer = None
    if args.record_backward:
        model, tokenizer = load_model(MODEL_DIR, device)
        logger.info(f"  KoELECTRA 로드 완료(기록 전용, 판정에는 미사용): {MODEL_DIR}")
    else:
        logger.info("  KoELECTRA 미로드 — 판정은 L == L'(논문 정의)이라 필요 없다. "
                    "기록이 필요하면 --record-backward")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "fb_check_results.jsonl"

    # 체크포인트: 이미 처리된 건 건너뜀
    from backend.fb_check.forward_labeling import FORWARD_MODEL, PROMPT_VERSION as FWD_PROMPT
    from backend.fb_check.consistency_verification import VERIFY_MODEL, PROMPT_VERSION as VER_PROMPT
    expect = {
        "forward_model":  args.model or FORWARD_MODEL,
        "forward_prompt": FWD_PROMPT,
        "verify_model":   args.model or VERIFY_MODEL,
        "verify_prompt":  VER_PROMPT,
    }
    # 두 단계 모델을 **각각** 찍는다 — 하나만 찍으면 forward/verify가 갈린 걸 못 본다.
    logger.info(f"  ★ forward: {expect['forward_model']} ({expect['forward_prompt']})")
    logger.info(f"  ★ verify : {expect['verify_model']} ({expect['verify_prompt']})")
    if expect["forward_model"] != expect["verify_model"]:
        logger.warning("  ★ forward와 verify의 모델이 다르다 — 의도한 것인지 확인할 것")
    redo = frozenset(x.strip() for x in args.redo_reason.split(",") if x.strip())
    if redo:
        logger.info(f"  ★ 재처리 대상 사유: {sorted(redo)} — 해당 NOISE 건을 다시 태운다")
    done_ids = _load_checkpoint(results_path, expect=expect, redo_reasons=redo)
    if done_ids:
        logger.info(f"  체크포인트 복원: {len(done_ids)}건 건너뜀")
    pending = [r for r in records if r["chunk_id"] not in done_ids]
    if args.only_redo:
        if not redo:
            raise SystemExit("--only-redo는 --redo-reason과 함께 써야 한다")
        targets = _redo_ids(results_path, redo)
        pending = [r for r in pending if r["chunk_id"] in targets]
        logger.info(f"  ★ --only-redo: 재처리 대상만 처리한다 (미처리분은 건드리지 않는다)")
    log_scope(pending, records, len(done_ids), redo, args.only_redo)
    if args.dry_scope:
        logger.info("  --dry-scope: 범위만 확인하고 종료한다 (API 호출 없음)")
        return

    clean_n = noise_n = 0
    window: list[dict] = []      # 구간 지표용 최근 결과
    consecutive_errors = 0
    clean_baseline: list[float] = []       # 정상 구간들의 CLEAN 비율 — 붕괴 문턱을 여기서 도출

    for i, record in enumerate(pending, 1):
        try:
            result = run_fb_check(record, client, model, tokenizer, device, llm_model=args.model)
        except FatalAPIError as e:
            # 크레딧 소진·키 오류 — 남은 건을 다 태워봐야 전부 ERROR다. 즉시 끊는다.
            logger.error(f"  ■ 치명적 API 오류로 중단한다 ({i - 1}건 처리 완료, {len(pending) - i + 1}건 남음)")
            logger.error(f"    {e}")
            logger.error("    조치 후 같은 명령으로 재개하면 체크포인트가 이어받는다")
            break
        _append_result(results_path, result)          # 즉시 디스크에 기록
        window.append(result)

        if result.get("status") == "CLEAN":
            clean_n += 1
        elif result.get("status") == "NOISE":
            noise_n += 1

        # 연속 실패 차단기. `FatalAPIError`가 아닌 오류(네트워크 단절, 모델 응답 붕괴 등)로
        # 조용히 전부 ERROR가 되는 경우를 잡는다. 08-23 실행에서 구간 지표가
        # `ERROR 500/500`을 세 번 찍었는데도 5시간을 더 돌았다 — 지표는 **보고**만 하고
        # **멈추지는** 않았던 것이 설계 구멍이었다.
        if result.get("status") == "ERROR":
            consecutive_errors += 1
            if consecutive_errors >= _ERROR_STREAK_ABORT:
                logger.error(f"  ■ 연속 {consecutive_errors}건 실패로 중단한다 "
                             f"({i}건 처리, {len(pending) - i}건 남음) — 마지막 오류: "
                             f"{result.get('error')}")
                logger.error("    원인을 고친 뒤 같은 명령으로 재개하면 체크포인트가 이어받는다")
                break
        else:
            consecutive_errors = 0

        if i % 10 == 0 or i == len(pending):
            logger.info(f"  [{i}/{len(pending)}] CLEAN={clean_n} NOISE={noise_n}")

        # 구간 지표 — 8.4시간 무인 실행에서 품질이 중간에 무너져도 끝나야 아는 상황을 막는다.
        # 셔플한 뒤라야 의미가 있다(안 섞으면 블록 경계에서 급변해 해석이 안 된다).
        #
        # **그리고 판정한다.** 08-23 실행에서 이 지표는 붕괴를 정확히 탐지했지만
        # (`[2001~2500] CLEAN 36% ERROR 263` → `[2501~3000] CLEAN 0% ERROR 500`)
        # 보고만 하고 멈추지 않아 5시간을 헛돌았다. 스모크의 GO/NO-GO를 판정하고
        # 멈추게 만든 것과 같은 원리를 여기에도 적용한다.
        if i % _SEGMENT_EVERY == 0:
            seg = window[-_SEGMENT_EVERY:]
            n_seg = len(seg)
            c = sum(1 for r in seg if r.get("status") == "CLEAN")
            e = sum(1 for r in seg if r.get("status") == "ERROR")
            empty = sum(1 for r in seg
                        if r.get("status") == "CLEAN" and not (r.get("agreed_articles") or []))
            by_src = Counter(r.get("source", "?") for r in seg)
            clean_ratio, error_ratio = c / n_seg, e / n_seg
            logger.info(f"  ── 구간 지표 [{i - _SEGMENT_EVERY + 1}~{i}] "
                        f"CLEAN {clean_ratio * 100:.0f}% · 빈배열(CLEAN 중) {empty / max(c, 1) * 100:.0f}% · "
                        f"ERROR {e} · 출처 {dict(by_src)}")

            if error_ratio > _SEGMENT_ERROR_ABORT:
                logger.error(f"  ■ 구간 ERROR {error_ratio * 100:.0f}%가 문턱 "
                             f"{_SEGMENT_ERROR_ABORT * 100:.0f}%를 넘어 중단한다 "
                             f"({i}건 처리, {len(pending) - i}건 남음)")
                logger.error("    원인을 고친 뒤 같은 명령으로 재개하면 체크포인트가 이어받는다")
                break

            # ERROR가 아니어도 붕괴할 수 있다 — 모델 응답 형식이 바뀌거나 프롬프트가
            # 무너지면 status는 멀쩡한데 CLEAN만 사라진다. 08-23에도 ERROR보다
            # CLEAN 붕괴(71%→36%)가 한 구간 **먼저** 보였다.
            floor = _clean_floor(clean_baseline)
            if floor is not None and clean_ratio < floor:
                logger.error(f"  ■ 구간 CLEAN {clean_ratio * 100:.1f}%가 문턱 {floor * 100:.1f}% "
                             f"(정상 {len(clean_baseline)}구간 μ−{_SEGMENT_CLEAN_SIGMA:g}σ) 미만이라 "
                             f"중단한다 ({i}건 처리, {len(pending) - i}건 남음)")
                logger.error("    셔플했으므로 구간별 CLEAN은 안정적이어야 한다 — 실제 이상이다")
                break
            if len(clean_baseline) < _SEGMENT_BASELINE_MAX:
                clean_baseline.append(clean_ratio)
                nf = _clean_floor(clean_baseline)
                logger.info(f"     기준 구간 {len(clean_baseline)}개 · CLEAN 문턱 "
                            + (f"{nf * 100:.1f}%" if nf is not None
                               else f"미정({_SEGMENT_BASELINE_MIN}구간 필요)"))

        # 주기적으로 clean/noise 파일 갱신
        if i % args.save_every == 0:
            all_results = _dedup_last(_read_results(results_path))
            report, clean, noise = build_report(all_results)
            _save_atomic(clean, OUT_DIR / "clean.jsonl")
            _save_atomic(noise, OUT_DIR / "noise.jsonl")
            _save_json_atomic(report, OUT_DIR / "fb_check_report.json")
            logger.info(f"  중간 저장 완료 ({i}건)")

    # 최종 저장
    all_results = _dedup_last(_read_results(results_path))
    report, clean, noise = build_report(all_results)
    _save_atomic(clean, OUT_DIR / "clean.jsonl")
    _save_atomic(noise, OUT_DIR / "noise.jsonl")
    _save_json_atomic(report, OUT_DIR / "fb_check_report.json")

    logger.info(f"  결과: CLEAN={report['CLEAN']} / NOISE={report['NOISE']} / ERROR={report['ERROR']}")
    logger.info(f"  CLEAN 비율: {report['CLEAN_비율']:.1%}")
    logger.info(f"  노이즈 원인: {report['노이즈_원인']}")
    logger.info("========== FB-Check 완료 ==========")


if __name__ == "__main__":
    main()
