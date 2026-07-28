# backend/scripts/rebuild_ftc_ground_truth.py
"""
Low/Medium 정답 데이터가 없는 문제를 수작업 라벨링 없이 해결하기 위한 후보 생성 스크립트.

seed.py는 FTC 시정조치 케이스 전체를 무조건 risk_level=High로 하드코딩한다.

1차 시도(대표조치유형: 시정명령→High, 시정권고→Medium)는 스팟체크에서 기각됐다 —
시정권고 케이스가 오히려 위반유형·근거법령 수가 더 많아(평균 2.2 vs 1.0),
행정 트랙 구분이 조항 심각도와 무관한 것으로 확인됨.

2차 시도(채택): 케이스별 위반_유형 개수를 심각도 프록시로 쓴다 — 단일 조문
위반(예: 약관규제법 제9조 하나만) vs 복합 위반(여러 조문 동시 위반, 더 광범위한
불공정성)은 스팟체크로 실제 조항 성격 차이가 확인됨.
  - 위반유형 1개  → Medium 후보 (제한적 단일 위반)
  - 위반유형 2개+ → High 후보 (복합·광범위 위반)
  - 위반유형 0개 / 재결기각(이의제기로 뒤집힘) / 사건병합 / 고발 → 제외(신호 없음 또는 확정 위반 아님)

**3차 보정(품질 검증, 이번 라운드)**: High/Medium 10건씩 스팟체크한 결과 진짜 문제가
발견됐다 — FTC 케이스 하나에 조항이 여러 개(`조항_원문`) 추출되는데, "이 케이스는
확정 위반"이라는 케이스 단위 라벨을 추출된 모든 조항 조각에 그대로 적용하고 있어서,
실제 위반 조항이 아닌 옆 조항(심지어 소비자에게 유리한 조항, 또는 공정위 명령문
자체)까지 같은 라벨을 받는 문제가 있었다(전체의 4.5%가 명령문 오염, 일부는 명백한
반대 라벨). 대응:
  - 조항이 1개뿐인 케이스(273건)는 라벨-텍스트 매칭이 모호할 여지가 없어 그대로 신뢰
  - 조항이 2개 이상인 케이스(463건)에서 나온 조항은 GPT-4o-mini로 "이 텍스트가 이
    사건의 위반_유형이 설명하는 실제 위반 근거 조항이 맞는가"를 검증 — 이건 위험도를
    다시 판단하는 게 아니라 "라벨이 붙은 텍스트가 맞는 텍스트인가"를 확인하는 사실
    매칭 작업이라 순환논리가 없다

표준계약서(공정위 공식 템플릿, "표준약관" 카테고리 포함 6종 전체)는 정부가
"공정하다"고 발행한 문서라는 점 자체를 Low 근거로 삼는다(정규식 위험도 판정은
쓰지 않음). 원본의 상당수가 계약 조문이 아닌 크롤링 노이즈(보도자료 등)라서
조문 패턴으로 시작하지 않는 문서는 제외한다. **Low도 High/Medium과 동일한 강도로
검증한다** — 스팟체크에서 "택배란 ...을 말한다" 같은 단순 정의 조항이 키워드만
우연히 걸려 Low로 들어간 사례가 확인됨. GPT-4o-mini로 "이 텍스트가 실제로 해지·
책임제한에 관한 실질적 계약 조항인가(단순 정의·행정 절차가 아닌가)"를 검증한다.

이 스크립트의 출력은 seed_labeled.jsonl(학습 데이터)에 영향을 주지 않는다 —
평가 전용 후보 풀이다.

실행: python -m backend.scripts.rebuild_ftc_ground_truth
"""

import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI

from backend.labeling.seed import classify_domain, split_articles
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json, save_jsonl

logger = load_logger("rebuild_ftc_ground_truth.log")

FTC_PATH     = Path(os.environ.get("FTC_PATH",     str(PROJECT_ROOT / "data/raw/ftc_cases/ftc_cases_parsed.json")))
CONTRACT_DIR = Path(os.environ.get("CONTRACT_DIR", str(PROJECT_ROOT / "data/raw/contract")))
CLEAN_PATH   = Path(os.environ.get("CLEAN_PATH",   str(PROJECT_ROOT / "data/fb_check/clean.jsonl")))
OUT_DIR      = Path(os.environ.get("EVAL_CANDIDATE_DIR", str(PROJECT_ROOT / "data/eval/candidates")))
EVAL_DIR     = Path(os.environ.get("EVAL_DIR",      str(PROJECT_ROOT / "data/eval")))
VERIFY_CACHE_DIR = EVAL_DIR / "verify_cache"

VERIFY_MODEL = os.environ.get("GT_VERIFY_MODEL", "gpt-4o-mini")

_ARTICLE_START_PAT = re.compile(r"^\s*제\s*\d+\s*조")
_CONFIRMED_ACTION_TYPES = {"시정명령", "시정권고", "시정요청(약관)"}  # 확정 위반(뒤집히지 않음) — 행정 트랙 무관

# 조항 대신 공정위 명령문 자체가 섞인 경우(4.5% 오염 확인됨) — GPT 호출 전에 기계적으로 먼저 제외
_PREAMBLE_CONTAMINATION_PAT = ["시정권고를 받은 날부터", "삭제 또는 수정하", "시정명령을 받은 날부터"]


def _risk_from_violation_count(n_violations: int) -> str | None:
    if n_violations == 0:
        return None
    return "Medium" if n_violations == 1 else "High"


def _is_preamble_contaminated(text: str) -> bool:
    return any(p in text for p in _PREAMBLE_CONTAMINATION_PAT)


# ---------------------------------------------------------------------------
# GPT 검증 (캐시 사용 — 중단돼도 재실행 시 이어서 진행)
# ---------------------------------------------------------------------------

def _load_verify_cache(name: str) -> dict[str, bool]:
    path = VERIFY_CACHE_DIR / f"{name}.jsonl"
    if not path.exists():
        return {}
    return {r["chunk_id"]: r["passed"] for r in load_jsonl(path)}


def _append_verify_cache(name: str, chunk_id: str, passed: bool) -> None:
    VERIFY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(VERIFY_CACHE_DIR / f"{name}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"chunk_id": chunk_id, "passed": passed}, ensure_ascii=False) + "\n")


_FTC_MATCH_SYSTEM = """당신은 한국 계약법 전문가입니다. 공정거래위원회 사건 정보와 조항 텍스트가 주어집니다.
이 텍스트가 그 사건에서 지적된 위반 사항과 실질적으로 관련된 "계약 조항 원문"이 맞는지 판단하세요.
다음이면 아니오(false)로 답하세요:
- 공정위의 명령문·권고문·안내문 자체(예: "~할 것을 권고한다", "~일 이내에 삭제할 것")
- 사건과 무관하거나 위반과 관계없는 절차·정의 조항
- 오히려 고객에게 유리한 내용을 설명하는 조항(위반 사항과 반대되는 내용)
반드시 JSON으로: {"is_violating_clause": true 또는 false}"""


def verify_ftc_clause_match(client: OpenAI, clause_text: str, violation_types: list[str], case_name: str, retries: int = 3) -> bool | None:
    user_msg = f"사건명: {case_name}\n위반 유형: {violation_types}\n\n조항 텍스트:\n{clause_text[:1500]}"
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=VERIFY_MODEL,
                messages=[{"role": "system", "content": _FTC_MATCH_SYSTEM}, {"role": "user", "content": user_msg}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return json.loads(resp.choices[0].message.content).get("is_violating_clause")
        except Exception as e:
            logger.warning(f"  FTC 조항 매칭 검증 실패 ({attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


_CONTRACT_RELEVANCE_SYSTEM = """당신은 한국 계약법 전문가입니다. 아래 텍스트가 실제로 "계약 해지" 또는
"손해배상·책임제한"에 관한 실질적인 계약 조항인지 판단하세요.
다음이면 아니오(false)로 답하세요: 단순 용어 정의, 목적·총칙 조항, 해지·배상과 무관한 행정 절차 안내.
반드시 JSON으로: {"is_substantive_clause": true 또는 false}"""


def verify_contract_clause_relevance(client: OpenAI, clause_text: str, retries: int = 3) -> bool | None:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=VERIFY_MODEL,
                messages=[{"role": "system", "content": _CONTRACT_RELEVANCE_SYSTEM}, {"role": "user", "content": clause_text[:1500]}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return json.loads(resp.choices[0].message.content).get("is_substantive_clause")
        except Exception as e:
            logger.warning(f"  표준계약서 관련성 검증 실패 ({attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# FTC 추출
# ---------------------------------------------------------------------------

def _build_ftc_record(text: str, doc_id: str, idx: int, domain: str, risk_level: str, n_violations: int, meta: dict) -> dict:
    return {
        "chunk_id":    f"ftc_case:{doc_id}:{idx}",
        "source":      "ftc_case",
        "doc_id":      doc_id,
        "text":        text.strip(),
        "domain":      domain,
        "risk_level":  risk_level,
        "risk_basis":  f"violation_type_count:{n_violations}",
        "metadata":    meta,
    }


def _process_ftc_case(case: dict, risk_level: str, n_violations: int, client: OpenAI, verify_cache: dict[str, bool], counters: Counter) -> list[dict]:
    """케이스 하나에서 조항 레코드를 뽑는다 — 복수 조항 케이스만 GPT 매칭 검증을 거친다."""
    cell = case.get("셀_데이터", {})
    clauses = case.get("조항_원문", [])
    is_single_clause = len(clauses) == 1  # 조항이 하나뿐이면 라벨-텍스트 매칭이 모호할 여지가 없음
    doc_id = cell.get("사건번호", case.get("사건명", ""))
    meta = {"사건명": case.get("사건명", ""), "사건번호": doc_id, "대표조치유형": cell.get("대표조치유형", ""), "위반유형_개수": n_violations, "의결일": cell.get("의결일", "")}

    records: list[dict] = []
    for idx, clause in enumerate(clauses):
        text = str(clause).strip()
        domain = classify_domain(text)
        if not domain or len(text) < 30:
            continue
        if _is_preamble_contaminated(text):
            counters["preamble"] += 1
            continue

        chunk_id = f"ftc_case:{doc_id}:{idx}"
        if not is_single_clause:
            passed = verify_cache.get(chunk_id)
            if passed is None:
                passed = verify_ftc_clause_match(client, text, case.get("위반_유형", []), meta["사건명"])
                if passed is not None:
                    _append_verify_cache("ftc_match", chunk_id, passed)
            if passed is False:
                counters["gpt_mismatch"] += 1
                continue

        records.append(_build_ftc_record(text, doc_id, idx, domain, risk_level, n_violations, meta))
    return records


def extract_ftc_candidates(ftc_path: Path, client: OpenAI) -> tuple[list[dict], dict]:
    """위반_유형 개수(단일=Medium, 복합=High) 기준으로 분류하고, 복수 조항 케이스는 GPT로 텍스트-사건 매칭을 검증한다."""
    with open(ftc_path, encoding="utf-8") as f:
        raw = json.load(f)
    cases = raw.get("사례", [])

    records: list[dict] = []
    action_type_counts = Counter()
    counters = Counter()  # preamble, gpt_mismatch
    excluded_not_confirmed = excluded_no_violation_signal = 0
    verify_cache = _load_verify_cache("ftc_match")

    for case in cases:
        action_type = case.get("셀_데이터", {}).get("대표조치유형", "")
        action_type_counts[action_type] += 1
        if action_type not in _CONFIRMED_ACTION_TYPES:
            excluded_not_confirmed += 1
            continue

        n_violations = len(case.get("위반_유형", []))
        risk_level = _risk_from_violation_count(n_violations)
        if risk_level is None:
            excluded_no_violation_signal += 1
            continue

        records.extend(_process_ftc_case(case, risk_level, n_violations, client, verify_cache, counters))

    stats = {
        "action_type_counts": dict(action_type_counts),
        "excluded_not_confirmed": excluded_not_confirmed,
        "excluded_no_violation_signal": excluded_no_violation_signal,
        "excluded_preamble_contamination": counters["preamble"],
        "excluded_gpt_mismatch": counters["gpt_mismatch"],
        "extracted_records": len(records),
    }
    return records, stats


# ---------------------------------------------------------------------------
# 표준계약서 추출
# ---------------------------------------------------------------------------

def _build_contract_record(text: str, doc_id: str, idx: int, domain: str, meta: dict) -> dict:
    return {
        "chunk_id":    f"standard_contract:{doc_id}:{idx}",
        "source":      "standard_contract",
        "doc_id":      doc_id,
        "text":        text.strip(),
        "domain":      domain,
        "risk_level":  "Low",
        "risk_basis":  "ftc_official_standard_contract_provenance",
        "metadata":    meta,
    }


def extract_standard_contract_low_candidates(contract_dir: Path, client: OpenAI) -> tuple[list[dict], dict]:
    """공정위 공식 표준계약서에서 조문 패턴 문서만 추출하고, GPT로 해지/책임제한 실질 조항인지 검증한다."""
    article_pat = re.compile(r"(제\s*\d+\s*조\s*(?:\([^)]*\))?)", re.MULTILINE)
    records: list[dict] = []
    stats = {"total_docs": 0, "noise_skipped_docs": 0, "excluded_gpt_irrelevant": 0, "extracted_records": 0}
    verify_cache = _load_verify_cache("contract_relevance")

    for path in sorted(contract_dir.glob("contracts_표준*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        category = data.get("카테고리", path.stem)

        for case in data.get("사례", []):
            stats["total_docs"] += 1
            full_text = case.get("추출_텍스트", "")
            if not _ARTICLE_START_PAT.match(full_text.strip()[:50]) and "제" not in full_text[:30]:
                stats["noise_skipped_docs"] += 1
                continue

            meta = {"제목": case.get("제목", ""), "카테고리": category}
            doc_id = case.get("파일ID", case.get("제목", ""))
            for art_idx, (title, body) in enumerate(split_articles(full_text, article_pat)):
                text = f"{title} {body}".strip()
                domain = classify_domain(text)
                if not domain or len(text) < 30:
                    continue

                chunk_id = f"standard_contract:{doc_id}:{art_idx}"
                passed = verify_cache.get(chunk_id)
                if passed is None:
                    passed = verify_contract_clause_relevance(client, text)
                    if passed is not None:
                        _append_verify_cache("contract_relevance", chunk_id, passed)
                if passed is False:
                    stats["excluded_gpt_irrelevant"] += 1
                    continue

                records.append(_build_contract_record(text, doc_id, art_idx, domain, meta))
                stats["extracted_records"] += 1

    return records, stats


def split_held_out(records: list[dict], trained_chunk_ids: set[str]) -> tuple[list[dict], int]:
    """모델 학습(clean.jsonl)에 이미 쓰인 chunk_id를 제외해 평가 누출을 막는다."""
    held_out = [r for r in records if r["chunk_id"] not in trained_chunk_ids]
    return held_out, len(records) - len(held_out)


def main() -> None:
    logger.info("========== FTC ground-truth 후보 재구축 시작(GPT 검증 포함) ==========")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    ftc_records, ftc_stats = extract_ftc_candidates(FTC_PATH, client)
    logger.info(f"  FTC 대표조치유형 분포: {ftc_stats['action_type_counts']}")
    logger.info(f"  제외(미확정): {ftc_stats['excluded_not_confirmed']} | 제외(위반유형 없음): {ftc_stats['excluded_no_violation_signal']} | "
                f"제외(명령문 오염): {ftc_stats['excluded_preamble_contamination']} | 제외(GPT 불일치): {ftc_stats['excluded_gpt_mismatch']}")
    logger.info(f"  추출된 조항: {ftc_stats['extracted_records']}건")

    contract_records, contract_stats = extract_standard_contract_low_candidates(CONTRACT_DIR, client)
    logger.info(f"  표준계약서 노이즈 제외: {contract_stats['noise_skipped_docs']}/{contract_stats['total_docs']} | "
                f"GPT 무관 제외: {contract_stats['excluded_gpt_irrelevant']}")
    logger.info(f"  추출된 Low 후보 조항: {contract_stats['extracted_records']}건")

    high    = [r for r in ftc_records if r["risk_level"] == "High"]
    medium  = [r for r in ftc_records if r["risk_level"] == "Medium"]
    save_jsonl(high, OUT_DIR / "ftc_high_candidates.jsonl")
    save_jsonl(medium, OUT_DIR / "ftc_medium_candidates.jsonl")
    save_jsonl(contract_records, OUT_DIR / "standard_contract_low_candidates.jsonl")

    trained_chunk_ids = {r["chunk_id"] for r in load_jsonl(CLEAN_PATH)}
    high_ho, high_dropped     = split_held_out(high, trained_chunk_ids)
    medium_ho, medium_dropped = split_held_out(medium, trained_chunk_ids)
    low_ho, low_dropped       = split_held_out(contract_records, trained_chunk_ids)

    precedent_path = OUT_DIR / "precedent_candidates.jsonl"
    precedent_records = load_jsonl(precedent_path) if precedent_path.exists() else []
    if precedent_records:
        logger.info(f"  판례 기반 후보(제3의 사법부 소스) 추가: {len(precedent_records)}건 — extract_precedent_ground_truth.py 실행 결과")

    ground_truth = high_ho + medium_ho + low_ho + precedent_records
    save_jsonl(ground_truth, EVAL_DIR / "ground_truth_3class.jsonl")
    logger.info(f"  clean.jsonl 학습 데이터 제외(누출 방지): High -{high_dropped} / Medium -{medium_dropped} / Low -{low_dropped}")
    logger.info(f"  최종 held-out 3-class 평가셋: High {len(high_ho)} / Medium {len(medium_ho)} / Low {len(low_ho)} + 판례 {len(precedent_records)} (총 {len(ground_truth)}건)")
    logger.info(f"  저장: {EVAL_DIR / 'ground_truth_3class.jsonl'}")

    report = {
        "ftc_stats": ftc_stats,
        "contract_stats": contract_stats,
        "candidate_counts": {"High": len(high), "Medium": len(medium), "Low": len(contract_records)},
        "held_out_counts": {"High": len(high_ho), "Medium": len(medium_ho), "Low": len(low_ho)},
        "dropped_as_training_leakage": {"High": high_dropped, "Medium": medium_dropped, "Low": low_dropped},
        "note": (
            "High/Medium 둘 다 위반_유형 개수(복합=High, 단일=Medium) 프록시 — FTC가 공식 부여한 등급이 "
            "아니라 이 프로젝트가 만든 추정치. 복수 조항 케이스는 GPT로 '이 텍스트가 실제 위반 근거 "
            "조항이 맞는지' 검증했지만(사실 매칭, 위험도 재판단 아님), 그 검증 자체의 오류 가능성은 "
            "남아있음. 표준계약서 Low도 동일 강도로 '실질적 해지/책임제한 조항인지' 검증함. 기존 "
            "883/534건 평가(models/README.md)와는 High 정의가 달라 직접 비교 불가."
        ),
    }
    save_json(report, OUT_DIR / "rebuild_report.json")
    logger.info(f"  리포트: {report}")
    logger.info("========== 완료 ==========")


if __name__ == "__main__":
    main()
