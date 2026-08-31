# backend/labeling/seed.py
"""
Seed 데이터 라벨링 스크립트

두 소스에서 해지·책임제한 조항을 추출하고 리스크 라벨을 부착한다.
  - FTC 시정조치(ftc_cases_parsed.json) → risk_level=High (공정위 확정 위반)
  - 표준계약서(contracts_*.json)         → Low/Medium (패턴 기반)

출력:
    data/labels/seed_labeled.jsonl
    data/labels/seed_label_report.json

사용법:
    python -m backend.labeling.seed
"""

import json
import os
import argparse
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from backend.utils import load_logger, save_json, save_jsonl, PROJECT_ROOT

logger = load_logger("seed_label.log")

FTC_PATH     = Path(os.environ.get("FTC_PATH",     str(PROJECT_ROOT / "data/raw/ftc_cases/ftc_cases_parsed.json")))
CONTRACT_DIR = Path(os.environ.get("CONTRACT_DIR", str(PROJECT_ROOT / "data/raw/contract")))
SEED_DIR     = Path(os.environ.get("SEED_DIR",     str(PROJECT_ROOT / "data/labels")))

_TERMINATION_KW = [
    "해지", "해제", "계약을 해지", "계약을 해제", "해지권", "해약", "계약 종료",
    "해지 통보", "해지 통지", "해지 사유",
]
_LIABILITY_KW = [
    "손해배상", "책임제한", "면책", "배상책임", "배상액", "책임을 지지 않",
    "책임을 부담하지 않", "손해를 배상", "배상 한도", "배상의 범위",
]

_TERMINATION_HIGH_PAT = [
    r"즉시\s*해지", r"사전\s*통보?\s*없이", r"일방적", r"귀책사유\s*불문",
    r"어떠한\s*경우에도\s*해지", r"임의로\s*해지", r"필요하다고\s*인정하는\s*경우\s*해지",
]
_TERMINATION_MEDIUM_PAT = [
    r"7일\s*이내", r"정당한\s*이유\s*없이", r"재량으로\s*해지",
    r"통보\s*후\s*(?:[3-9]|1\d)\s*일", r"통지\s*후\s*(?:[3-9]|1\d)\s*일",
]
_LIABILITY_HIGH_PAT = [
    r"모든\s*손해에\s*대하여\s*책임을\s*지지\s*않",
    r"어떠한\s*경우에도.{0,30}책임",
    r"일체의\s*책임.{0,20}지지\s*않",
    r"완전\s*면책",
]
_LIABILITY_MEDIUM_PAT = [
    r"간접\s*손해", r"특별\s*손해", r"배상액.{0,15}상한",
    r"배상액.{0,20}한도", r"\d+배\s*(?:이내|한도)",
]


def classify_domain(text: str) -> str | None:
    """텍스트에서 도메인을 결정한다."""
    t_score = sum(1 for kw in _TERMINATION_KW if kw in text)
    l_score = sum(1 for kw in _LIABILITY_KW if kw in text)
    if t_score == 0 and l_score == 0:
        return None
    return "해지_조항" if t_score >= l_score else "책임제한_조항"


def find_matches(text: str, patterns: list[str]) -> list[str]:
    """패턴 목록에서 텍스트에 매칭되는 항목을 반환한다."""
    return [p for p in patterns if re.search(p, text)]


def assess_risk(domain: str, text: str) -> tuple[str, list[str]]:
    """도메인에 따라 risk_level을 결정한다 (표준계약서 전용)."""
    high_pat = _TERMINATION_HIGH_PAT if domain == "해지_조항" else _LIABILITY_HIGH_PAT
    mid_pat  = _TERMINATION_MEDIUM_PAT if domain == "해지_조항" else _LIABILITY_MEDIUM_PAT
    high_m = find_matches(text, high_pat)
    if high_m:
        return "High", high_m
    mid_m = find_matches(text, mid_pat)
    if mid_m:
        return "Medium", mid_m
    return "Low", []


def build_record(
    text: str,
    source: str,
    doc_id: str,
    idx: int,
    domain: str | None,
    risk_level: str | None,
    risk_basis: str,
    patterns: list[str],
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "chunk_id":         f"{source}:{doc_id}:{idx}",
        "source":           source,
        "doc_id":           doc_id,
        "text":             text.strip(),
        "domain":           domain,
        "risk_level":       risk_level,
        "risk_basis":       risk_basis,
        "patterns_matched": patterns,
        "metadata":         meta,
    }


_DISMISSED_ACTION_TYPES = {"재결기각"}  # 이의제기로 뒤집힌 케이스 — 위반이 확정된 게 아니므로 Seed에서 제외


def extract_ftc_records(ftc_path: Path) -> tuple[list[dict], dict]:
    """FTC 시정조치에서 조항 레코드를 추출한다(도메인 무관 — 전량).

    예전에는 `classify_domain()`이 None을 주면 조항을 버렸다. 그 결과 공정위가 불공정으로
    확정한 조항 중 해지·면책 키워드가 없는 것(전속관할·급부 일방변경·의사표시 의제 등)이
    통째로 빠졌다. 도메인 판정은 LLM 단계로 미룬다.
    """
    with open(ftc_path, encoding="utf-8") as f:
        raw = json.load(f)
    cases   = raw.get("사례", [])
    records: list[dict] = []
    stats   = {"total": len(cases), "해지_조항": 0, "책임제한_조항": 0, "미판정": 0,
               "too_short": 0, "skipped": 0, "dismissed_excluded": 0}

    for case in cases:
        cell = case.get("셀_데이터", {})
        if cell.get("대표조치유형") in _DISMISSED_ACTION_TYPES:
            # 이의제기로 뒤집힌(재결기각) 케이스는 확정 위반이 아니므로 High 라벨 부착 대상에서 제외한다.
            stats["dismissed_excluded"] += 1
            continue
        clauses = case.get("조항_원문", [])
        if not clauses:
            stats["skipped"] += 1
            continue
        meta   = {"사건명": case.get("사건명", ""), "사건번호": cell.get("사건번호", ""), "의결일": cell.get("의결일", "")}
        doc_id = cell.get("사건번호", case.get("사건명", ""))
        risk_level = case.get("risk_level", "High")
        risk_basis = case.get("risk_level_근거", "ftc_confirmed")
        added  = False
        for idx, clause in enumerate(clauses):
            text   = str(clause).strip()
            if len(text) < 30:
                stats["too_short"] += 1
                continue
            # 도메인은 여기서 확정하지 않는다 — 키워드로 못 맞히는 유형(제10·11·12·14조 등)을
            # 통째로 버리게 되기 때문. 약관규제법 유형 판정은 LLM(Forward Labeling)이 한다.
            domain = classify_domain(text)
            records.append(build_record(text, "ftc_case", doc_id, idx, domain, risk_level, risk_basis, [], meta))
            stats[domain or "미판정"] += 1
            added = True
        if not added:
            stats["skipped"] += 1

    return records, stats


def extract_contract_records(contract_dir: Path) -> tuple[list[dict], dict]:
    """표준계약서에서 조문 단위로 분리하여 레코드를 추출한다."""
    article_pat = re.compile(r"(제\s*\d+\s*조\s*(?:\([^)]*\))?)", re.MULTILINE)
    records: list[dict] = []
    stats: dict[str, Any] = {"file_count": 0, "total_articles": 0, "해지_조항": 0,
                             "책임제한_조항": 0, "미판정": 0, "too_short": 0, "skipped": 0}

    for path in sorted(contract_dir.glob("contracts_표준*.json")):
        stats["file_count"] += 1
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        category = data.get("카테고리", path.stem)

        for case in data.get("사례", []):
            full_text = case.get("추출_텍스트", "")
            if not full_text:
                continue
            meta   = {"제목": case.get("제목", ""), "카테고리": category}
            doc_id = case.get("파일ID", case.get("제목", ""))

            for art_idx, (title, body) in enumerate(split_articles(full_text, article_pat)):
                text = f"{title} {body}".strip()
                stats["total_articles"] += 1
                if len(text) < 30:
                    stats["too_short"] += 1
                    continue
                domain = classify_domain(text)
                if domain:
                    risk_level, matched = assess_risk(domain, text)
                    basis = "pattern_match"
                else:
                    # 정규식 판정은 해지·책임제한 도메인 전용이다. 그 밖의 유형은 seed 단계에서
                    # risk를 정하지 않고 LLM 판정에 맡긴다 — 임의로 Low를 붙이면
                    # "표준계약서면 Low"라는 출처 지름길이 오히려 강화된다.
                    risk_level, matched, basis = None, [], "llm_pending"
                records.append(build_record(text, "standard_contract", doc_id, art_idx, domain, risk_level, basis, matched, meta))
                stats[domain or "미판정"] += 1

    return records, stats


def split_articles(text: str, pattern: re.Pattern) -> list[tuple[str, str]]:
    """텍스트를 조문 단위로 분리한다."""
    parts = pattern.split(text)
    return [(h.strip(), b.strip()) for h, b in zip(parts[1::2], parts[2::2])]


def sample_by_document(records: list[dict], target: int, seed: int = 42) -> list[dict]:
    """표준계약서를 **문서 단위로** 무작위 추출한다(조항 단위 금지).

    조항 단위로 뽑으면 같은 계약서의 조항이 학습·평가에 흩어진다 — doc-level leakage로,
    07-21에 검색기반 비교 결론을 뒤집었던 바로 그 함정이다. 문서를 먼저 뽑고 그 문서의
    조항을 통째로 가져온다.

    **비율은 임의 선택이다.** FTC 2,055 : 표준계약서 ~2,000은 1:1이 자연스러워 보여서
    고른 값이고, 실제 계약서에서 위반 조항이 차지하는 비율을 반영하지 않는다(아무도
    모른다). 나중에 학습 시점에 비율을 조절해 A/B 할 수 있도록 라벨은 넉넉히 만든다.
    """
    by_doc: dict[str, list[dict]] = {}
    for r in records:
        by_doc.setdefault(r.get("doc_id") or r["chunk_id"], []).append(r)
    docs = sorted(by_doc)
    random.Random(seed).shuffle(docs)

    picked: list[dict] = []
    for d in docs:
        if len(picked) >= target:
            break
        picked.extend(by_doc[d])
    logger.info(f"  표준계약서 문서 단위 추출: 문서 {len(by_doc)}개 중 일부 → 조항 {len(picked)}건 "
                f"(목표 {target}, 문서 경계 유지)")
    return picked


def main() -> None:
    """Seed 라벨링 진입점.

    여기서 만드는 것은 **조항 텍스트 목록**이지 최종 라벨이 아니다. 조 라벨은
    `backend.fb_check`가 GPT로 붙인다 — 표준계약서를 "정부 발행이니 위반 없음"으로
    찍으면 출처 → 라벨이 결정적 함수가 되어 교락을 코드로 굳히게 된다.
    """
    ap = argparse.ArgumentParser(description="Seed 조항 추출")
    ap.add_argument("--contract-sample", type=int, default=2000,
                    help="표준계약서에서 뽑을 조항 수(문서 단위 추출, 0=전량). "
                         "기본 2000은 FTC(~2,055)와 대략 1:1을 맞춘 **임의 선택**이며 "
                         "실제 위반 비율을 반영하지 않는다")
    ap.add_argument("--seed", type=int, default=42, help="추출 재현성 시드")
    args = ap.parse_args()

    logger.info("========== Seed 라벨링 시작 ==========")

    logger.info("  [1/2] FTC 시정조치 추출 중 ...")
    ftc_records, ftc_stats = extract_ftc_records(FTC_PATH)
    logger.info(f"  FTC: {len(ftc_records)}건 추출 | {ftc_stats}")

    logger.info("  [2/2] 표준계약서 조문 추출 중 ...")
    contract_records, contract_stats = extract_contract_records(CONTRACT_DIR)
    logger.info(f"  계약서: {len(contract_records)}건 추출 | {contract_stats}")

    # 표준계약서를 전량(9,523건) 쓰면 82%를 차지해 "위반 없음" 쪽으로 심하게 기울고,
    # 출처 교락(의결서 문체 vs 표준계약서 문체)이 그대로 재현된다.
    if args.contract_sample > 0:
        contract_records = sample_by_document(contract_records, args.contract_sample, args.seed)

    all_records = ftc_records + contract_records
    save_jsonl(all_records, SEED_DIR / "seed_labeled.jsonl")
    logger.info(f"  저장 완료: {SEED_DIR / 'seed_labeled.jsonl'} ({len(all_records)}건)")

    risk_dist   = dict(Counter(r["risk_level"] for r in all_records))
    domain_dist = dict(Counter(r["domain"] for r in all_records))

    report = {
        "total_records":    len(all_records),
        "risk_distribution": risk_dist,
        "domain_distribution": domain_dist,
        "ftc_stats":        ftc_stats,
        "contract_stats":   contract_stats,
    }
    save_json(report, SEED_DIR / "seed_label_report.json")
    logger.info(f"  리포트: {report}")
    logger.info("========== Seed 라벨링 완료 ==========")


if __name__ == "__main__":
    main()
