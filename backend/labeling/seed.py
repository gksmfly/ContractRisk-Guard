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
    domain: str,
    risk_level: str,
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


def extract_ftc_records(ftc_path: Path) -> tuple[list[dict], dict]:
    """FTC 시정조치에서 해지·책임제한 조항 레코드를 추출한다."""
    with open(ftc_path, encoding="utf-8") as f:
        raw = json.load(f)
    cases   = raw.get("사례", [])
    records: list[dict] = []
    stats   = {"total": len(cases), "해지_조항": 0, "책임제한_조항": 0, "skipped": 0}

    for case in cases:
        clauses = case.get("조항_원문", [])
        if not clauses:
            stats["skipped"] += 1
            continue
        cell   = case.get("셀_데이터", {})
        meta   = {"사건명": case.get("사건명", ""), "사건번호": cell.get("사건번호", ""), "의결일": cell.get("의결일", "")}
        doc_id = cell.get("사건번호", case.get("사건명", ""))
        risk_level = case.get("risk_level", "High")
        risk_basis = case.get("risk_level_근거", "ftc_confirmed")
        added  = False
        for idx, clause in enumerate(clauses):
            text   = str(clause).strip()
            domain = classify_domain(text)
            if not domain or len(text) < 30:
                continue
            records.append(build_record(text, "ftc_case", doc_id, idx, domain, risk_level, risk_basis, [], meta))
            stats[domain] += 1
            added = True
        if not added:
            stats["skipped"] += 1

    return records, stats


def extract_contract_records(contract_dir: Path) -> tuple[list[dict], dict]:
    """표준계약서에서 조문 단위로 분리하여 레코드를 추출한다."""
    article_pat = re.compile(r"(제\s*\d+\s*조\s*(?:\([^)]*\))?)", re.MULTILINE)
    records: list[dict] = []
    stats: dict[str, Any] = {"file_count": 0, "total_articles": 0, "해지_조항": 0, "책임제한_조항": 0, "skipped": 0}

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

            for art_idx, (title, body) in enumerate(_split_articles(full_text, article_pat)):
                text = f"{title} {body}".strip()
                stats["total_articles"] += 1
                domain = classify_domain(text)
                if not domain or len(text) < 30:
                    stats["skipped"] += 1
                    continue
                risk_level, matched = assess_risk(domain, text)
                records.append(build_record(text, "standard_contract", doc_id, art_idx, domain, risk_level, "pattern_match", matched, meta))
                stats[domain] += 1

    return records, stats


def _split_articles(text: str, pattern: re.Pattern) -> list[tuple[str, str]]:
    """텍스트를 조문 단위로 분리한다."""
    parts = pattern.split(text)
    return [(h.strip(), b.strip()) for h, b in zip(parts[1::2], parts[2::2])]


def main() -> None:
    """Seed 라벨링 진입점."""
    logger.info("========== Seed 라벨링 시작 ==========")

    logger.info("  [1/2] FTC 시정조치 추출 중 ...")
    ftc_records, ftc_stats = extract_ftc_records(FTC_PATH)
    logger.info(f"  FTC: {len(ftc_records)}건 추출 | {ftc_stats}")

    logger.info("  [2/2] 표준계약서 조문 추출 중 ...")
    contract_records, contract_stats = extract_contract_records(CONTRACT_DIR)
    logger.info(f"  계약서: {len(contract_records)}건 추출 | {contract_stats}")

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
