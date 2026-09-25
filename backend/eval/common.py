# backend/eval/common.py
"""여러 평가 스크립트가 공유하는 법조문 검색 정답셋(FTC 근거_법령) 로더."""

import json
import os
import random
import re
from pathlib import Path

from backend.utils import PROJECT_ROOT

PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", str(PROJECT_ROOT / "data/processed")))
LAWS_PATH = PROCESSED_DIR / "laws.jsonl"
FTC_PATH = Path(os.environ.get("FTC_PATH", str(PROJECT_ROOT / "data/raw/ftc_cases/ftc_cases_parsed.json")))

N_QUERIES = 100

_PAT = re.compile(r"^(.*?)\s*제\s*(\d+)\s*조")


def build_ground_truth(law_recs: list[dict], n_cases: int, seed: int = 42) -> list[dict]:
    """FTC 근거_법령 전체(3개 법령)를 파싱해 평가 쿼리를 만든다. 정답 조문의 실제 본문도 같이 붙인다."""
    extract_article_text = {(r["metadata"]["law_name"], r["metadata"]["article_no"]): r["text"] for r in law_recs}

    with open(FTC_PATH, encoding="utf-8") as f:
        cases = json.load(f).get("사례", [])

    candidates = []
    for case in cases:
        grounds = case.get("근거_법령", [])
        clauses = case.get("조항_원문", [])
        if not grounds or not clauses:
            continue
        pairs = set()
        for g in grounds:
            m = _PAT.match(g.replace("\n", " ").strip())
            if m:
                pairs.add((m.group(1).strip(), m.group(2)))
        pairs = {p for p in pairs if p in extract_article_text}  # 코퍼스에 실제 있는 것만
        if not pairs:
            continue
        candidates.append({
            "case_name": case.get("사건명", ""),
            "clause": str(clauses[0])[:300],
            "correct_pairs": sorted(pairs),
            "correct_texts": [extract_article_text[p][:60] for p in pairs],  # 판정용 — 조문 본문 앞부분
        })

    random.seed(seed)
    return random.sample(candidates, min(n_cases, len(candidates)))
