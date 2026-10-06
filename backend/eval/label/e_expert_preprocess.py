# backend/eval/label/e_expert_preprocess.py
"""
E_expert — AI Hub 약관 데이터 전처리

## 변환 규칙
- clauseArticle 리스트의 각 항목 → 개별 레코드 (text 필드)
- dvAntageous=2(불리) + relateLaword에 제6~14조 있음 → articles 지목
- dvAntageous=1(유리) → articles=[] (음성)
- dvAntageous=2이나 relateLaword에 제6~14조 없음 → 제외 (애매한 케이스)
- text 길이 < 30자 → 제외

## 출력
- data/aihub_tos/train.jsonl  (TL_2 기반)
- data/aihub_tos/val.jsonl    (VL_2 기반)

## 주의
- 테스트셋은 gold 255 (FTC). AI Hub 데이터는 학습/검증에만 쓴다.
- 도메인 이동: AI Hub(민사계약) vs gold(사업자 약관). 상한선 해석 시 명시.

실행:
    .venv/bin/python -m backend.eval.label.e_expert_preprocess
"""

import json
import re
from pathlib import Path

from backend.labeling.articles import ARTICLE_IDS
from backend.utils import PROJECT_ROOT, load_logger

logger = load_logger("e_expert_preprocess.log")

AIHUB_BASE = PROJECT_ROOT / "data/raw/aihub_tos"
OUT_DIR    = PROJECT_ROOT / "data/aihub_tos"
_MIN_CHARS = 30


def _extract_articles(relateLaword: list[str]) -> list[str]:
    articles = set()
    for law in relateLaword:
        for m in re.findall(r"제(\d+)조", law):
            a = f"제{m}조"
            if a in ARTICLE_IDS:
                articles.add(a)
    return sorted(articles, key=lambda x: int(x.strip("제조")))


def _process_split(json_files: list[Path]) -> list[dict]:
    records = []
    skipped_ambiguous = 0
    skipped_short = 0

    for f in json_files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"파싱 실패 {f}: {e}")
            continue

        adv = d.get("dvAntageous", "")
        clause_texts = d.get("clauseArticle", [])
        relateLaword = d.get("relateLaword", [])

        if adv == "2":
            articles = _extract_articles(relateLaword)
            if not articles:
                skipped_ambiguous += 1
                continue
        elif adv == "1":
            articles = []
        else:
            continue

        for text in clause_texts:
            text = (text or "").strip()
            if len(text) < _MIN_CHARS:
                skipped_short += 1
                continue
            records.append({
                "text": text,
                "articles": articles,
                "agreed_articles": articles,   # train_article.py 호환
                "forward_articles": articles,
                "source": "aihub",
                "dvAntageous": adv,
                "chunk_id": f"aihub_{f.stem}",  # document 그룹핑용
            })

    logger.info(
        f"  변환 완료: {len(records)}건 "
        f"(애매케이스 제외 {skipped_ambiguous}, 짧은텍스트 제외 {skipped_short})"
    )
    return records


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_files = sorted(AIHUB_BASE.rglob("TL_2.약관/**/*.json"))
    val_files   = sorted(AIHUB_BASE.rglob("VL_2.약관/**/*.json"))
    logger.info(f"Train JSON: {len(train_files)}, Val JSON: {len(val_files)}")

    logger.info("Train 전처리 중...")
    train_recs = _process_split(train_files)

    logger.info("Val 전처리 중...")
    val_recs = _process_split(val_files)

    # 저장
    train_out = OUT_DIR / "train.jsonl"
    val_out   = OUT_DIR / "val.jsonl"

    with train_out.open("w", encoding="utf-8") as f:
        for r in train_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with val_out.open("w", encoding="utf-8") as f:
        for r in val_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 통계 요약
    from collections import Counter
    train_pos = sum(1 for r in train_recs if r["articles"])
    train_neg = sum(1 for r in train_recs if not r["articles"])
    val_pos   = sum(1 for r in val_recs if r["articles"])
    val_neg   = sum(1 for r in val_recs if not r["articles"])

    art_cnt = Counter()
    for r in train_recs:
        for a in r["articles"]:
            art_cnt[a] += 1

    print("\n=== E_expert 전처리 결과 ===")
    print(f"Train: {len(train_recs)}건 (양성 {train_pos}, 음성 {train_neg})")
    print(f"Val:   {len(val_recs)}건 (양성 {val_pos}, 음성 {val_neg})")
    print(f"Test:  gold 255 (FTC) — 학습/전처리와 분리")
    print(f"Train 조항 분포: {dict(art_cnt.most_common())}")
    print(f"저장: {train_out}, {val_out}")

    return {"train_n": len(train_recs), "val_n": len(val_recs)}


if __name__ == "__main__":
    run()
