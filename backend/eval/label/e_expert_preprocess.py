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

## 누수 제거 (2026-09-30 추가)
AI Hub 약관 데이터는 **공정위 의결서 조항을 가공한 것**이다(`ftcCnclsns` 필드, gold 255와
같은 조항·같은 조 라벨이 띄어쓰기·번호만 바뀌어 들어 있다). 그대로 학습하면 정답을 외운
모델로 gold를 채점하게 된다. 그래서 평가셋(FTC gold 전 지층 327 + 음성 홀드아웃 표준계약서)과
**공백·기호 제거 후 30자 조각을 하나라도 공유하는** AI Hub 조항은 전부 뺀다(보수적 기준 —
보일러플레이트 공유도 함께 빠지지만 그 손실은 수십 건이다).

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
_SHINGLE_K, _SHINGLE_STEP = 30, 3


def _norm(t: str) -> str:
    # 공백뿐 아니라 기호·번호 표기까지 지운다 — AI Hub는 "(7)"를 "제7항"으로, "10%"를 "10퍼센트"로
    # 바꾼 가공본이라 공백만 지우면 40자 조각 기준을 빠져나가는 근사 중복이 남았다(2026-10-03 확인:
    # 20자 조각의 절반 이상이 겹치는 gold 14건)
    return re.sub(r"[\s\W_]+", "", t or "")


def _shingles(t: str, step: int = _SHINGLE_STEP) -> set[str]:
    t = _norm(t)
    return {t[i:i + _SHINGLE_K] for i in range(0, max(1, len(t) - _SHINGLE_K + 1), step)}


def _eval_shingles() -> set[str]:
    """평가에 쓰는 텍스트 전부(gold 327 + 음성 홀드아웃)의 조각. 조각은 step 1로 촘촘히."""
    from backend.training.retrain import eval_sets
    from backend.training.train_article import load_ftc_gold

    holdout = json.loads((PROJECT_ROOT / "backend/training/negative_holdout_docs.json").read_text("utf-8"))
    _, neg = eval_sets(holdout)
    texts = [g["text"] for g in load_ftc_gold("all")] + [r["text"] for r in neg]
    out: set[str] = set()
    for t in texts:
        out |= _shingles(t, step=1)
    return out


def _drop_leaks(recs: list[dict], banned: set[str]) -> tuple[list[dict], int]:
    kept = [r for r in recs if not (_shingles(r["text"]) & banned)]
    return kept, len(recs) - len(kept)


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

    banned = _eval_shingles()
    train_recs, leak_tr = _drop_leaks(train_recs, banned)
    val_recs, leak_va = _drop_leaks(val_recs, banned)
    logger.info(f"  평가셋 누수 제거: train {leak_tr}건 · val {leak_va}건")

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
    print(f"누수 제거: train {leak_tr} · val {leak_va} (평가셋과 공백·기호 제거 후 30자 조각 공유)")
    print("Test:  gold 255 (FTC) — 학습/전처리와 분리")
    print(f"Train 조항 분포: {dict(art_cnt.most_common())}")
    print(f"저장: {train_out}, {val_out}")

    return {"train_n": len(train_recs), "val_n": len(val_recs), "leak_train": leak_tr, "leak_val": leak_va}


if __name__ == "__main__":
    run()
