# backend/eval/precedent_ground_truth.py
"""
판례 검색 평가용 ground truth 생성 — 판례 원본의 `참조조문`을 FTC 근거_법령과 조인한다.

배경: 법령(3,323청크)은 FTC 근거_법령을 정답으로 13종 대안까지 비교했지만, 판례
(30,154청크, 코퍼스의 90%)는 정확도를 한 번도 측정한 적이 없다. 측정 경로가 없으니
`evidence_selection_agent`의 법원 심급 가중치 버그(고등법원 문자열 불일치, 대법원 가산
과대)가 드러날 방법도 없었다.

정답 구성 원리 — 법령 평가와 **같은 축**을 쓴다:

    FTC 케이스 조항_원문  ──(쿼리)──▶  판례 검색
    FTC 케이스 근거_법령  ──(조인)──▶  그 조문을 참조조문으로 갖는 판례 = 정답

원본 판례 JSON(`data/raw/case/{doc_id}.json`)의 `PrecService.참조조문`에 그 판례가 다룬
법조문이 들어 있다(예: "근로기준법 제27조"). `chunks.doc_id`가 원본 파일명과 1:1로
대응하므로(실측: 1,995/1,995 매칭, 실패 0건) 추가 수집 없이 조인된다. 참조조문 보유율은
98.9%.

**gold 정의 3종** — 단순 합집합은 정답이 너무 넓어(중앙값 131건/1,995건) 무작위로 찍어도
맞는다. 세 정의를 모두 지원해 실측으로 채택한다:

| mode | 뜻 | 커버 | 정답 중앙값 | 무작위 hit@5 |
|---|---|---|---|---|
| `union` | 근거_법령 중 **어느 것이든** 참조 | 99/100 | 131 | 20.9% |
| `intersection` | 근거_법령을 **전부** 참조 | 45/100 | 17 | 10.1% |
| `rarest` | 근거_법령 중 **가장 희소한 조문**만 | 99/100 | 3 | **5.5%** |

`rarest`가 유력하다 — 커버리지를 잃지 않으면서 무작위 기저율이 낮아 판별력이 있고,
법령 평가의 RRF 기저선(8%)과 같은 축에서 읽힌다. 근거: FTC가 인용한 조문 중 가장
희소한 것이 그 사건의 특징적 쟁점이다(약관법 제6조 같은 일반조항 131건보다 구체
조항이 사건을 규정한다).

쿼리 생성은 `lightrag_compare.build_ground_truth()`를 그대로 재사용한다 — 같은 seed·같은
샘플링이라 기존 13종 실험과 쿼리 집합이 동일하고, 결과를 나란히 읽을 수 있다.

실행(인덱스 캐시 생성):
    .venv/bin/python -m backend.eval.precedent_ground_truth
"""

import json
import re
from collections import Counter
from pathlib import Path

from backend.db.connection import get_conn
from backend.eval.lightrag_compare import LAWS_PATH, build_ground_truth
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("precedent_ground_truth.log")

RAW_CASE_DIR = Path("data/raw/case")
INDEX_PATH = PROJECT_ROOT / "data/eval/precedent_ref_index.json"

# "약관의 규제에 관한 법률 제6조" / "약관의규제에관한법률 제6조" 양쪽을 잡는다.
_ARTICLE_RE = re.compile(r"([가-힣·\s]{2,20}법(?:률)?)\s*제(\d+)조")

_index_cache: dict | None = None


def normalize_law_name(name: str) -> str:
    """법령명 표기 변형 흡수 — 원본 판례는 `약관의규제에관한법률`(붙여쓰기), FTC는
    `약관의 규제에 관한 법률`(띄어쓰기)로 쓴다. 공백·가운뎃점을 제거해 같은 키로 만든다."""
    return re.sub(r"[\s·]", "", name or "")


def _parse_articles(ref_text: str) -> set[tuple[str, str]]:
    ref_text = (ref_text or "").replace("<br/>", " ")
    return {(normalize_law_name(m.group(1)), m.group(2)) for m in _ARTICLE_RE.finditer(ref_text)}


def build_reference_index(force: bool = False) -> dict:
    """DB에 적재된 판례의 doc_id → 참조조문 인덱스를 만든다(캐시).

    반환: {
      "art2docs":       {"법령명|조번호": [doc_id, ...]},
      "chunk2doc":      {chunk_id: doc_id},
      "chunks_per_doc": {doc_id: 청크수},
    }
    chunk2doc가 필요한 이유: 검색 결과(`fetch_candidates`)는 chunk_id·metadata만 돌려주고
    doc_id는 안 준다. 정답 판정을 사건 단위로 하려면 chunk_id를 doc_id로 되돌려야 한다.
    """
    global _index_cache
    if _index_cache is not None and not force:
        return _index_cache
    if INDEX_PATH.exists() and not force:
        _index_cache = json.loads(INDEX_PATH.read_text())
        return _index_cache

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT chunk_id, doc_id FROM chunks WHERE source = 'precedent'")
        chunk2doc = {c: d for c, d in cur.fetchall()}

    chunks_per_doc = Counter(chunk2doc.values())
    art2docs: dict[str, set] = {}
    missing = 0
    for doc_id in chunks_per_doc:
        fp = RAW_CASE_DIR / f"{doc_id}.json"
        if not fp.exists():
            missing += 1
            continue
        try:
            rec = json.loads(fp.read_text(encoding="utf-8"))["PrecService"]
        except Exception:
            missing += 1
            continue
        for law, art in _parse_articles(rec.get("참조조문", "")):
            art2docs.setdefault(f"{law}|{art}", set()).add(doc_id)

    logger.info(f"  판례 {len(chunks_per_doc):,}건 / 청크 {len(chunk2doc):,}개, 원본 미매칭 {missing}건")
    logger.info(f"  참조조문 인덱스: {len(art2docs):,}개 (법령,조) 쌍")

    _index_cache = {
        "art2docs": {k: sorted(v) for k, v in art2docs.items()},
        "chunk2doc": chunk2doc,
        "chunks_per_doc": dict(chunks_per_doc),
    }
    save_json(_index_cache, INDEX_PATH)
    return _index_cache


def build_precedent_gold(n_cases: int = 100, seed: int = 42, mode: str = "rarest") -> list[dict]:
    """FTC 케이스별로 (쿼리, 정답 판례 doc_id 집합)을 만든다.

    mode: "union" | "intersection" | "rarest" (모듈 docstring의 표 참고)
    반환 항목: {case_name, clause, correct_pairs, gold_docs, gold_chunk_ratio}
      gold_chunk_ratio — 정답 판례가 전체 판례 청크에서 차지하는 비율(무작위 기저율 계산용)
    """
    if mode not in ("union", "intersection", "rarest"):
        raise ValueError(f"알 수 없는 mode: {mode}")

    index = build_reference_index()
    art2docs = {k: set(v) for k, v in index["art2docs"].items()}
    chunks_per_doc = index["chunks_per_doc"]
    total_chunks = sum(chunks_per_doc.values())

    queries = build_ground_truth(load_jsonl(LAWS_PATH), n_cases=n_cases, seed=seed)

    out = []
    for q in queries:
        keys = [f"{normalize_law_name(law)}|{art}" for law, art in q["correct_pairs"]]
        sets = [art2docs.get(k, set()) for k in keys]
        present = [s for s in sets if s]

        if not present:
            gold = set()
        elif mode == "union":
            gold = set().union(*present)
        elif mode == "intersection":
            gold = set.intersection(*sets) if all(sets) else set()
        else:  # rarest — 가장 적은 판례가 인용한 조문 = 그 사건의 특징적 쟁점
            gold = min(present, key=len)

        out.append({
            "case_name": q["case_name"],
            "clause": q["clause"],
            "correct_pairs": q["correct_pairs"],
            "gold_docs": sorted(gold),
            "gold_chunk_ratio": sum(chunks_per_doc.get(d, 0) for d in gold) / total_chunks,
        })
    return out


def main() -> None:
    build_reference_index(force=True)
    for mode in ("union", "intersection", "rarest"):
        gold = build_precedent_gold(n_cases=100, mode=mode)
        covered = [g for g in gold if g["gold_docs"]]
        sizes = sorted(len(g["gold_docs"]) for g in covered)
        # 무작위로 top-5를 뽑았을 때의 기대 적중률 — 이 값보다 못 하면 검색이 무의미하다
        rnd5 = sum(1 - (1 - g["gold_chunk_ratio"]) ** 5 for g in covered) / max(len(covered), 1)
        logger.info(
            f"  [{mode:<12}] 커버 {len(covered)}/100 | 정답 중앙값 {sizes[len(sizes) // 2] if sizes else 0}건 "
            f"| 무작위 hit@5 {rnd5 * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
