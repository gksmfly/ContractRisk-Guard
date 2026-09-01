# backend/preprocess/__main__.py
"""
필터링된 문서 전처리 진입점 (텍스트 정제 + 청킹)

사용법:
    python -m backend.preprocess
    python -m backend.preprocess --source law
    python -m backend.preprocess --chunk-size 512 --overlap 50 --min-chunk 100

출력:
    data/processed/{laws,precedents,interpretations}.jsonl
    data/processed/preprocess_report.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from backend.preprocess.cleaner import split_chunks
from backend.preprocess.extractor import EXTRACTORS
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("preprocess.log")

DOMAIN_DIR    = Path(os.environ.get("DOMAIN_DIR",    str(PROJECT_ROOT / "data/domain")))
PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", str(PROJECT_ROOT / "data/processed")))

SOURCE_DIRS = {
    "law":            DOMAIN_DIR / "law",
    "precedent":      DOMAIN_DIR / "case",
    "interpretation": DOMAIN_DIR / "commentary",
}


def build_chunk_record(
    source: str,
    doc_id: str,
    rec_idx: int,
    chunk_idx: int,
    text: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "chunk_id":    f"{source}:{doc_id}:{rec_idx}:{chunk_idx}",
        "source":      source,
        "doc_id":      doc_id,
        "rec_index":   rec_idx,
        "chunk_index": chunk_idx,
        "text":        text,
        "metadata":    metadata,
    }


def process_source(
    source: str,
    src_dir: Path,
    out_path: Path,
    chunk_size: int,
    overlap: int,
    min_chunk: int,
    law_chunk_size: int = 800,
) -> dict[str, int]:
    extractor = EXTRACTORS[source]
    files = sorted(src_dir.glob("*.json"))
    chunk_count = skipped = errors = 0

    logger.info(f"  {source}: {len(files)}건 전처리 시작 (chunk_size={chunk_size}, overlap={overlap}, min_chunk={min_chunk})")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for fp in files:
            try:
                doc = json.loads(fp.read_text(encoding="utf-8"))
                records = extractor(doc)
                if not records:
                    skipped += 1
                    continue
                doc_chunk_count = 0
                for rec_idx, (text, meta) in enumerate(records):
                    if not text.strip():
                        continue
                    if source == "law":
                        # 법령 조문: 조문 단위 유지가 원칙. 다만 항·호·목까지 담게 되면서
                        # 일부 조문(민법·상법의 긴 조문 등 2.4%)이 KoE5의 512토큰을 넘는다.
                        # 넘는 텍스트는 SentenceTransformer가 조용히 잘라내므로 — 뒤쪽 호가
                        # 통째로 색인에서 사라진다 — 넘는 것만 나눠 담는다.
                        pieces = ([text] if len(text) <= law_chunk_size
                                  else split_chunks(text, law_chunk_size, overlap, min_chunk=1))
                        head = meta.get("article_head", "")
                        for chunk_idx, piece in enumerate(pieces):
                            # 2번째 조각부터는 "제N조(제목)" 머리말을 다시 붙인다.
                            # 조각만 검색돼도 몇 조인지 알 수 있어야 근거로 인용할 수 있다.
                            if chunk_idx and head and not piece.startswith(head[:20]):
                                piece = f"{head}\n{piece}"
                            record = build_chunk_record(source, fp.stem, rec_idx, chunk_idx, piece, meta)
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            chunk_count += 1
                            doc_chunk_count += 1
                    else:
                        # 판례·해석례: 문장 경계 청킹 + min_chunk 필터
                        chunks = split_chunks(text, chunk_size, overlap, min_chunk)
                        for chunk_idx, chunk_text in enumerate(chunks):
                            record = build_chunk_record(source, fp.stem, rec_idx, chunk_idx, chunk_text, meta)
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            chunk_count += 1
                            doc_chunk_count += 1
                if doc_chunk_count == 0:
                    skipped += 1
            except Exception as e:
                logger.error(f"  전처리 실패 {fp.name}: {e}")
                errors += 1

    logger.info(f"  {source}: {chunk_count}개 청크 생성 (스킵 {skipped}건, 오류 {errors}건) → {out_path}")
    return {"total_docs": len(files), "total_chunks": chunk_count, "skipped": skipped, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description="필터링 데이터 전처리 (정제 + 청킹)")
    parser.add_argument("--source",     default="all", help="처리할 소스 (law/precedent/interpretation/all)")
    parser.add_argument("--chunk-size", type=int, default=512,  help="청크당 최대 문자 수 (기본 512)")
    parser.add_argument("--overlap",    type=int, default=50,   help="청크 간 오버랩 문자 수 (기본 50)")
    parser.add_argument("--min-chunk",  type=int, default=100,  help="최소 청크 길이, 미만 버림 (기본 100)")
    parser.add_argument("--law-chunk-size", type=int, default=800,
                        help="법령 조문을 나누기 시작하는 길이 (기본 800자 ≈ 480토큰, KoE5 상한 512 아래)")
    args = parser.parse_args()

    logger.info("========== 전처리 시작 ==========")
    sources = list(SOURCE_DIRS.keys()) if args.source == "all" else [args.source]
    report_path = PROCESSED_DIR / "preprocess_report.json"
    report: dict = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8")).get("preprocess_result", {})
        except Exception:
            pass

    for source in sources:
        src_dir = SOURCE_DIRS[source]
        if not src_dir.exists():
            logger.warning(f"  {source}: {src_dir} 없음 — domain 먼저 실행하세요.")
            continue
        report[source] = process_source(
            source, src_dir, PROCESSED_DIR / f"{source}s.jsonl",
            args.chunk_size, args.overlap, args.min_chunk, args.law_chunk_size,
        )

    save_json({"preprocess_result": report}, PROCESSED_DIR / "preprocess_report.json")
    logger.info("========== 전처리 완료 ==========")


if __name__ == "__main__":
    main()
