# backend/db/embedding_benchmark.py
"""
프로덕션 Dense Retrieval 임베딩 모델 비교

backend/db/loader.py는 chunks/seed_clauses/clean_clauses/noise_clauses 테이블에
OpenAI text-embedding-3-large를 사용해 임베딩을 적재하지만, 이 선택이 실제로
한국어 법률 도메인 검색에 적합한지 검증된 적이 없었다. 이 스크립트는 해지·책임제한
도메인 판례(data/domain/case) vs 비도메인 무작위 판례를 계약 리스크 검색 쿼리로
구분해내는 능력을 기준으로 후보 모델들을 비교한다.

비교 대상:
  - BAAI/bge-m3                    (다국어, 8,192 토큰)
  - nlpai-lab/KURE-v1              (한국어 Retrieval 특화, BGE-M3 기반 추가 학습)
  - nlpai-lab/KoE5                 (한국어 특화 E5, query:/passage: 프리픽스)
  - openai text-embedding-3-large  (현재 backend/db/loader.py가 사용 중인 모델)

평가 기준:
  - 분리도 (separation): 관련 문서 평균 유사도 - 비관련 문서 평균 유사도
  - 다중 threshold 통과율: 0.40 / 0.45 / 0.50 / 0.55 / 0.60
  - 처리 속도 (docs/sec)

실행:
  python -m backend.db.embedding_benchmark
"""
import gc
import json
import os
import random
import time

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from backend.domain.config import DOMAIN_DIR, PREC_DIR
from backend.utils import PROJECT_ROOT, load_logger

load_dotenv()

logger = load_logger("embedding_benchmark.log")

DOMAIN_PREC_DIR = DOMAIN_DIR / "case"

SAMPLE_N   = 500  # OpenAI 후보의 TPM 레이트리밋(배치당 35초 대기) 때문에 규모를 제한함
DEVICE     = "cuda:1"
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]

# 실제 계약 리스크 분석 상황에서 나올 법한 검색 쿼리
DENSE_QUERIES: list[str] = [
    "약관 불공정 조항 소비자 계약 해지 효력",
    "책임제한 약관 조항 무효 소비자 보호",
    "약관의 규제에 관한 법률 위반 해지 조항",
    "위약금 약관 조항 효력 계약 해제",
    "표준약관 소비자 해지권 제한 유효",
    "전자상거래 방문판매 약관 불공정 손해배상 면책",
]


def _build_prec_text(doc: dict) -> str:
    svc = doc.get("PrecService", {})
    parts = [svc.get("사건명", ""), svc.get("판시사항", ""), svc.get("판결요지", "")]
    return " ".join(p for p in parts if p).strip()


def _load_sample(n: int) -> tuple[list[str], list[str]]:
    domain_ids = {fp.stem for fp in DOMAIN_PREC_DIR.glob("*.json")}

    rel_fps = random.sample(list(DOMAIN_PREC_DIR.glob("*.json")), min(n, len(domain_ids)))
    relevant = []
    for fp in rel_fps:
        try:
            t = _build_prec_text(json.loads(fp.read_text()))
            if t:
                relevant.append(t)
        except Exception:
            pass

    non_domain = [fp for fp in PREC_DIR.glob("*.json") if fp.stem not in domain_ids]
    rand_fps = random.sample(non_domain, min(n, len(non_domain)))
    random_docs = []
    for fp in rand_fps:
        try:
            t = _build_prec_text(json.loads(fp.read_text()))
            if t:
                random_docs.append(t)
        except Exception:
            pass

    return relevant[:n], random_docs[:n]


def _compute_scores(doc_embs: np.ndarray, query_embs: np.ndarray) -> np.ndarray:
    doc_norm   = doc_embs   / (np.linalg.norm(doc_embs,   axis=1, keepdims=True) + 1e-9)
    query_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-9)
    return (query_norm @ doc_norm.T).max(axis=0)


def _make_result(name: str, max_sim: np.ndarray, n_rel: int, elapsed: float, n_total: int) -> dict:
    rel  = max_sim[:n_rel]
    rand = max_sim[n_rel:]
    thresh_stats = {}
    for t in THRESHOLDS:
        thresh_stats[str(t)] = {
            "relevant_pct": round(float((rel  >= t).mean()) * 100, 1),
            "random_pct":   round(float((rand >= t).mean()) * 100, 1),
        }
    return {
        "model":      name,
        "speed":      round(n_total / elapsed, 1),
        "elapsed":    round(elapsed, 2),
        "relevant":   {"mean": round(float(rel.mean()), 4), "std": round(float(rel.std()), 4)},
        "random":     {"mean": round(float(rand.mean()), 4), "std": round(float(rand.std()), 4)},
        "separation": round(float(rel.mean() - rand.mean()), 4),
        "thresholds": thresh_stats,
    }


def _log_result(result: dict) -> None:
    logger.info(f"속도:   {result['speed']} docs/sec ({result['elapsed']:.1f}s)")
    logger.info(f"[관련] 평균={result['relevant']['mean']:.4f} ± {result['relevant']['std']:.4f}")
    logger.info(f"[랜덤] 평균={result['random']['mean']:.4f} ± {result['random']['std']:.4f}")
    logger.info(f"분리도: {result['separation']:.4f}")
    logger.info(f"{'threshold':>10}  {'관련통과%':>9}  {'랜덤통과%':>9}")
    for t, s in result["thresholds"].items():
        logger.info(f"{t:>10}  {s['relevant_pct']:>8.1f}%  {s['random_pct']:>8.1f}%")


def _eval_bgem3(relevant: list[str], random_docs: list[str]) -> dict:
    from FlagEmbedding import BGEM3FlagModel
    name = "bge-m3"
    logger.info(f"\n{'='*60}")
    logger.info(f"[후보] {name} (8,192 tokens)")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=DEVICE)
    all_texts = relevant + random_docs

    t0 = time.time()
    doc_embs   = model.encode(all_texts,     batch_size=32, max_length=8192)["dense_vecs"]
    query_embs = model.encode(DENSE_QUERIES, batch_size=32, max_length=512)["dense_vecs"]
    elapsed    = time.time() - t0

    result = _make_result(name, _compute_scores(doc_embs, query_embs), len(relevant), elapsed, len(all_texts))
    _log_result(result)
    del model; gc.collect(); torch.cuda.empty_cache()
    logger.info("GPU 메모리 해제 완료")
    return result


def _eval_kure(relevant: list[str], random_docs: list[str]) -> dict:
    name = "KURE-v1"
    logger.info(f"\n{'='*60}")
    logger.info(f"[후보] {name} (한국어 Retrieval 특화, BGE-M3 기반)")
    model = SentenceTransformer("nlpai-lab/KURE-v1", device=DEVICE)
    all_texts = relevant + random_docs

    t0 = time.time()
    doc_embs = model.encode(
        all_texts, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    try:
        query_embs = model.encode(
            DENSE_QUERIES, batch_size=64, convert_to_numpy=True,
            normalize_embeddings=True, prompt_name="query",
        )
    except Exception:
        query_embs = model.encode(
            DENSE_QUERIES, batch_size=64, convert_to_numpy=True, normalize_embeddings=True,
        )
    elapsed = time.time() - t0

    result = _make_result(name, _compute_scores(doc_embs, query_embs), len(relevant), elapsed, len(all_texts))
    _log_result(result)
    del model; gc.collect(); torch.cuda.empty_cache()
    logger.info("GPU 메모리 해제 완료")
    return result


def _eval_koe5(relevant: list[str], random_docs: list[str]) -> dict:
    name = "KoE5"
    logger.info(f"\n{'='*60}")
    logger.info(f"[후보] {name} (한국어 특화 E5)")
    model = SentenceTransformer("nlpai-lab/KoE5", device=DEVICE)
    all_texts = relevant + random_docs
    docs_prefixed    = [f"passage: {t}" for t in all_texts]
    queries_prefixed = [f"query: {q}" for q in DENSE_QUERIES]

    t0 = time.time()
    doc_embs   = model.encode(docs_prefixed,    batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    query_embs = model.encode(queries_prefixed, batch_size=64, convert_to_numpy=True)
    elapsed    = time.time() - t0

    result = _make_result(name, _compute_scores(doc_embs, query_embs), len(relevant), elapsed, len(all_texts))
    _log_result(result)
    del model; gc.collect(); torch.cuda.empty_cache()
    logger.info("GPU 메모리 해제 완료")
    return result


def _eval_openai(relevant: list[str], random_docs: list[str]) -> dict:
    from openai import OpenAI

    from backend.db.loader import EMBED_DIM, EMBED_MODEL, embed_texts

    name = f"openai/{EMBED_MODEL} (dim={EMBED_DIM}, 운영 중)"
    logger.info(f"\n{'='*60}")
    logger.info(f"[현재 운영 모델] {name}")
    all_texts = relevant + random_docs

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        t0 = time.time()
        doc_embs   = np.array(embed_texts(client, all_texts))
        query_embs = np.array(embed_texts(client, DENSE_QUERIES))
        elapsed    = time.time() - t0
    except Exception as e:
        logger.error(f"OpenAI 임베딩 API 호출 실패: {e}")
        raise

    result = _make_result(name, _compute_scores(doc_embs, query_embs), len(relevant), elapsed, len(all_texts))
    _log_result(result)
    return result


def main() -> None:
    random.seed(42)
    logger.info(f"샘플 로딩 중 (관련 {SAMPLE_N}건 + 랜덤 {SAMPLE_N}건)...")
    logger.info(f"관련 출처: {DOMAIN_PREC_DIR}")
    logger.info(f"쿼리: DENSE_QUERIES ({len(DENSE_QUERIES)}개)")
    relevant, random_docs = _load_sample(SAMPLE_N)
    logger.info(f"관련 {len(relevant)}건 / 랜덤 {len(random_docs)}건 로드 완료")

    results = [
        _eval_bgem3(relevant, random_docs),
        _eval_kure(relevant, random_docs),
        _eval_koe5(relevant, random_docs),
        _eval_openai(relevant, random_docs),
    ]

    logger.info(f"\n{'='*60}")
    logger.info("최종 비교")
    logger.info(f"{'모델':<45} {'분리도':>8} {'관련평균':>8} {'랜덤평균':>8} {'속도(d/s)':>10}")
    logger.info("-" * 85)
    for r in results:
        logger.info(
            f"{r['model']:<45} {r['separation']:>8.4f} "
            f"{r['relevant']['mean']:>8.4f} {r['random']['mean']:>8.4f} "
            f"{r['speed']:>10.1f}"
        )

    best = max(results, key=lambda x: x["separation"])
    logger.info(f"\n추천 모델: {best['model']} (분리도 {best['separation']:.4f})")

    out_path = PROJECT_ROOT / "data" / "embedding_benchmark_result.json"
    out_path.write_text(
        json.dumps({"추천_모델": best["model"], "결과": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
