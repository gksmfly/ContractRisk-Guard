"""
Dense Retrieval 모델 비교
  - Baseline : ko-sroberta-multitask (512 tokens, 2021)
  - 후보 1   : bge-m3              (8,192 tokens, FlagEmbedding)
  - 후보 2   : multilingual-e5-large-instruct (instruction-tuned)

평가 기준:
  - 분리도 (separation): 관련 문서 평균 유사도 - 비관련 문서 평균 유사도
  - 다중 threshold 통과율: 0.40 / 0.45 / 0.50 / 0.55 / 0.60
  - 처리 속도 (docs/sec)

실행:
  python -m backend.domain.benchmark
"""
import gc
import json
import random
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from backend.domain.config import DOMAIN_DIR, PREC_DIR, PREC_DENSE_QUERIES

DOMAIN_PREC_DIR = DOMAIN_DIR / "case"

SAMPLE_N   = 10_000  # 사실상 전체 사용 — min(SAMPLE_N, 실제 건수)로 자동 제한됨
DEVICE     = "cuda:1"
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]

E5_TASK = "계약 해지 또는 책임제한 조항과 관련된 판례를 검색"


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


def _print_result(result: dict) -> None:
    print(f"  속도:   {result['speed']} docs/sec ({result['elapsed']:.1f}s)")
    print(f"  [관련] 평균={result['relevant']['mean']:.4f} ± {result['relevant']['std']:.4f}")
    print(f"  [랜덤] 평균={result['random']['mean']:.4f} ± {result['random']['std']:.4f}")
    print(f"  분리도: {result['separation']:.4f}")
    print(f"  {'threshold':>10}  {'관련통과%':>9}  {'랜덤통과%':>9}")
    for t, s in result["thresholds"].items():
        print(f"  {t:>10}  {s['relevant_pct']:>8.1f}%  {s['random_pct']:>8.1f}%")


def _eval_sroberta(relevant: list[str], random_docs: list[str]) -> dict:
    name = "ko-sroberta-multitask"
    print(f"\n{'='*60}")
    print(f"[BASELINE] {name} (512 tokens)")
    model = SentenceTransformer("jhgan/ko-sroberta-multitask", device=DEVICE)
    all_texts = relevant + random_docs

    t0 = time.time()
    doc_embs   = model.encode(all_texts,          batch_size=512, show_progress_bar=True, convert_to_numpy=True)
    query_embs = model.encode(PREC_DENSE_QUERIES, batch_size=512, convert_to_numpy=True)
    elapsed    = time.time() - t0

    result = _make_result(name, _compute_scores(doc_embs, query_embs), len(relevant), elapsed, len(all_texts))
    _print_result(result)
    del model; gc.collect(); torch.cuda.empty_cache()
    print("  GPU 메모리 해제 완료")
    return result


def _eval_bgem3(relevant: list[str], random_docs: list[str]) -> dict:
    from FlagEmbedding import BGEM3FlagModel
    name = "bge-m3"
    print(f"\n{'='*60}")
    print(f"[후보1] {name} (8,192 tokens)")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=DEVICE)
    all_texts = relevant + random_docs

    t0 = time.time()
    doc_embs   = model.encode(all_texts,          batch_size=32, max_length=8192)["dense_vecs"]
    query_embs = model.encode(PREC_DENSE_QUERIES, batch_size=32, max_length=512)["dense_vecs"]
    elapsed    = time.time() - t0

    result = _make_result(name, _compute_scores(doc_embs, query_embs), len(relevant), elapsed, len(all_texts))
    _print_result(result)
    del model; gc.collect(); torch.cuda.empty_cache()
    print("  GPU 메모리 해제 완료")
    return result


def _eval_e5_instruct(relevant: list[str], random_docs: list[str]) -> dict:
    name = "multilingual-e5-large-instruct"
    print(f"\n{'='*60}")
    print(f"[후보2] {name} (instruction-tuned)")
    model = SentenceTransformer("intfloat/multilingual-e5-large-instruct", device=DEVICE)
    all_texts = relevant + random_docs

    queries_with_instr = [f"Instruct: {E5_TASK}\nQuery: {q}" for q in PREC_DENSE_QUERIES]

    t0 = time.time()
    doc_embs   = model.encode(all_texts,          batch_size=128, show_progress_bar=True, convert_to_numpy=True)
    query_embs = model.encode(queries_with_instr, batch_size=128, convert_to_numpy=True)
    elapsed    = time.time() - t0

    result = _make_result(name, _compute_scores(doc_embs, query_embs), len(relevant), elapsed, len(all_texts))
    _print_result(result)
    del model; gc.collect(); torch.cuda.empty_cache()
    print("  GPU 메모리 해제 완료")
    return result


def main() -> None:
    random.seed(42)
    print(f"샘플 로딩 중 (관련 {SAMPLE_N}건 + 랜덤 {SAMPLE_N}건)...")
    print(f"  관련 출처: {DOMAIN_PREC_DIR}")
    print(f"  쿼리: PREC_DENSE_QUERIES ({len(PREC_DENSE_QUERIES)}개)")
    relevant, random_docs = _load_sample(SAMPLE_N)
    print(f"  관련 {len(relevant)}건 / 랜덤 {len(random_docs)}건 로드 완료")

    results = [
        _eval_sroberta(relevant, random_docs),
        _eval_bgem3(relevant, random_docs),
        _eval_e5_instruct(relevant, random_docs),
    ]

    print(f"\n{'='*60}")
    print("최종 비교")
    print(f"{'모델':<42} {'분리도':>8} {'관련평균':>8} {'랜덤평균':>8} {'속도(d/s)':>10}")
    print("-" * 82)
    for r in results:
        print(
            f"{r['model']:<42} {r['separation']:>8.4f} "
            f"{r['relevant']['mean']:>8.4f} {r['random']['mean']:>8.4f} "
            f"{r['speed']:>10.1f}"
        )

    best = max(results, key=lambda x: x["separation"])
    print(f"\n추천 모델: {best['model']} (분리도 {best['separation']:.4f})")
    print("\n[threshold 재보정 참고]")
    print(f"  → 관련통과% ≥ 70% & 랜덤통과% ≤ 30% 구간을 threshold로 선택하세요")

    out_path = DOMAIN_DIR / "benchmark_result.json"
    out_path.write_text(
        json.dumps({"추천_모델": best["model"], "결과": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
