# backend/eval/lbox_embedding_compare.py
"""
Contextual Retrieval(옵션4, contextual_retrieval_compare.py)에서 쓴 KoE5(범용 임베딩)를
LBOX(한국 리걸테크 회사)의 법률 도메인 특화 임베딩 모델(`lbox/legalstructure-aware-
embedding-pretrained-model`, bge-m3-retromae 기반)로 바꿔서 같은 dense-only 비교를
재실행한다.

**주의(로드 시 확인된 제약)**: 이 체크포인트의 아키텍처(`RobertaForSAILER`, SAILER
논문의 구조인식 판결 인코더)는 표준 transformers에 없는 커스텀 클래스라 `AutoModel`로
못 그대로 로드된다. `trust_remote_code`용 모델링 코드도 이 캐시엔 없어서, 표준
`RobertaModel`로 로드하면 SAILER 전용 헤드(jud_head/reason_head, 판결이유 구조를
반영하는 특수 부분)는 전부 버려지고 pooler.dense도 체크포인트에 없어 무작위 초기화된다
— 즉 여기서 쓰는 건 "LBOX가 학습시킨 공유 인코더 바디"(mean-pooling으로 문장 벡터 추출)
뿐이고, 이 모델의 핵심 셀링포인트인 "구조인식" 부분은 검증하지 못한다. 그래도 공유
인코더 바디 자체는 법률 텍스트로 사전학습된 가중치라 범용 KoE5와 비교할 가치는 있다.

Dense-only 순수 비교(contextual_retrieval_compare.py와 동일 방법론) — 법령 3,323청크
전체를 메모리에 올려 numpy 코사인 유사도로 비교.

실행: .venv/bin/python -m backend.eval.lbox_embedding_compare
"""

from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from backend.eval.lightrag_compare import _N_QUERIES, LAWS_PATH, build_ground_truth
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("lbox_embedding_compare.log")
OUT_PATH = PROJECT_ROOT / "data/eval/lbox_embedding_vs_koe5_report.json"

_MODEL_ID = "lbox/legalstructure-aware-embedding-pretrained-model"
_TOP_K = 5
_BATCH = 32
_MAX_LEN = 256
_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_lbox() -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
    model = AutoModel.from_pretrained(_MODEL_ID).to(_DEVICE)
    model.eval()
    return model, tokenizer


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts


def _encode_all(model: Any, tokenizer: Any, texts: list[str]) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), _BATCH):
        batch = texts[i:i + _BATCH]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=_MAX_LEN, return_tensors="pt").to(_DEVICE)
        with torch.no_grad():
            out = model(**enc)
        pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        vecs.extend(pooled.cpu().numpy())
        logger.info(f"    임베딩 진행: {min(i + _BATCH, len(texts))}/{len(texts)}")
    return np.array(vecs, dtype=np.float32)


def _cosine_top_k(query_vec: np.ndarray, corpus_vecs: np.ndarray, k: int) -> list[int]:
    q = query_vec / np.linalg.norm(query_vec)
    c = corpus_vecs / np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
    sims = c @ q
    return list(np.argsort(-sims)[:k])


def main() -> None:
    law_recs = load_jsonl(LAWS_PATH)
    queries = build_ground_truth(law_recs, n_cases=_N_QUERIES)
    logger.info(f"  평가 쿼리 {len(queries)}건 생성(contextual_retrieval_compare.py와 동일 seed/로직)")

    logger.info(f"  LBOX 임베딩 모델 로드: {_MODEL_ID}")
    model, tokenizer = _load_lbox()

    logger.info("  법령 3,323청크 LBOX 임베딩 계산 중...")
    corpus_vecs = _encode_all(model, tokenizer, [r["text"] for r in law_recs])

    # KoE5는 이 스크립트에서 다시 돌리지 않는다 — 아래 출력의 "KoE5 참조값 12%"는
    # 이전 측정에서 가져온 상수다. 그래서 카운터도 LBOX 쪽 하나뿐이다
    # (`koe5_hits`가 0으로 초기화만 되고 한 번도 증가하지 않은 채 남아 있었다).
    lbox_hits = 0
    per_query = []
    for i, q in enumerate(queries):
        enc = tokenizer([q["clause"]], padding=True, truncation=True, max_length=_MAX_LEN, return_tensors="pt").to(_DEVICE)
        with torch.no_grad():
            out = model(**enc)
        query_vec = _mean_pool(out.last_hidden_state, enc["attention_mask"])[0].cpu().numpy()

        correct = set(map(tuple, q["correct_pairs"]))
        idx = _cosine_top_k(query_vec, corpus_vecs, _TOP_K)
        found = {(law_recs[j]["metadata"]["law_name"], law_recs[j]["metadata"]["article_no"]) for j in idx}
        l_hit = bool(found & correct)
        lbox_hits += l_hit
        per_query.append({"case_name": q["case_name"], "correct_pairs": q["correct_pairs"], "lbox_dense_hit": l_hit})
        if (i + 1) % 25 == 0:
            logger.info(f"  평가 진행: {i + 1}/{len(queries)} | 누적 LBOX={lbox_hits}")

    n = len(queries)
    result = {
        "n_queries": n,
        "lbox_dense_hit_rate": lbox_hits / n,
        "koe5_baseline_dense_hit_rate_reference": 0.12,
        "lbox_dense_hits": lbox_hits,
        "per_query": per_query,
        "note": (
            "KoE5 dense-only baseline은 contextual_retrieval_compare.py의 12%(동일 방법론, "
            "동일 100건)를 그대로 참조 — 재계산 안 함. LBOX 체크포인트의 SAILER 전용 헤드는 "
            "표준 아키텍처 미지원으로 로드 시 버려짐(공유 인코더 바디만 mean-pooling 사용) — "
            "이 모델의 '구조인식' 핵심 기능은 검증 못 함, caveat 있음."
        ),
    }
    save_json(result, OUT_PATH)
    logger.info(f"========== 최종 결과 (n={n}) ==========")
    logger.info(f"  LBOX dense-only:  {lbox_hits}/{n} ({lbox_hits/n*100:.1f}%)  [KoE5 참조값: 12%]")
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
