# backend/eval/lightrag_resume_indexing.py
"""
lightrag_compare.py의 본작업이 잔액 부족으로 1,500/3,323청크에서 중단됐다. 기존
인덱스(data/eval/lightrag_full_storage, 이미 119MB 저장됨)를 지우지 않고 그대로 열어서
나머지 1,823청크(주로 민법)만 이어서 넣는다 — 처음부터 다시 인덱싱하면 이미 쓴 비용이
낭비된다.

잔액이 빠듯한 상태(약 $0.82)라 진행률 로그를 100청크마다(원래 250) 더 촘촘히 남겨서
중간에 잔액이 바닥나 실패하더라도 어디까지 됐는지 바로 알 수 있게 한다.

실행: python -m backend.eval.lightrag_resume_indexing
"""

import asyncio

from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc

from backend.eval.lightrag_compare import WORKING_DIR, LAWS_PATH
from backend.utils import load_jsonl, load_logger

logger = load_logger("lightrag_resume_indexing.log")

_ALREADY_INDEXED = 1500
_LOG_EVERY = 100


async def main() -> None:
    all_recs = load_jsonl(LAWS_PATH)
    remaining = all_recs[_ALREADY_INDEXED:]
    logger.info(f"  기존 인덱스 유지, 이어서 인덱싱: {len(remaining)}청크 남음(전체 {len(all_recs)}청크 중 {_ALREADY_INDEXED}는 완료)")

    rag = LightRAG(
        working_dir=WORKING_DIR,  # rmtree 안 함 — 기존 인덱스 그대로 이어씀
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=EmbeddingFunc(embedding_dim=1536, max_token_size=8192, func=openai_embed),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    ids = [r["chunk_id"] for r in remaining]
    texts = [r["text"] for r in remaining]

    for i in range(0, len(texts), _LOG_EVERY):
        await rag.ainsert(texts[i:i + _LOG_EVERY], ids=ids[i:i + _LOG_EVERY])
        done = _ALREADY_INDEXED + min(i + _LOG_EVERY, len(texts))
        logger.info(f"    인덱싱 진행: {done}/{len(all_recs)} (전체 기준)")

    logger.info("========== 법령 전체 인덱싱 완료 ==========")


if __name__ == "__main__":
    asyncio.run(main())
