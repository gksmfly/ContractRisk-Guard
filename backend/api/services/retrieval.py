# backend/api/services/retrieval.py
"""
근거 법령·판례 검색 — chunks 테이블(법령 조문 + 판례) 대상 Hybrid(Dense+Sparse) Retrieval,
clean_clauses 테이블 대상 유사 라벨 조항 검색(Red-team Agent용)

analyze.py가 하드코딩된 LEGAL_BASIS 대신 실제 조항 텍스트로 관련 법조문·판례를
검색하도록 지원한다. backend.agents의 retrieval_strategy_agent/red_team_agent가
이 모듈을 호출하는 얇은 LangGraph 노드다 — 실제 검색 로직(DB 접근, 임베딩,
랭킹 융합)은 SRP상 여기(DB 서비스 레이어)에 남겨둔다.

법령(source='law')과 판례(source='precedent')를 각각 top_k씩 따로 검색한다 —
판례가 법령보다 10배 가까이 많아(30,154 vs 3,323 청크), 하나로 합쳐서
유사도 순 정렬하면 법령 인용이 거의 안 나올 수 있어 소스별로 결과를 보장한다.

Dense(의미 검색, KoE5 임베딩 + pgvector 코사인) 단독으로는 "제9조" 같은 정확한
조문 번호나 특정 법률 용어를 놓칠 수 있어, Sparse(어휘 검색, pg_trgm 문자
3-gram 유사도) 결과를 Reciprocal Rank Fusion(RRF)으로 합친다. 한국어 형태소
분석기(예: Mecab)는 시스템 바이너리 설치가 추가로 필요해 이번 단계에서는
쓰지 않고, 토크나이저 없이도 바로 쓸 수 있는 pg_trgm으로 Sparse 신호를 만든다.

Evidence Selection Agent 구축 전 사전 실험(clean_clauses 478건, 정답 조문
적중률 기준)에서 Cross-Encoder(BAAI/bge-reranker-v2-m3) 재랭킹이 RRF-only보다
오히려 낮은 결과(10.7% vs 20.1%)를 보여 채택하지 않았다 — 이 모듈은 RRF까지만
하고, 그 이상의 재랭킹(법원 심급 가중치 등)은 backend.agents.evidence_selection_agent
가 담당한다. (주의: 이 실험 스크립트는 레포에 안 남아있어 재현 불가 — 아래
LightRAG 비교가 훨씬 신뢰도 높은 재현 가능한 수치.)

**LightRAG 대안 비교(2026-08-05, `backend/eval/lightrag_compare*.py`, 법령 전체
확정)**: 원래 설계는 LightRAG였는데 Mecab 의존성 때문에 성능 비교 없이 Hybrid
RRF로 대체됐다. FTC 근거_법령(공정위 실제 인용 법조문, 981케이스 2,444건, 100%
파싱·매칭 확인)을 ground truth로 재현 가능한 비교를 진행. 법령 전체 3,323청크
(민법 포함, 100% 인덱싱 완료) 대상 최종 100건 평가:
RRF 12/100(12.0%) vs LightRAG 20/100(20.0%) — LightRAG만 맞은 14건, RRF만 맞은
6건, 둘 다 맞은 6건, 둘 다 못 맞춘 74건. McNemar 정확검정 p=0.115로 100건
표본에서는 통계적 유의성 미확보(discordant pair 20건 중 14:6). 절대 적중률
자체도 두 방식 다 낮다(20% 이하) — "LightRAG가 낫다"보다 "RRF의 법조문 검색
자체가 이 코퍼스에서 전반적으로 약하다"는 해석이 더 정확할 수 있음.
(참고: 1,500/3,323청크만 인덱싱했던 예산 제약 하의 이전 예비 결과는 RRF
10.0% vs LightRAG 35.0%였으나, 쿼리 표본 자체가 달라 이 최종 결과와 직접
비교 불가 — 이 최종 수치로 대체됨.) 결과는
`data/eval/lightrag_vs_rrf_report_final.json`. 아키텍처 전환 여부는 미결정
— 유의성 부족·전면 재인덱싱 비용(그래프 추출 LLM 호출) 대비 이득이 작아
현재는 Hybrid RRF 유지, 판례 코퍼스(30,154청크)는 LightRAG 미검증.

**검색 아키텍처 대안 13종 실측(2026-08-06, `backend/eval/*_compare.py`,
`backend/eval/retrieval_alternatives_survey.md`)**: LightRAG의 실패 원인(코퍼스
확장 시 그래프 희석)을 근거로 그래프 없는 대안·그래프 스코핑 대안·모델/임베딩
교체까지 전부 같은 100건 ground truth로 비교했다(GraphRAG 스코핑·SEAL-RAG도
"판단만 하고 넘어가지 말라"는 요청에 따라 실제 구현). 최고 성능은 로컬
EXAONE-3.5-7.8B-Instruct(OpenAI 비용 0)로 쿼리를 재구성(LegalMALR-lite)하거나
법령을 먼저 라우팅(RAPTOR-lite)하는 두 방법, 둘 다 RRF 8%→33%(p<0.0001, LightRAG
20%를 크게 앞섬). **의외의 발견**: 이 두 기법 + 재랭킹까지 전부 합친 "종합 콤보"는
28%로 오히려 개별 최고보다 낮았고, 더 큰 모델(Qwen2.5-14B)로 교체해도 26~28%로
EXAONE(7.8B, 한국어 특화)보다 낮았다 — "더 합치면/더 크면 좋다"는 가정이 이
태스크에서는 성립하지 않음. 그래프도 LLM도 없는 방법 중에는 도메인 파티션별
후보 확보+Cross-Encoder 재랭킹 조합만 유의성을 확보했다(8%→17%, p=0.0225).
운영 반영 여부는 미결정 — LegalMALR-lite/RAPTOR-lite는 쿼리마다 로컬 LLM
추론이 추가로 필요하다(레이턴시·GPU 상주 비용 트레이드오프). 상세 결과·caveat는
survey 문서 참고.
"""

from backend.api.schemas import LegalBasis
from backend.db.connection import get_conn
from backend.db.loader import embed_texts, get_embedder
from backend.utils import load_logger

logger = load_logger("retrieval.log")

# RRF 등 랭킹 융합 단계에서 후보를 넉넉히 확보하기 위한 Dense/Sparse 각각의 기본 후보 수.
# top_k(최종 반환 개수)보다 넉넉히 크게 잡아야 두 검색 결과가 겹치지 않을 때도
# 융합 순위가 의미를 가진다.
_CANDIDATE_K = 8

_embedder = None
_law_names_cache: list[str] | None = None


def _get_cached_embedder():
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
    return _embedder


def get_law_names() -> list[str]:
    """chunks 테이블(source='law')에 실제로 존재하는 법령명 목록(캐싱).

    backend.agents.query_router.route_law_names()가 EXAONE에게 "이 목록 중에서
    골라라"는 선택지로 넘긴다 — DB에 없는 법령명을 예측해서 필터가 텅 비는 걸 방지.
    """
    global _law_names_cache
    if _law_names_cache is None:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT DISTINCT metadata->>'law_name' FROM chunks WHERE source = 'law'")
            _law_names_cache = [row[0] for row in cur.fetchall() if row[0]]
    return _law_names_cache


def _truncate(text: str, max_len: int = 100) -> str:
    text = text.strip()
    return text[:max_len] + "..." if len(text) > max_len else text


def _law_to_legal_basis(metadata: dict, text: str) -> LegalBasis:
    metadata = metadata or {}
    article_no    = metadata.get("article_no", "")
    article_title = metadata.get("article_title", "")
    article = f"제{article_no}조" if article_no else ""
    if article_title:
        article = f"{article}({article_title})" if article else article_title
    return LegalBasis(law=metadata.get("law_name", ""), article=article, description=_truncate(text))


def _precedent_to_legal_basis(metadata: dict, text: str) -> LegalBasis:
    metadata = metadata or {}
    court     = metadata.get("court", "")
    case_name = metadata.get("case_name", "")
    law = f"{court} 판례" if court else "판례"
    if case_name:
        law = f"{law}({case_name})"
    return LegalBasis(law=law, article=metadata.get("case_number", ""), description=_truncate(text))


def candidate_to_legal_basis(candidate: dict) -> LegalBasis:
    """fetch_candidates()가 반환한 후보 dict 하나를 LegalBasis로 변환한다."""
    if candidate["source"] == "law":
        return _law_to_legal_basis(candidate["metadata"], candidate["text"])
    return _precedent_to_legal_basis(candidate["metadata"], candidate["text"])


def _search_dense(cur, sources: list[str], vec_literal: str, top_k: int) -> list[tuple[str, str, dict, str]]:
    cur.execute(
        """
        SELECT chunk_id, source, metadata, text
        FROM chunks
        WHERE source = ANY(%s)
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (sources, vec_literal, top_k),
    )
    return cur.fetchall()


def _search_sparse(
    cur, sources: list[str], query_text: str, top_k: int, similarity_threshold: float = 0.1,
) -> list[tuple[str, str, dict, str]]:
    """pg_trgm 문자 3-gram 유사도 기반 어휘(Sparse) 검색.

    기본 유사도 임계값(pg_trgm.similarity_threshold=0.3)은 조항 원문처럼 긴
    쿼리 대비 짧은 법조문 텍스트를 매칭하기엔 너무 엄격해서, 세션 단위로
    낮춰 재현율을 확보한다(정밀도는 이후 RRF 융합·Dense 신호가 보완).
    """
    cur.execute("SET pg_trgm.similarity_threshold = %s", (similarity_threshold,))
    cur.execute(
        """
        SELECT chunk_id, source, metadata, text
        FROM chunks
        WHERE source = ANY(%s) AND text %% %s
        ORDER BY similarity(text, %s) DESC
        LIMIT %s
        """,
        (sources, query_text, query_text, top_k),
    )
    return cur.fetchall()


def _search_dense_one_law(cur, law_name: str, vec_literal: str, top_k: int) -> list[tuple[str, str, dict, str, float]]:
    """단일 law_name 파티션 안에서만 dense 검색 — 실제 코사인 유사도 점수도 같이 반환한다
    (파티션을 넘어 병합할 때 순위가 아니라 점수로 비교해야 하기 때문, 아래 docstring 참고)."""
    cur.execute(
        """
        SELECT chunk_id, source, metadata, text, 1 - (embedding <=> %s::vector) AS score
        FROM chunks
        WHERE source = 'law' AND metadata->>'law_name' = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (vec_literal, law_name, vec_literal, top_k),
    )
    return cur.fetchall()


def _search_sparse_one_law(cur, law_name: str, query_text: str, top_k: int, similarity_threshold: float = 0.1) -> list[tuple[str, str, dict, str, float]]:
    cur.execute("SET pg_trgm.similarity_threshold = %s", (similarity_threshold,))
    cur.execute(
        """
        SELECT chunk_id, source, metadata, text, similarity(text, %s) AS score
        FROM chunks
        WHERE source = 'law' AND metadata->>'law_name' = %s AND text %% %s
        ORDER BY score DESC
        LIMIT %s
        """,
        (query_text, law_name, query_text, top_k),
    )
    return cur.fetchall()


def _search_law_partitioned(
    cur, vec_literal: str, query_text: str, top_k: int, sparse_similarity_threshold: float, law_names: list[str],
) -> tuple[list[tuple], list[tuple]]:
    """법령별로 따로 top_k를 확보한 뒤 실제 유사도 점수로 재병합한다.

    law_names 전체를 하나의 SQL IN절로 묶어 공통 top_k를 나눠 쓰게 하면, 법령
    코퍼스가 43청크(약관규제법)~1,305청크(민법)로 불균형해서 정답이 소수 법령에
    있어도 대형 법령 후보에 밀려난다 — 도메인 필터링 없는 RRF와 똑같은 문제를
    법령 2개 규모로 재현할 뿐이다. 그래서 법령마다 **독립적으로** top_k를 확보해
    희소 법령도 반드시 후보 풀에 들어오게 보장한 뒤, 파티션 경계 없이 실제
    코사인/trigram 점수로 다시 정렬한다(순위가 아니라 점수 — 파티션마다 밀도가
    달라 순위만으론 비교가 안 됨). 이 방식으로 실측된 결과는
    `backend/eval/retrieval_alternatives_survey.md`의 RAPTOR-lite(RRF 8%→33%,
    p<0.0001) — 이 함수는 그 실험(raptor_lite_compare.py)의 로직을 프로덕션에
    그대로 옮긴 것이다.
    """
    all_dense, all_sparse = [], []
    for law_name in law_names:
        all_dense.extend(_search_dense_one_law(cur, law_name, vec_literal, top_k))
        all_sparse.extend(_search_sparse_one_law(cur, law_name, query_text, top_k, sparse_similarity_threshold))

    dense_sorted  = [row[:4] for row in sorted(all_dense, key=lambda r: r[4], reverse=True)]
    sparse_sorted = [row[:4] for row in sorted(all_sparse, key=lambda r: r[4], reverse=True)]
    return dense_sorted, sparse_sorted


def _reciprocal_rank_fusion(
    dense_rows: list[tuple[str, str, dict, str]],
    sparse_rows: list[tuple[str, str, dict, str]],
    k: int = 60,
) -> list[dict]:
    """Dense/Sparse 두 랭킹을 chunk_id 기준 RRF로 합쳐 점수 내림차순으로 정렬한다.

    RRF는 두 검색 방식의 점수 스케일(코사인 거리 vs 문자열 유사도)이 서로 달라
    직접 비교할 수 없을 때 널리 쓰이는 융합 방법이다 — 각 결과의 "순위"만 보고
    score = Σ 1/(k+rank)로 합산하므로 점수 정규화가 필요 없다. 어느 한쪽에만
    나온 항목도 그쪽 순위로 점수를 받아 후보에서 빠지지 않는다.

    반환하는 각 dict의 "in_both"는 Dense·Sparse 양쪽에서 다 나온 후보인지를
    나타낸다 — Evidence Verification Agent가 근거 신뢰도 신호로 쓴다(사전 실험에서
    "양쪽 다 동의한 후보"가 "한쪽만 찾은 후보"보다 정답 적중률이 뚜렷이 높았음:
    24.5% vs 14.9%, clean_clauses 478건 기준).
    """
    scores: dict[str, float] = {}
    payload: dict[str, dict] = {}
    dense_ids  = {row[0] for row in dense_rows}
    sparse_ids = {row[0] for row in sparse_rows}

    for rank, (chunk_id, source, metadata, text) in enumerate(dense_rows):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        payload[chunk_id] = {
            "chunk_id": chunk_id, "source": source, "metadata": metadata, "text": text,
            "in_both": chunk_id in sparse_ids,
        }
    for rank, (chunk_id, source, metadata, text) in enumerate(sparse_rows):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        payload.setdefault(chunk_id, {
            "chunk_id": chunk_id, "source": source, "metadata": metadata, "text": text,
            "in_both": chunk_id in dense_ids,
        })

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [payload[cid] for cid in ranked_ids]


def fetch_candidates(
    query_text: str,
    top_k_per_source: int = _CANDIDATE_K,
    sparse_similarity_threshold: float = 0.1,
    unified: bool = False,
    law_names: list[str] | None = None,
) -> dict[str, list[dict]]:
    """법령·판례 검색 후보 풀을 소스별로 반환한다(Evidence Selection Agent가 재랭킹할 원본).

    각 후보 dict: {chunk_id, source, metadata, text, in_both}.
    unified=True면 law/precedent를 나누지 않고 한 번에 검색한 뒤 소스별로 재분류한다
    (Evidence Verification의 마지막 재검색 단계 — 소스를 나눠서 못 찾았으면 안 나눠서라도
    찾아본다는 전략).

    law_names가 주어지면 law 소스 검색만 그 법령들로 제한한다(unified 모드에선 무시 —
    통합 재검색은 범위를 넓히는 마지막 시도이므로 좁히는 필터와 상충한다).
    backend.agents.query_router.route_law_names()가 예측한 값을 넘긴다 — 법령
    코퍼스가 43청크(약관규제법)~1,305청크(민법)로 극단적으로 불균형해서, 필터
    없이 전체를 한 풀에서 경쟁시키면 소수 법령이 밀리는 문제를 완화한다
    (`backend/eval/retrieval_alternatives_survey.md`의 RAPTOR-lite 실측:
    RRF 8%→33%, McNemar p<0.0001).
    """
    if not query_text.strip():
        return {"law": [], "precedent": []}

    try:
        embedder = _get_cached_embedder()
        query_vec = embed_texts(embedder, [query_text], prefix="query: ")[0]
        vec_literal = "[" + ",".join(repr(x) for x in query_vec) + "]"

        with get_conn() as conn, conn.cursor() as cur:
            if unified:
                sources = ["law", "precedent"]
                dense  = _search_dense(cur, sources, vec_literal, top_k_per_source)
                sparse = _search_sparse(cur, sources, query_text, top_k_per_source, sparse_similarity_threshold)
                fused = _reciprocal_rank_fusion(dense, sparse)
                result = {"law": [], "precedent": []}
                for c in fused:
                    result.setdefault(c["source"], []).append(c)
            else:
                result = {}
                for source in ("law", "precedent"):
                    if source == "law" and law_names:
                        dense, sparse = _search_law_partitioned(
                            cur, vec_literal, query_text, top_k_per_source, sparse_similarity_threshold, law_names,
                        )
                    else:
                        dense  = _search_dense(cur, [source], vec_literal, top_k_per_source)
                        sparse = _search_sparse(cur, [source], query_text, top_k_per_source, sparse_similarity_threshold)
                    result[source] = _reciprocal_rank_fusion(dense, sparse)
    except Exception as e:
        logger.warning(f"법령/판례 후보 검색 실패, 빈 결과 반환: {e}")
        return {"law": [], "precedent": []}

    return result


def search_legal_basis(query_text: str, top_k: int = 2) -> list[LegalBasis]:
    """조항 텍스트(또는 근거 문구)로 관련 법령 조문·판례를 Hybrid 검색한다.

    소스별 RRF 융합 결과 상위 top_k를 그대로 반환한다(재랭킹 없음 — 재랭킹은
    backend.agents.evidence_selection_agent가 fetch_candidates()로 후보 풀을
    직접 받아서 처리한다). 이 함수는 하위호환·간단한 조회용으로 남겨둔다.
    """
    candidates = fetch_candidates(query_text, top_k_per_source=_CANDIDATE_K)
    law_top       = candidates.get("law", [])[:top_k]
    precedent_top = candidates.get("precedent", [])[:top_k]
    return [candidate_to_legal_basis(c) for c in law_top + precedent_top]


def search_similar_labeled_clauses(
    query_text: str, top_k: int = 5, exclude_chunk_id: str | None = None,
) -> list[dict]:
    """clean_clauses에서 의미상 가장 비슷한 검증 완료 조항을 찾는다(Red-team Agent용).

    Judgment Agent가 방금 내린 판단이 기존에 검증된 유사 사례들의 라벨과
    일관적인지 확인하는 데 쓴다. FB-Check로 검증된 데이터만 대상으로 하므로
    (clean_clauses, 478건) 비교 기준 자체의 신뢰도가 높다.
    """
    if not query_text.strip():
        return []

    try:
        embedder = _get_cached_embedder()
        query_vec = embed_texts(embedder, [query_text], prefix="query: ")[0]
        vec_literal = "[" + ",".join(repr(x) for x in query_vec) + "]"

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, domain, risk_level,
                       COALESCE(NULLIF(evidence_span, ''), text) AS span,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM clean_clauses
                WHERE chunk_id != %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vec_literal, exclude_chunk_id or "", vec_literal, top_k),
            )
            rows = cur.fetchall()
    except Exception as e:
        logger.warning(f"유사 라벨 조항 검색 실패, 빈 결과 반환: {e}")
        return []

    return [
        {"chunk_id": chunk_id, "domain": domain, "risk_level": risk_level, "text": text, "similarity": float(sim)}
        for chunk_id, domain, risk_level, text, sim in rows
    ]
