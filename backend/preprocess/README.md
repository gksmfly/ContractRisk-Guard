# backend/preprocess/

Step 2: `data/domain/{law,case,commentary}/`의 필터링된 문서를 텍스트 정제 + 청킹해 `data/processed/{laws,precedents,interpretations}.jsonl`로 만듭니다. RAG(`backend/api/services/retrieval.py`)와 DB 적재(`backend/db/loader.py`)가 이 출력을 그대로 씁니다.

---

## 파일 목록

### `extractor.py`
소스별 원본 JSON에서 (텍스트, 메타데이터) 쌍을 뽑는다.
- `extract_law()`: 조문 단위로 그대로(청킹 없음). 장 제목("제N장 ...")·부칙은 제외.
- `extract_precedent()`: 판시사항/판결요지/판례내용 3개 섹션을 각각 별도 레코드로. 판례내용만 `clean_precedent_content()`(당사자 정보 제거)를 추가로 거침.
- `extract_interpretation()`: 회답+이유를 합쳐 레코드 1개.

`EXTRACTORS` 딕셔너리(`{"law": ..., "precedent": ..., "interpretation": ...}`)로 `__main__.py`가 소스별 함수를 선택.

### `cleaner.py`
- `clean_text()`: HTML 태그 제거, 전각 공백/nbsp 정규화, 원 문자(①②③) → "(1)(2)(3)" 변환, 낫표(「」) 제거, 공백 정규화.
- `clean_precedent_content()`: 판례내용 전용 — 【원고】【피고】 등 당사자 메타정보 블록을 정규식으로 먼저 제거한 뒤 `clean_text()` 적용.
- `split_chunks()`: 문장 경계(마침표·물음표·느낌표·개행) 기준 청킹. 문장 하나가 `chunk_size`보다 길면 강제로 문자 단위 슬라이딩 윈도우(overlap 적용)로 쪼갬. `min_chunk` 미만 청크는 버림.

### `__main__.py`
진입점.
```bash
python -m backend.preprocess
python -m backend.preprocess --source law
python -m backend.preprocess --chunk-size 512 --overlap 50 --min-chunk 100
```
- 법령은 청킹하지 않고 조문 그대로 저장(`chunk_index=0` 고정), 판례·해석례는 `split_chunks()` 적용
- `chunk_id` 형식: `{source}:{doc_id}:{rec_index}:{chunk_index}` — `rec_index`는 한 문서에서 나온 여러 레코드(예: 판례의 판시사항/판결요지/판례내용) 구분, `chunk_index`는 그 레코드 내 청크 순번
- 결과는 `data/processed/preprocess_report.json`에 소스별 문서 수/청크 수/스킵/오류 건수로 누적 저장

---

## 참고

`data/domain/{law,case,commentary}/`가 없으면(도메인 필터링을 먼저 안 돌렸으면) 해당 소스는 경고만 남기고 건너뜁니다 — 에러로 죽지 않습니다.
