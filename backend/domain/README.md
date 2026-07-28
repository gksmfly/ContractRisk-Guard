# backend/domain/

Step 1: 법령·판례·해석례 원천 데이터에서 해지·책임제한 도메인과 관련 없는 문서를 걸러냅니다. 세 소스 모두 메타데이터·키워드 기반이며(임베딩·Dense Retrieval 미사용), 통과한 문서를 `data/raw/*` → `data/domain/*`로 복사합니다.

> ⚠️ `Claude.md`(프로젝트 루트)에는 판례·해석례 필터링이 Dense Retrieval(임베딩 유사도)도 함께 쓴다고 적혀 있으나, 실제 코드는 메타데이터/키워드 필터만 존재합니다. Dense 단계는 설계 초안에 있었으나 채택되지 않은 것으로 보입니다 — 재도입 논의 없이는 이 문서(코드 기준)를 신뢰하세요.

---

## 파일 목록

### `filter_law.py`
법령명이 `ALLOWED_LAW_NAMES`(민법·상법·약관의 규제에 관한 법률·할부거래법·방문판매법·전자상거래소비자보호법·소비자기본법 7개, `config.py`) 중 하나와 정확히 일치하거나 그 "시행령/시행규칙"이면 통과. 5,585건 → 16건.

### `filter_precedent.py`
1. `사건종류명`이 형사/가사/세무/특허/선거특별이면 제외
2. 사건명·판시사항·판결요지에 `PREC_KEYWORDS`(약관, 계약해지, 책임제한, 위약금, 면책 조항 등 9개, `config.py`) 중 하나라도 있으면 통과
3. 139,350건 → 1,995건

### `filter_interpretation.py`
안건명에 `TARGET_LAWS`(약관규제법·소비자기본법·전자상거래소비자보호법·방문판매법·할부거래법 등) 이름이 포함되면 통과. 8,666건 → 26건.

### `config.py`
`ALLOWED_LAW_NAMES`, `PREC_KEYWORDS`, 소스별 raw/domain 디렉토리 경로(env override 가능).

### `common.py`
`copy_domain_docs()` — 필터를 통과한 문서 ID 집합을 받아 `data/raw/*` → `data/domain/*`로 `shutil.copy2` 복사. 원본 파일이 없으면 경고 로그만 남기고 계속 진행.

### `__main__.py`
진입점.
```bash
python -m backend.domain                        # 전체(법령+판례+해석례)
python -m backend.domain --source law            # 특정 소스만
python -m backend.domain --source precedent
python -m backend.domain --source interpretation
```
결과는 `data/domain/filter_report.json`에 누적 저장(기존 결과 유지하며 갱신).

---

## 참고

`--device` 인자가 남아있지만(`filter_precedent`/`filter_interpretation` 함수 시그니처에 `device` 파라미터 존재) 현재 두 함수 다 임베딩을 쓰지 않아 실질적으로 무시됩니다 — Dense Retrieval이 있던 시절의 흔적으로 보입니다.
