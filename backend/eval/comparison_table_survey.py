# backend/eval/comparison_table_survey.py
"""의결서 **대조표** 조사 — 보정셋을 만들 수 있는지, 규모가 얼마인지.

## 왜 필요한가

배포 임계값이 어긋나 있다(gold 교차적합 +6.2%p 이김 vs dev 임계값 +2.5%p 미판정,
차이 3.9%p). 닫으려면 **배포 분포를 닮은 보정셋**이 필요한데 지금 가진 것 셋이 다 아니다:

    dev              빈 라벨 67%   표준계약서 과다
    gold             빈 라벨  0%   FTC 전용
    표준계약서 holdout  빈 라벨 97%   반대 극단
    실제 계약서        ?            한 문서에 대부분 정상 + 소수 위반   ← 셋 다 아니다

의결서 대조표(변경 전 = 위반 / 변경 후 = 비위반)는 **같은 문서·같은 문체에서 위반과
비위반이 함께 나오는 유일한 재료**다.

## 사전 조사에서 확인된 형태 (표본 60건)

    대조표 신호어 2회+   5.0%  →  1,163건 환산 ~58건
    pdfplumber 표 객체  16.7%  →  ~194건
    열 병합(구분자 없음)  0.0%  →  0건    ← 최악을 가정했는데 없었다

깔끔하게 둘 중 하나로 나온다:

    표 객체    ['수정전약관', '제11조(계약해지 사유) ...']
              ['수정후약관', '제11조(계약해지 사유) ...']
    평문 마커   수정전 【회원가입계약서】 3. 갑은 ...
              수정후 【멤버쉽 등록계약서】 4. 등록비는 ...

## 무엇을 세나

    ① 사건당 (변경 전, 변경 후) 조항 쌍 수
    ② 그 사건의 근거_법령 개수      ← **1개짜리만 쓸 수 있다**
    ③ 파싱 실패 사유

②가 필요한 이유: 변경 전 조항이 어느 조를 위반하는지는 사건의 `근거_법령`에서 오는데,
근거가 2개+면 조항 하나에 여러 조가 귀속돼(gold에서 확인한 그 문제) 임계값을 잘못된
표본으로 튜닝하게 된다.

## 판정 (사전 등록, `prompt_block_ablation.md` 참고)

    유효 쌍 200개 이상  →  조별 임계값 8개를 교차적합으로 보정
    100개 미만         →  전역 임계값 1개로 축소
    100~200개          →  전역 1개 + support 큰 조(제6·8·9조)만 조별

실행:
    .venv/bin/python -m backend.eval.comparison_table_survey --scan 500 --sample 20
"""

import argparse
import json
import random
import re
from collections import Counter

from backend.labeling.articles import normalize
from backend.scripts.parse_ftc_case_pdf import extract_text_from_pdf
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("comparison_table_survey.log")

FTC_JSON = PROJECT_ROOT / "data/raw/ftc_cases/ftc_cases_parsed.json"
PDF_DIR  = PROJECT_ROOT / "data/raw/ftc_cases/pdfs"
OUT_PATH = PROJECT_ROOT / "data/eval/comparison_table_survey.json"

_BEFORE = re.compile(r"(수\s*정\s*전|변\s*경\s*전|시\s*정\s*전|현\s*행)\s*(약\s*관)?")
_AFTER  = re.compile(r"(수\s*정\s*후|변\s*경\s*후|시\s*정\s*후|개\s*정\s*안?)\s*(약\s*관)?")
_SIGNAL = re.compile(r"(변\s*경|수\s*정|시\s*정|개\s*정)\s*(전|후|안)")
_MIN_CHARS = 30


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def pairs_from_tables(tables: list) -> list[tuple[str, str]]:
    """`['수정전약관', 조항텍스트]` / `['수정후약관', 조항텍스트]` 형태의 표에서 쌍을 뽑는다."""
    out = []
    for t in tables:
        before = after = None
        for row in t:
            cells = [str(c or "") for c in row]
            head, body = _norm(cells[0]), max(cells[1:], key=len, default="") if len(cells) > 1 else ""
            if len(body.strip()) < _MIN_CHARS:
                continue
            if _BEFORE.search(head):
                before = body.strip()
            elif _AFTER.search(head):
                after = body.strip()
        if before and after and _norm(before) != _norm(after):
            out.append((before, after))
    return out


def pairs_from_text(text: str) -> list[tuple[str, str]]:
    """평문에서 `수정전 … 수정후 …` 마커 사이를 잘라 쌍으로 만든다."""
    marks = []
    for m in _BEFORE.finditer(text):
        marks.append((m.start(), "B", m.end()))
    for m in _AFTER.finditer(text):
        marks.append((m.start(), "A", m.end()))
    marks.sort()
    out = []
    for i in range(len(marks) - 1):
        pos, kind, end = marks[i]
        nxt_pos, nxt_kind, nxt_end = marks[i + 1]
        if kind != "B" or nxt_kind != "A":
            continue
        before = text[end:nxt_pos].strip()
        after = text[nxt_end:marks[i + 2][0] if i + 2 < len(marks) else min(len(text), nxt_end + 1200)].strip()
        if len(before) >= _MIN_CHARS and len(after) >= _MIN_CHARS and _norm(before) != _norm(after):
            out.append((before[:1500], after[:1500]))
    return out


def survey_case(case: dict) -> dict:
    pdf = PDF_DIR / (case.get("pdf_파일") or "")
    doc = str((case.get("셀_데이터") or {}).get("사건번호") or case.get("사건명", ""))
    arts = {a for g in (case.get("근거_법령") or []) if (a := normalize(str(g)))}
    row = {"doc_id": doc, "n_articles": len(arts), "articles": sorted(arts),
           "n_violation_types": len(case.get("위반_유형") or [])}
    if not pdf.exists():
        return {**row, "ok": False, "reason": "PDF 없음", "n_pairs": 0}
    try:
        import pdfplumber
        text = extract_text_from_pdf(pdf)
        with pdfplumber.open(pdf) as f:
            tables = [t for p in f.pages[:30] for t in p.extract_tables()]
    except Exception as e:
        return {**row, "ok": False, "reason": f"PDF 열기 실패: {type(e).__name__}", "n_pairs": 0}

    if len(_SIGNAL.findall(text)) < 2:
        return {**row, "ok": False, "reason": "대조표 신호어 없음", "n_pairs": 0}

    tp, xp = pairs_from_tables(tables), pairs_from_text(text)
    pairs = tp or xp
    if not pairs:
        return {**row, "ok": False, "reason": "신호어는 있으나 쌍 추출 실패", "n_pairs": 0}
    return {**row, "ok": True, "reason": "", "n_pairs": len(pairs),
            "source": "table" if tp else "text",
            "sample": [{"before": pairs[0][0][:200], "after": pairs[0][1][:200]}]}


def main() -> None:
    ap = argparse.ArgumentParser(description="의결서 대조표 조사")
    ap.add_argument("--scan", type=int, default=500, help="후보를 찾기 위해 훑을 사건 수")
    ap.add_argument("--seed", type=int, default=42,
                    help="표집 시드. **무작위로 훑는다** — 원본 JSON이 연도순이라 "
                         "앞에서부터 자르면 특정 시대만 본다(앞 600건=2004~2009, 뒤=1995~2002). "
                         "실제로 그렇게 훑었다가 후보율을 0.5%%로 과소추정했다")
    ap.add_argument("--sample", type=int, default=20, help="쌍을 세어볼 후보 수")
    a = ap.parse_args()

    all_cases = json.loads(FTC_JSON.read_text(encoding="utf-8"))["사례"]
    cases = random.Random(a.seed).sample(all_cases, min(a.scan, len(all_cases)))
    logger.info(f"========== 대조표 조사 | 전체 {len(all_cases)}건 중 무작위 {len(cases)}건 "
                f"(seed={a.seed}) ==========")
    rows, scanned = [], 0
    for c in cases:
        scanned += 1
        r = survey_case(c)
        if r["ok"] or r["reason"] not in ("대조표 신호어 없음", "PDF 없음"):
            rows.append(r)
        if scanned % 100 == 0:
            logger.info(f"  [{scanned}/{len(cases)}] 후보 {sum(1 for x in rows if x['ok'])}건")
        if sum(1 for x in rows if x["ok"]) >= a.sample:
            break

    ok = [r for r in rows if r["ok"]]
    logger.info(f"  훑은 사건 {scanned}건 → 대조표 후보 {len(ok)}건 ({len(ok) / scanned * 100:.1f}%)")
    logger.info("  ----- ③ 실패 사유 -----")
    for k, v in Counter(r["reason"] for r in rows if not r["ok"]).most_common():
        logger.info(f"    {k:<28}{v:>4}건")
    if not ok:
        logger.warning("  후보가 없다 — --scan을 늘릴 것")
        return

    single = [r for r in ok if r["n_articles"] == 1]
    tot_pairs = sum(r["n_pairs"] for r in ok)
    eff_pairs = sum(r["n_pairs"] for r in single)
    logger.info("  ----- ① 쌍 수율 -----")
    logger.info(f"    전체 후보 {len(ok)}건 → {tot_pairs}쌍 (사건당 {tot_pairs / len(ok):.2f})")
    logger.info(f"    추출 경로 {dict(Counter(r['source'] for r in ok))}")
    logger.info("  ----- ② 근거_법령 개수 -----")
    logger.info(f"    1개 {len(single)}건 ({len(single) / len(ok) * 100:.1f}%) | "
                f"2개+ {len(ok) - len(single)}건")
    logger.info(f"    **유효 쌍**(근거 1개만) {eff_pairs}쌍 (사건당 {eff_pairs / max(1, len(single)):.2f})")

    rate = len(ok) / scanned
    est_cases = rate * 1163
    est_pairs = est_cases * (len(single) / len(ok)) * (eff_pairs / max(1, len(single)))
    logger.info("  ----- 전량(1,163건) 환산 -----")
    logger.info(f"    대조표 후보 약 {est_cases:.0f}건 → **유효 쌍 약 {est_pairs:.0f}개**")
    verdict = ("조별 임계값 8개를 교차적합으로 보정" if est_pairs >= 200 else
               "전역 임계값 1개로 축소" if est_pairs < 100 else
               "전역 1개 + support 큰 조(제6·8·9조)만 조별")
    logger.info(f"  → 사전 등록한 판정: **{verdict}**")

    save_json({"scanned": scanned, "candidates": len(ok), "candidate_rate": rate,
               "pairs_total": tot_pairs, "single_article_cases": len(single),
               "effective_pairs": eff_pairs, "est_cases_full": est_cases,
               "est_effective_pairs_full": est_pairs, "verdict": verdict, "rows": rows}, OUT_PATH)
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
