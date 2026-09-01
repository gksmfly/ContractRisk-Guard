# backend/eval/smoke_check.py
"""
전량 라벨링 go/no-go 판정 — `fb_check --limit 300` 결과를 채점한다.

## 왜 별도 스크립트가 아니라 채점만 하는가

스모크의 목적은 "전량을 돌려도 되는가"다. 그런데 `label_pilot`으로 재면 **전량이 탈 경로를
안 재는 것**이다 — resume, ERROR 재시도, 깨진 줄 처리, CLEAN/NOISE 판정, 파일 쓰기가 전부
`backend.fb_check` 쪽에 있고 `label_pilot`에는 없다.

그래서 스모크는 실제 파이프라인을 300건에서 멈춰 돌린다:

    python -m backend.fb_check --model gpt-4o --limit 300   # 전량과 같은 경로
    python -m backend.eval.smoke_check                      # 이 스크립트로 채점
    python -m backend.fb_check --model gpt-4o               # 통과하면 resume이 301번부터 이어받는다

`--model`은 필수다 — 예전에 이 스모크가 `.env`의 gpt-4o-mini로 돌아 빈 배열 63%가 나왔다.

**스모크 비용이 0이 된다** — 300건이 버려지는 게 아니라 전량의 첫 300건이 된다.
그리고 resume이 실전에서 한 번 검증되고 들어간다(중간에 Ctrl+C를 넣어보면 깨진 줄
처리까지 확인된다).

## 판정 기준 (실행 전 확정)

    단일조항 구간   건당 편차 ≤ 0.72        B 기준선 0.48 (n=100에서 총편차 48)
                    빈 배열  ≤ 30%          B 기준선 10%
    다조항 구간     사건 단위 union hit@any  ← **기준선 없음. 이번이 최초 측정**
                    빈 배열  ≤ 30%
    공통            파싱 실패(JSON 깨짐) ≤ 2%
                    ERROR 레코드 ≤ 5%

    통과 → 그대로 `python -m backend.fb_check`로 이어서 전량
    위반 → 중단하고 해당 구간 표본을 눈으로 확인

**총편차는 건당 평균으로 정규화한다.** Σ|pred−gold|는 표본 크기에 비례하므로 n=100에서
얻은 48을 n=150 구간에 그대로 대면 품질이 같아도 72가 나와 문턱에 걸린다.

**다조항은 사건 단위 union으로 채점한다.** gold(`근거_법령`)가 사건 단위라 조항별 채점이
불공정하다. 한 사건의 모든 조항 예측을 합쳐 그 사건의 gold와 비교하면 정의가 일치하고,
"이 사건의 위반 조를 빠짐없이 잡았는가"를 직접 잰다. 빈 배열 비율만으로는 "형식은 멀쩡한데
내용이 엉망"인 경우를 못 잡는다.

다조항 union hit@any에는 **문턱을 걸지 않는다** — 기준선이 없는데 문턱을 지어내면 그게 또
근거 없는 상수가 된다. 숫자를 받아 기록하고 단일 구간 대비 얼마나 낮은지를 보고 판단한다.

실행:
    .venv/bin/python -m backend.eval.smoke_check
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from backend.labeling.articles import normalize
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("smoke_check.log")

RESULTS_PATH = PROJECT_ROOT / "data/fb_check/fb_check_results.jsonl"
FTC_PATH     = PROJECT_ROOT / "data/raw/ftc_cases/ftc_cases_parsed.json"
OUT_PATH     = PROJECT_ROOT / "data/eval/smoke_check_report.json"

# ── 판정 문턱 ────────────────────────────────────────────────────────────────
# **환경변수로 빼지 않는다.** env로 빼는 순간 결과를 보고 문턱을 조정할 수 있게 되어
# 사전 등록의 의미가 사라진다(`article_gold_eval.py`가 임계값 없으면 에러로 중단하는 것과
# 같은 이유). 다만 **유도 과정을 코드에 남겨서** 매직 넘버가 되지 않게 한다 —
# `_FINAL_K = 2`, `[:20]`, `_SIMILARITY_THRESHOLD = 0.75`가 출처를 잃고 조용히 낡았던 전례.
#
# 다섯 값 중 **측정된 것은 하나뿐이고 넷은 임의다.** 문턱이 필요하니 임의여도 되지만,
# 임의라고 적혀 있어야 6개월 뒤에 "2%는 어디서 나온 값이지?"가 안 된다.

# [측정] B 블록(fwd-v5-ordered-summaryblock) 단일조항 100건 실측
#        출처: backend/eval/prompt_block_ablation.md (2026-08-22, 총편차 48 / n=100)
_BASELINE_DEV_PER_RECORD = 0.48
# [임의] 노이즈 바닥은 Δ2(=0.02/건)로 측정됐다. 1.5배는 그것과 무관한 여유폭으로,
#        다조항 구간이 구조적으로 나쁠 것을 감안해 고른 값이지 측정된 값이 아니다.
_DEV_TOLERANCE = 1.5
_MAX_DEV_PER_RECORD = _BASELINE_DEV_PER_RECORD * _DEV_TOLERANCE   # 0.72

# [정정 2026-08-23] **빈 배열 게이트를 출처별로 분리한다.**
#
# 사유: 조건 ①에서 음성 표본(표준계약서)을 의도적으로 포함시켰으므로, 전체 빈배열률은
#       음성 표본 비율에 따라 **기계적으로** 변한다. 단일 문턱은 두 모집단을 섞어
#       해석 불가능한 값을 만든다.
#
#     gpt-4o 100건 실측:  ftc_case 8/56(14.3%) · standard_contract 31/44(70.5%) → 합계 39%
#     파일럿(FTC만 100건) 10.0%와 비교하면 FTC 쪽은 사실상 같고, 29%p 격차는 전부 구성 차이다.
#
# ⚠️ 이것은 **결과를 보고 한 게이트 재해석**이다. 그 점을 숨기지 않는다.
#    정당하다고 판단한 근거는 하나뿐이다 — FTC 14.3%가 **사전에 존재하던 독립 참조값**
#    (파일럿 gpt-4o 10%)과 수렴한다. 문턱을 새로 지어내 통과시키는 게 아니라 이미 있던
#    숫자와 만나는 것이고, FTC가 35%였다면 중단이 맞았다.
#    그리고 원래 30% 문턱의 논리("학습 데이터의 3분의 1이 위반 없음이 되면 안 된다")는
#    **위반이 있으리라 기대하는 조항**에 대한 진술이지, 일부러 넣은 음성 표본을 포함한
#    값이 아니었다. 완화가 아니라 분리다.
#
# ※ 이 분리가 **출처 교락을 승인하는 것은 아니다.** FTC 14% vs 표준계약서 70%는
#   출처가 라벨을 상당 부분 예측한다는 뜻이고, 그건 조건 ③(`confound_articles.py`)이
#   따로 재야 할 문제다. 게이트 통과와 교락 해결은 별개다.
#
# [임의] FTC 문턱 30%는 그대로 둔다(B 기준선 10%의 3배). 표준계약서는 문턱을 걸지 않는다 —
#        높은 게 정상이고 **낮으면 오히려 의심**해야 한다(음성 표본이 아니게 되므로).
_MAX_EMPTY_RATE = 0.30                    # ftc_case 전용
_EMPTY_GATE_SOURCES = ("ftc_case",)       # 이 출처만 게이트에 건다
# [임의] JSON 파싱 실패. 재시도 로직이 있으니 이보다 높으면 프롬프트/모델 문제로 본다.
_MAX_PARSE_FAIL = 0.02
# [임의] ERROR 레코드(호출 3회 재시도 후 최종 실패). resume이 재시도하므로 치명적이진 않다.
_MAX_ERROR_RATE = 0.05

_THRESHOLD_PROVENANCE = {
    "max_dev_per_record": f"측정 기준선 {_BASELINE_DEV_PER_RECORD} "
                          f"(prompt_block_ablation.md, B/n=100) × {_DEV_TOLERANCE} (임의)",
    "max_empty_rate":     "임의. B 기준선 0.10의 3배. **ftc_case에만 적용** — "
                          "표준계약서는 의도한 음성 표본이라 높은 게 정상(2026-08-23 정정)",
    "max_parse_fail":     "임의. 재시도 로직이 있으므로 이보다 높으면 프롬프트/모델 문제",
    "max_error_rate":     "임의. resume이 ERROR를 재시도하므로 치명적이지 않음",
}


def _doc_of(chunk_id: str) -> str:
    parts = str(chunk_id).split(":")
    return ":".join(parts[:2]) if len(parts) >= 3 else str(chunk_id)


def load_results(path: Path) -> tuple[list[dict], int]:
    """결과 JSONL을 읽는다. 깨진 줄 수도 함께 낸다(중단 시점의 잘린 기록)."""
    rows, broken = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                broken += 1
    # chunk_id마다 마지막 기록만(ERROR 재시도로 중복될 수 있음)
    by = {r.get("chunk_id"): r for r in rows}
    return list(by.values()), broken


def load_gold_by_doc() -> dict[str, set[str]]:
    """사건번호 → 약관규제법 조 집합. `chunk_id`의 문서 부분과 맞춘다."""
    cases = json.loads(FTC_PATH.read_text(encoding="utf-8"))["사례"]
    gold: dict[str, set[str]] = {}
    for c in cases:
        doc = str((c.get("셀_데이터") or {}).get("사건번호") or c.get("사건명", ""))
        arts = {a for g in (c.get("근거_법령") or []) if (a := normalize(str(g)))}
        if doc and arts:
            gold[f"ftc_case:{doc}"] = arts
    return gold


def _articles(r: dict) -> list[str]:
    """확정 조 목록 — 합의(F∩V) 우선, 없으면 forward."""
    a = r.get("agreed_articles")
    return a if a is not None else (r.get("forward_articles") or [])


def label_divergence(rows: list[dict]) -> dict:
    """CLEAN 중 `forward_articles ≠ agreed_articles`인 비율 — **Low 예외 경로 점검.**

    판정 규칙에 예외가 하나 있다(`fb_check/__main__.py`):

        if forward_label != "Low" and not agreed_articles:
            NOISE(article_mismatch)

    즉 **Low 레코드는 조 합의를 건너뛴다.** "위반 없음 표본을 살린다"는 의도는 맞지만
    부작용이 있다:

        표준계약서 조항, forward=[제9조] risk=Low, verify=[]
          → CLEAN. forward_articles=[제9조] 인데 agreed_articles=[]
          → 학습에서 forward를 쓰면 "제9조", agreed를 쓰면 "위반 없음". **정반대다.**

    표준계약서 2,036건이 대부분 Low로 들어오므로 이 경로가 학습 데이터의 절반을
    좌우할 수 있다. 비율이 작으면(<5%) 무시하고 가고, 크면 전량 전에 Low 예외 조건을
    다시 본다.
    """
    clean = [r for r in rows if r.get("status") == "CLEAN"]
    if not clean:
        return {"n_clean": 0}
    diverged = [r for r in clean
                if sorted(r.get("forward_articles") or []) != sorted(r.get("agreed_articles") or [])]
    low_gap = [r for r in diverged if r.get("forward_label") == "Low"]
    # 라벨이 실제로 갈리는 건 forward가 비어 있지 않은 경우다(agreed=[] 이면 정반대가 된다)
    flips = [r for r in low_gap if r.get("forward_articles")]
    # **출처별로 나눈다.** agreed를 쓰면 표준계약서 → 빈 배열이 다시 결정적 함수에
    # 가까워진다 — 조건 ①에서 provenance 찍기를 금지하고 GPT를 태운 이유가 그 결정성을
    # 깨는 거였는데, 라벨 선택 단계에서 되살아나는 셈이다. flip이 표준계약서에 몰려
    # 있으면 `--label-source`는 정밀/재현 트레이드오프가 아니라 **교락 조절 손잡이**다.
    by_src_clean = Counter(r.get("source", "?") for r in clean)
    by_src_flip = Counter(r.get("source", "?") for r in flips)
    flip_by_source = {s: {"flips": by_src_flip.get(s, 0), "clean": n_s,
                          "rate": by_src_flip.get(s, 0) / n_s if n_s else 0.0}
                      for s, n_s in by_src_clean.items()}
    return {
        "n_clean": len(clean),
        "diverged": len(diverged), "diverged_rate": len(diverged) / len(clean),
        "low_exception": len(low_gap),
        "label_flips": len(flips), "flip_rate": len(flips) / len(clean),
        "flip_by_source": flip_by_source,
        "flip_examples": [
            {"chunk_id": r.get("chunk_id"), "source": r.get("source"),
             "forward": r.get("forward_articles"), "agreed": r.get("agreed_articles"),
             "verify": r.get("verify_articles")}
            for r in flips[:5]
        ],
    }


def score_segment(rows: list[dict], gold: dict[str, set[str]], label: str) -> dict:
    """구간별 지표. 건당 편차는 표본 크기로 정규화한다."""
    n = len(rows)
    if not n:
        return {"n": 0}
    pred_dist = Counter(a for r in rows for a in _articles(r))
    gold_dist: Counter = Counter()
    for doc in {_doc_of(r.get("chunk_id", "")) for r in rows}:
        for a in gold.get(doc, ()):
            gold_dist[a] += 1
    dev = sum(abs(pred_dist.get(a, 0) - gold_dist.get(a, 0))
              for a in set(pred_dist) | set(gold_dist))
    empty = sum(1 for r in rows if not _articles(r)) / n

    # 사건 단위 union — gold가 사건 단위이므로 정의가 일치한다
    by_doc: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_doc[_doc_of(r.get("chunk_id", ""))].update(_articles(r))
    scored = [(d, p) for d, p in by_doc.items() if d in gold]
    hit = sum(1 for d, p in scored if p & gold[d])
    return {
        "n": n, "n_docs": len(by_doc), "n_docs_with_gold": len(scored),
        "total_deviation": dev, "dev_per_record": dev / n,
        "empty_rate": empty,
        "union_hit_any": hit / len(scored) if scored else None,
        "pred_dist": dict(pred_dist.most_common()),
        "gold_dist": dict(gold_dist.most_common()),
    }


def main() -> None:
    rows, broken = load_results(RESULTS_PATH)
    gold = load_gold_by_doc()
    n = len(rows)
    logger.info(f"========== 스모크 채점 ({n}건) ==========")
    logger.info(f"  문턱: 건당편차 ≤{_MAX_DEV_PER_RECORD:.2f}(측정 {_BASELINE_DEV_PER_RECORD}×{_DEV_TOLERANCE} 임의) "
                f"· 빈배열 ≤{_MAX_EMPTY_RATE:.0%}(임의) "
                f"· 파싱실패 ≤{_MAX_PARSE_FAIL:.0%}(임의) · ERROR ≤{_MAX_ERROR_RATE:.0%}(임의)")

    parse_fail = broken / max(n + broken, 1)
    err_rate = sum(1 for r in rows if r.get("status") == "ERROR") / max(n, 1)

    doc_sizes = Counter(_doc_of(r.get("chunk_id", "")) for r in rows)
    single = [r for r in rows if doc_sizes[_doc_of(r.get("chunk_id", ""))] == 1]
    multi  = [r for r in rows if doc_sizes[_doc_of(r.get("chunk_id", ""))] > 1]

    seg = {"단일조항": score_segment(single, gold, "단일조항"),
           "다조항":   score_segment(multi,  gold, "다조항")}

    logger.info(f"  {'구간':<10}{'건수':>6}{'문서':>6}{'건당편차':>10}{'빈배열':>9}{'union hit@any':>15}")
    for k, s in seg.items():
        if not s.get("n"):
            logger.info(f"  {k:<10}{'(없음)':>6}")
            continue
        u = f"{s['union_hit_any'] * 100:.1f}%" if s["union_hit_any"] is not None else "gold 없음"
        logger.info(f"  {k:<10}{s['n']:>6}{s['n_docs']:>6}{s['dev_per_record']:>10.2f}"
                    f"{s['empty_rate'] * 100:>8.1f}%{u:>15}")
    logger.info(f"  공통 — 파싱 실패 {parse_fail * 100:.2f}% (깨진 줄 {broken}) | ERROR {err_rate * 100:.2f}%")

    div = label_divergence(rows)
    if div.get("n_clean"):
        logger.info("  ----- forward vs agreed 라벨 분기 (Low 예외 경로 점검) -----")
        logger.info(f"    CLEAN {div['n_clean']}건 중 두 라벨이 다름 {div['diverged']}건 "
                    f"({div['diverged_rate'] * 100:.1f}%)")
        logger.info(f"    그중 Low 예외(조 합의 건너뜀) {div['low_exception']}건 | "
                    f"**라벨이 정반대로 갈리는 건** {div['label_flips']}건 ({div['flip_rate'] * 100:.1f}%)")
        for s, v in div.get("flip_by_source", {}).items():
            logger.info(f"      [{s}] flip {v['flips']}/{v['clean']} ({v['rate'] * 100:.1f}%)")
        for e in div["flip_examples"]:
            logger.info(f"      · {e['source']} forward={e['forward']} verify={e['verify']} → agreed={e['agreed']}")
        sc = div.get("flip_by_source", {}).get("standard_contract", {})
        if sc.get("rate", 0) > 0.10:
            logger.warning("    ⚠️ flip이 표준계약서에 몰려 있다 — --label-source가 정밀/재현이 아니라 "
                           "**교락 조절 손잡이**다. A/B 시 F1만 보지 말고 confound_articles를 양쪽 다 돌릴 것")
        if div["flip_rate"] > 0.05:
            logger.warning("    ⚠️ 5% 초과 — 전량 전에 Low 예외 조건을 다시 볼 것 "
                           "(학습에서 forward/agreed 중 무엇을 쓰냐로 라벨이 뒤집힌다)")

    fails = []
    if parse_fail > _MAX_PARSE_FAIL:
        fails.append(f"파싱 실패 {parse_fail * 100:.2f}% > {_MAX_PARSE_FAIL * 100:.0f}%")
    if err_rate > _MAX_ERROR_RATE:
        fails.append(f"ERROR {err_rate * 100:.2f}% > {_MAX_ERROR_RATE * 100:.0f}%")
    s = seg["단일조항"]
    if s.get("n") and s["dev_per_record"] > _MAX_DEV_PER_RECORD:
        fails.append(f"단일조항 건당편차 {s['dev_per_record']:.2f} > {_MAX_DEV_PER_RECORD}")

    # CLEAN 비율도 **출처별로** 본다. 전체 값은 두 모집단의 가중평균이라 읽을 수 없다 —
    # 표준계약서는 대부분 Low라 `forward_label != "Low"` 조건으로 조 합의 검사를 건너뛰고
    # 거의 전부 CLEAN을 통과한다. 즉 **표준계약서 비율이 높을수록 전체 CLEAN이 기계적으로
    # 올라간다.** 빈 배열 39%에서 겪은 것과 같은 구조다.
    #
    # 옛 파이프라인 31.3%와의 비교는 성립하지 않는다 — taxonomy·판정규칙·표본이 전부 다르다.
    # **진짜 지표는 FTC 쪽 CLEAN 비율**이다. 거기가 낮으면 forward와 verify가 자주 갈린다는
    # 뜻이고, 그게 라벨 품질 신호다.
    logger.info("  ----- CLEAN 비율 (출처별) -----")
    clean_by_source = {}
    for src_name in sorted({r.get("source", "?") for r in rows}):
        sub = [r for r in rows if r.get("source") == src_name]
        cl = [r for r in sub if r.get("status") == "CLEAN"]
        reasons = Counter(r.get("noise_reason", "?").split(":")[0]
                          for r in sub if r.get("status") == "NOISE")
        clean_by_source[src_name] = {"n": len(sub), "clean": len(cl),
                                     "clean_rate": len(cl) / len(sub),
                                     "noise_reasons": dict(reasons)}
        note = "  ← 진짜 지표" if src_name == "ftc_case" else "  (Low 예외로 높은 게 당연)"
        logger.info(f"    {src_name:<20}{len(sub):>5}건  CLEAN {len(cl) / len(sub) * 100:5.1f}%{note}")
        if reasons:
            logger.info(f"      NOISE 사유: {dict(reasons.most_common())}")

    # 빈 배열은 **출처별로** 본다. 표준계약서는 의도한 음성 표본이라 문턱을 걸지 않는다.
    logger.info("  ----- 빈 배열 (출처별) -----")
    by_source_empty = {}
    for src_name in sorted({r.get("source", "?") for r in rows}):
        sub = [r for r in rows if r.get("source") == src_name]
        empty = sum(1 for r in sub if not _articles(r)) / len(sub)
        by_source_empty[src_name] = {"n": len(sub), "empty_rate": empty}
        gated = src_name in _EMPTY_GATE_SOURCES
        mark = f"(게이트 ≤{_MAX_EMPTY_RATE:.0%})" if gated else "(문턱 없음 — 높은 게 정상, 낮으면 의심)"
        logger.info(f"    {src_name:<20}{len(sub):>5}건  빈배열 {empty * 100:5.1f}%  {mark}")
        if gated and empty > _MAX_EMPTY_RATE:
            fails.append(f"{src_name} 빈배열 {empty * 100:.1f}% > {_MAX_EMPTY_RATE * 100:.0f}%")

    if fails:
        logger.warning("  ❌ NO-GO — 중단하고 해당 구간 표본을 눈으로 확인할 것")
        for f in fails:
            logger.warning(f"     · {f}")
    else:
        logger.info("  ✅ GO — `python -m backend.fb_check`로 이어서 전량 실행 "
                    "(resume이 이 300건 뒤부터 이어받는다)")
        # `seg["다조항"]`이다. 예전에는 정의된 적 없는 `m`을 참조해 **NameError로 죽었다** —
        # 그것도 `else`(GO) 분기 안이라, 게이트를 통과했을 때만 터졌다. "✅ GO"는 이미
        # 찍힌 뒤라 운영자는 "GO + 트레이스백"을 보게 된다. 08-23 실행이 NO-GO였던 덕에
        # 여태 안 걸렸을 뿐이고, 하필 전량 실행 직전에 돌리라고 문서에 적어둔 자리다.
        # 빈 구간이면 score_segment가 {"n": 0}만 주므로 .get()은 None으로 안전하게 빠진다.
        multi = seg["다조항"]
        if multi.get("union_hit_any") is not None:
            logger.info(f"     ※ 다조항 union hit@any {multi['union_hit_any'] * 100:.1f}%는 최초 측정이라 "
                        f"문턱이 없다. 단일 구간과의 격차를 기록만 한다")

    save_json({"n": n, "segments": seg, "label_divergence": div,
               "empty_by_source": by_source_empty,
               "clean_by_source": clean_by_source, "parse_fail_rate": parse_fail,
               "error_rate": err_rate, "broken_lines": broken,
               # 어떤 자로 쟀는지 리포트만 보고도 알 수 있어야 한다
               # (`article_gold_eval.py`가 checkpoint_criterion을 문자열로 남기는 것과 같은 패턴)
               "thresholds": {"max_dev_per_record": _MAX_DEV_PER_RECORD,
                              "max_empty_rate": _MAX_EMPTY_RATE,
                              "max_parse_fail": _MAX_PARSE_FAIL,
                              "max_error_rate": _MAX_ERROR_RATE},
               "threshold_provenance": _THRESHOLD_PROVENANCE,
               "verdict": "NO-GO" if fails else "GO", "failures": fails}, OUT_PATH)
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
