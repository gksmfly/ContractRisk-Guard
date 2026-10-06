# backend/eval/style_transfer_fpr.py
"""문체(文體) 교락 통제 실험 — B4.

## 가설

표준계약서(정부 양식)는 상업적 약관과 문체가 다르다. 모델이 조항 내용이 아니라
문체를 인식해 표준계약서에서 오경보를 덜 낸다면, 문체를 바꾸면 오경보가 올라간다.

    H0: 문체 변환 후 오경보율 ≈ 변환 전
    Ha: 문체 변환 후 오경보율 > 변환 전

## 실험 설계

1. 홀드아웃 12문서 CLEAN 조항(articles=[] 273건)을 상업적 약관 문체로 변환
2. 변환 전후에 `article_v2`(배포 임계값 그대로)를 돌린다
3. McNemar 검정 — 조항마다 (전, 후) 플래그 쌍이 있다

## 문체 변환 규칙

- 갑·을 → "서비스 제공자·이용자" 또는 "회사·고객"
- 공문서 어미(~함, ~임) → 상업 약관 어미(~합니다, ~합니다)
- 조항 번호 제거(모델이 번호 패턴을 쓰면 안 된다)
- 실질 의무·금지·권한 내용은 **그대로 유지** — 의미를 바꾸면 비교가 무의미하다

실행:
    .venv/bin/python -m eval.style_transfer_fpr
    .venv/bin/python -m eval.style_transfer_fpr --limit 20  # 연기 테스트
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from backend.training.train_article import (
    load_article_records,
    split_negative_holdout,
)
from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("style_transfer_fpr.log")

CLEAN_PATH = PROJECT_ROOT / "data/fb_check/clean.jsonl"
OUT_PATH   = PROJECT_ROOT / "data/eval/style_transfer_fpr.json"

_SYSTEM = (
    "너는 한국어 법률 문서 편집자다. 주어진 조항의 실질 의미(의무·금지·권한·금액·기간 등)는 "
    "**한 글자도 바꾸지 않고**, 문체만 상업 서비스 약관 스타일로 바꾼다.\n\n"
    "규칙:\n"
    "1. '갑'·'을'·'甲'·'乙' → '회사'·'이용자' (또는 문맥에 맞게 '서비스 제공자'·'고객')\n"
    "2. '공급업자'·'대리점' 같은 B2B 용어 → '회사'·'이용자'\n"
    "3. 조문 번호(제N조, 제N항, N. 등) 제거\n"
    "4. 명사형 어미(~함, ~임, ~음)·공문서 어미 → 상업 약관 어미(~합니다, ~됩니다, ~ます 금지)\n"
    "5. 원문의 괄호·빈칸(( ), [ ])은 그대로 유지\n"
    "6. 답변은 변환된 조항 텍스트만 출력한다 — 설명·주석 없음"
)

_USER_TMPL = "다음 조항을 상업 약관 문체로 변환하라:\n\n{text}"


def _transform_batch(texts: list[str], client: any, model: str, bs: int = 20) -> list[str]:
    """GPT로 문체 변환. 실패한 건은 원문을 그대로 돌려준다."""
    import openai
    results: list[str] = []
    for i in range(0, len(texts), bs):
        chunk = texts[i: i + bs]
        logger.info(f"  변환 {i+1}~{i+len(chunk)}/{len(texts)}건…")
        for text in chunk:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": _USER_TMPL.format(text=text)},
                    ],
                    max_tokens=800,
                    temperature=0,
                )
                results.append(resp.choices[0].message.content.strip())
            except Exception as e:
                logger.warning(f"  변환 실패 (원문 유지): {e}")
                results.append(text)
        time.sleep(0.05)
    return results


def _run_model(texts: list[str]) -> list[bool]:
    """article_v2 배포 임계값으로 각 조항의 플래그 여부를 반환."""
    from backend.agents.judgment_agent import predict_articles
    flags = []
    for i, t in enumerate(texts):
        if i % 50 == 0:
            logger.info(f"  모델 추론 {i}/{len(texts)}…")
        flags.append(bool(predict_articles(t)))
    return flags


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """쌍 비교. a=before, b=after. 비교값: b가 울리고 a가 안 울린 건 vs 반대."""
    n01 = sum(1 for x, y in zip(a, b) if not x and y)  # before 0, after 1 (올라감)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)  # before 1, after 0 (내려감)
    from scipy.stats import binomtest
    # H0: n01 = n10 (양방향). Ha: n01 > n10 (문체 교락이면 변환 후 올라간다)
    binom = binomtest(n01, n01 + n10, 0.5, alternative="greater") if (n01 + n10) > 0 else None
    return {
        "n": len(a),
        "before_flagged": int(sum(a)),
        "after_flagged": int(sum(b)),
        "fpr_before": float(np.mean(a)),
        "fpr_after": float(np.mean(b)),
        "delta_pp": float(np.mean(b) - np.mean(a)),
        "n01_up": n01,
        "n10_down": n10,
        "mcnemar_p": float(binom.pvalue) if binom else None,
        "verdict": (
            f"p={binom.pvalue:.4f} — "
            + ("문체 교락 신호 있음" if binom and binom.pvalue < 0.05 else "귀무가설 기각 불가(교락 없음)")
        ) if binom else "불일치 쌍 없음(전후 동일)",
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=0, help="0=전부. 연기 테스트용")
    ap.add_argument("--skip-transform", action="store_true",
                    help="변환 건너뛰기 — 저장된 변환 결과 재사용")
    args = ap.parse_args(argv)

    import openai
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # ── 홀드아웃 CLEAN 조항 로드 ─────────────────────────────────────────────
    recs = load_article_records(CLEAN_PATH, label_source="agreed")
    _, holdout = split_negative_holdout(recs, 12, 42)
    # CLEAN = articles 빈 것(비위반 음성)
    clauses = [r for r in holdout
               if r.get("source") == "standard_contract" and not r.get("articles")]
    if args.limit:
        clauses = clauses[: args.limit]
    logger.info(f"★ 홀드아웃 CLEAN {len(clauses)}건 로드")
    texts_orig = [r["text"] for r in clauses]

    # ── 문체 변환 ────────────────────────────────────────────────────────────
    cache_path = OUT_PATH.with_suffix(".cache.json")
    if args.skip_transform and cache_path.exists():
        texts_xfer = json.loads(cache_path.read_text(encoding="utf-8"))
        logger.info(f"  캐시 재사용: {cache_path}")
    else:
        logger.info(f"  GPT({args.model}) 문체 변환 시작…")
        texts_xfer = _transform_batch(texts_orig, client, args.model)
        cache_path.write_text(json.dumps(texts_xfer, ensure_ascii=False, indent=1),
                              encoding="utf-8")
        logger.info(f"  변환 완료 → 캐시 저장: {cache_path}")

    # ── 모델 추론 (전·후) ────────────────────────────────────────────────────
    logger.info("  원문 추론…")
    flags_orig = _run_model(texts_orig)
    logger.info("  변환 후 추론…")
    flags_xfer = _run_model(texts_xfer)

    # ── 통계 ─────────────────────────────────────────────────────────────────
    stat = mcnemar(flags_orig, flags_xfer)
    logger.info("=" * 66)
    logger.info(f"★ 오경보 전: {stat['fpr_before']*100:.1f}%  →  후: {stat['fpr_after']*100:.1f}%  "
                f"(Δ{stat['delta_pp']*100:+.1f}%p)")
    logger.info(f"  McNemar 불일치: ↑{stat['n01_up']}건 ↓{stat['n10_down']}건  →  {stat['verdict']}")
    logger.info("=" * 66)

    # 예시: 변환 후 새로 울린 것들
    new_flags = [{"orig": o[:120], "xfer": x[:120]}
                 for o, x, fo, fx in zip(texts_orig, texts_xfer, flags_orig, flags_xfer)
                 if not fo and fx]
    dropped_flags = [{"orig": o[:120], "xfer": x[:120]}
                     for o, x, fo, fx in zip(texts_orig, texts_xfer, flags_orig, flags_xfer)
                     if fo and not fx]

    save_json({
        "n_clauses": len(clauses),
        "transform_model": args.model,
        "fpr_before": stat["fpr_before"],
        "fpr_after": stat["fpr_after"],
        "delta_pp": stat["delta_pp"],
        "mcnemar": stat,
        "reference": {
            "article_v2_all_2036": 0.094,
            "article_v2_intact_908": 0.121,
            "holdout_all_300": 0.053,
            "holdout_clean_283": 0.046,
        },
        "new_flags_sample": new_flags[:10],
        "dropped_flags_sample": dropped_flags[:10],
        "note": (
            "교락 있음 → fpr_after >> fpr_before (p<0.05). "
            "교락 없음 → fpr_after ≈ fpr_before. "
            "12.1%와 5.3%의 격차가 문체 때문이라면 fpr_after가 12%대로 올라와야 한다."
        ),
    }, OUT_PATH)
    logger.info(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
