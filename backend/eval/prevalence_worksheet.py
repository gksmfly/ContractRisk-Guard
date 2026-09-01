# backend/eval/prevalence_worksheet.py
"""배포 위반 유병률 r 측정용 **블라인드 워크시트**를 만든다.

## r이 왜 필요한가

`threshold_r_sweep`이 배포 임계값을 r의 함수로 냈다. r은 "실제 계약서 한 장에서 약관규제법
제6~14조를 위반하는 조항의 비율"이고, 이 값이 정해져야 운영점 τ가 정해진다:

    τ=0.45, r=0.10  →  30조항 중 3.3건 표시, 조항 단위 정밀도 73%
    τ=0.45, r=0.05  →  2.2건 표시, 정밀도 56%

같은 모델·같은 임계값인데 r만으로 체감 품질이 갈린다.

## 왜 사람이 판단해야 하나

    모델로 세면      순환 — 지금 보정하려는 대상이 그 판정자다
    GPT-4o로 세면    같은 이유로 순환. gold에서 조 F1 45%인 판정자이기도 하다
    사람이 판단      가능. 조 단위 정확도가 필요 없다 —
                    "제6~14조 중 뭐라도 걸리나" **이진 판단**이면 된다

## 왜 블라인드인가

**워크시트에 모델 예측을 같이 넣으면 판단이 그쪽으로 끌려간다.** 예측은 별도 파일
(`*_predictions.json`)에 조항 id로 저장하고, 사람이 판단을 끝낸 뒤 `--join`으로 붙인다.
그러면 (a) r이 오염되지 않고 (b) 부수적으로 **사람–모델 일치율**이 공짜로 나온다.

## 표본의 한계 — 반드시 함께 보고할 것

지금 모은 문서는 공공기관·포털 이용약관이다. 상당수가 **공통 템플릿에서 복제**돼
서로 독립 표본이 아니고, 상업 계약(가맹·대리점·구독 결제 등)보다 불공정 조항이 적을
가능성이 높다. 즉 **여기서 나온 r은 하한에 가깝다.**

    표준약관(공정위 발행)   r ≈ 0        설계상 공정
    공공기관 이용약관        r ≈ ?        ← 이 워크시트
    상업 서비스 약관         r ≈ ?        더 높을 것으로 예상, 수집이 막혀 있음
    FTC 심사 대상 약관       r 편향 높음   신고돼 위반이 확정된 계약

실행:
    .venv/bin/python -m backend.eval.prevalence_worksheet            # 워크시트 생성
    .venv/bin/python -m backend.eval.prevalence_worksheet --join     # 판단 후 집계
"""

import argparse
import csv
import html
import json
import re
from pathlib import Path

from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("prevalence_worksheet.log")

SRC_DIR = PROJECT_ROOT / "data/raw/real_tos"
OUT_DIR = PROJECT_ROOT / "data/eval/prevalence"
SHEET = OUT_DIR / "worksheet.csv"
PRED = OUT_DIR / "predictions.json"
REPORT = OUT_DIR / "prevalence_report.json"

_MIN_CHARS = 40          # 목차 항목(제목만 있는 줄)을 걸러낸다
_HEAD = re.compile(r"제\s*\d+\s*조\s*[（(]")


def _text(path: Path) -> str:
    """HTML이면 태그를 벗기고, `.txt`면 그대로 쓴다.

    상업 약관 사이트는 대부분 SPA·봇 차단이라 크롤이 막힌다(30여 곳 시도, 거의 전멸).
    **브라우저로 열어 Ctrl+A/Ctrl+C 한 것을 `.txt`로 저장하면 전부 우회된다** —
    렌더링이 끝난 화면은 그냥 텍스트다. 손으로 5~10개면 크롤링보다 빠르다.

    제품이 겨냥하는 업종을 고를 것: 헬스장 회원약관 · 학원 수강규정 · 렌탈 · 이사 ·
    예식장 · 상조 · 임대차. 이런 업종은 PDF·HWP로 올려둔 곳이 많아 SPA 문제도 없다.
    """
    if path.suffix.lower() in (".txt", ".md"):
        t = path.read_text(encoding="utf-8", errors="replace")
        t = re.sub(r"[ \t\xa0]+", " ", t)
        return re.sub(r"\n\s*\n+", "\n", t)
    h = path.read_text(encoding="utf-8", errors="replace")
    h = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", h)
    t = re.sub(r"(?s)<[^>]+>", "\n", html.unescape(h))
    t = re.sub(r"[ \t\xa0]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n", t)


def clauses(path: Path) -> list[str]:
    """`제N조(제목) 본문` 단위로 자른다. 목차는 본문이 짧아 `_MIN_CHARS`에서 걸린다."""
    t = _text(path)
    pos = [m.start() for m in _HEAD.finditer(t)]
    out = []
    for i, p in enumerate(pos):
        end = pos[i + 1] if i + 1 < len(pos) else min(len(t), p + 3000)
        body = re.sub(r"\s+", " ", t[p:end]).strip()
        if len(body) >= _MIN_CHARS:
            out.append(body[:2000])
    return out


def _dedup(rows: list[dict], thr: float = 0.80) -> tuple[list[dict], int]:
    """문서 간 근사중복 조항을 제거한다.

    공공기관 이용약관은 **공통 템플릿에서 복제**된다. 실측(수집분 6개 문서):

        docu24 ↔ gov24        77~81% 중복   ← 사실상 같은 문서
        copyright ↔ docu24    22~24%
        kakao · keit · kjwon  서로 0%       ← 독립

    중복을 그대로 두고 124건을 독립 표본으로 세면 **CI가 거짓으로 좁아진다.**
    r은 비율이라 같은 조항을 두 번 세면 분산이 과소평가된다.
    """
    import difflib
    kept, dropped = [], 0
    norms: list[str] = []
    for r in rows:
        n = re.sub(r"\s+", "", r["text"])[:400]
        if any(difflib.SequenceMatcher(None, n, m).ratio() > thr for m in norms):
            dropped += 1
            continue
        norms.append(n)
        kept.append(r)
    return kept, dropped


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(list(SRC_DIR.glob("*.html")) + list(SRC_DIR.glob("*.txt"))):
        cs = clauses(f)
        if len(cs) < 5:
            continue                       # 약관 페이지가 아니거나 수집 실패
        for i, c in enumerate(cs, 1):
            rows.append({"id": f"{f.stem}:{i}", "source": f.stem, "text": c})
        logger.info(f"  {f.stem:<14}{len(cs):>4}개 조항")
    rows, dropped = _dedup(rows)
    logger.info(f"  근사중복 제거 {dropped}건 → {len(rows)}건 (공통 템플릿 복제, `_dedup` 참고)")
    if not rows:
        raise SystemExit(f"{SRC_DIR} 에 약관 파일(.html/.txt)이 없다")

    with open(SHEET, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "source", "violates_6_to_14", "note", "clause_text"])
        for r in rows:
            w.writerow([r["id"], r["source"], "", "", r["text"]])
    logger.info(f"  워크시트 {len(rows)}개 조항 → {SHEET}")
    logger.info("  `violates_6_to_14` 열에 1(위반 의심) / 0(아님) 만 채우세요.")
    logger.info("  **모델 예측은 일부러 넣지 않았습니다** — 보고 판단하면 r이 오염됩니다.")
    save_json({"n_clauses": len(rows), "by_source": {r["source"]: 0 for r in rows} | {},
               "note": "판단 전 상태"}, REPORT)


def join(model_dir: str, gpu: int) -> None:
    """사람 판단이 끝난 워크시트에 모델 예측을 붙여 r과 일치율을 낸다."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from backend.model.electra import ArticleMultiLabelElectra
    from backend.training.train_article import ArticleDataset

    rows = list(csv.DictReader(open(SHEET, encoding="utf-8-sig")))
    judged = [r for r in rows if (r.get("violates_6_to_14") or "").strip() in ("0", "1")]
    if not judged:
        raise SystemExit("판단된 행이 없다 — violates_6_to_14 열을 채우고 다시 실행할 것")

    y = np.array([int(r["violates_6_to_14"]) for r in judged])
    r_hat = float(y.mean())
    logger.info(f"========== 유병률 r ==========")
    logger.info(f"  판단 완료 {len(judged)}/{len(rows)}개 조항")
    by = {}
    for r in judged:
        by.setdefault(r["source"], []).append(int(r["violates_6_to_14"]))
    lo, hi = cluster_ci(by)
    logger.info(f"  **r = {r_hat * 100:.1f}%**  95% CI [{lo * 100:.1f}%, {hi * 100:.1f}%]  "
                f"(문서 {len(by)}개 클러스터 부트스트랩 — 조항 단위 Wilson은 거짓으로 좁다)")
    for s, v in sorted(by.items()):
        logger.info(f"    {s:<14}{sum(v):>3}/{len(v):<4} = {sum(v) / len(v) * 100:>5.1f}%")
    logger.warning("  ★ [사전 등록] 이 표본의 r은 **하한**이다. 공공기관은 영리 동기가 없어 "
                   "불공정 조항을 넣을 이유가 거의 없고 대형 플랫폼은 법무 검토를 거친다 — "
                   "가능한 표본 중 r이 가장 낮게 나올 조합이다.")
    logger.warning("  ★ **r < 0.05가 나와도 운영점 τ 결정에 쓰지 않는다.** τ 결정은 상업 약관"
                   "(가맹·헬스장·학원·렌탈·상조·임대차 등) 표본이 들어온 뒤로 미룬다.")
    logger.warning("  ★ 이걸 미리 정해두지 않으면 r=0.02를 보고 '배포 가치 없음'으로 읽게 되는데, "
                   "그건 모델이 아니라 표본이 만든 결론이다")

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    m = ArticleMultiLabelElectra.load(Path(model_dir)).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    ds = ArticleDataset([{"text": r["clause_text"], "articles": [], "group": "g"} for r in judged],
                        tok, 256, m.article_names)
    P = []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=32):
            P.append(torch.sigmoid(m(input_ids=b["input_ids"].to(device),
                                     attention_mask=b["attention_mask"].to(device),
                                     token_type_ids=b["token_type_ids"].to(device))).cpu().numpy())
    P = np.vstack(P)
    thr = np.load(Path(model_dir) / "thresholds.npy")

    logger.info("  ----- 사람 판단 대비 모델 (일치율·부수 지표) -----")
    logger.info(f"    {'τ':>6}{'모델 위반율':>12}{'일치율':>9}{'재현':>8}{'정밀':>8}")
    agree = {}
    for t in (0.15, 0.25, 0.35, 0.45, 0.65, float(np.mean(thr))):
        pred = np.array([1 if any(v >= t for v in P[i]) else 0 for i in range(len(judged))])
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        rec = tp / (tp + fn) if tp + fn else 0.0
        pre = tp / (tp + fp) if tp + fp else 0.0
        agree[f"{t:.2f}"] = {"model_rate": float(pred.mean()), "agreement": float((pred == y).mean()),
                             "recall": rec, "precision": pre}
        logger.info(f"    {t:>6.2f}{pred.mean() * 100:>11.1f}%{(pred == y).mean() * 100:>8.1f}%"
                    f"{rec * 100:>7.1f}%{pre * 100:>7.1f}%")
    save_json({"r": r_hat, "ci95": [lo, hi], "n_judged": len(judged), "n_total": len(rows),
               "by_source": {s: {"n": len(v), "r": sum(v) / len(v)} for s, v in by.items()},
               "model_agreement": agree, "model_dir": model_dir,
               "ci_method": "문서 단위 클러스터 부트스트랩 (조항 단위 Wilson은 상관을 무시해 거짓으로 좁다)",
               "preregistered": "이 표본의 r은 하한이다. r<0.05여도 운영점 τ 결정에 쓰지 않는다 — "
                               "상업 약관 표본이 들어온 뒤로 미룬다",
               "caveat": "공공기관 약관 위주 표본. 공통 템플릿 복제가 많아 독립 표본이 아니며 "
                         "상업 계약보다 r이 낮을 가능성이 높다 — 하한으로 읽을 것"}, REPORT)
    logger.info(f"  저장: {REPORT}")


def cluster_ci(by_doc: dict[str, list[int]], seed: int = 42, n_boot: int = 5000) -> tuple[float, float]:
    """**문서 단위** 클러스터 부트스트랩 95% CI.

    Wilson CI는 조항이 서로 독립이라고 가정한다. 그런데 **한 계약서 안의 조항은 강하게
    상관한다** — 대충 만든 약관은 여러 조항이 한꺼번에 불공정하고, 잘 만든 약관은 전부
    깨끗하다. 즉 r의 불확실성을 지배하는 것은 조항 간 변동이 아니라 **문서 간 변동**이다.

        Wilson on n=99      ±8%p 정도   ← 거짓으로 좁다
        문서 단위 클러스터     훨씬 넓다    ← 문서가 4~5개뿐이라 특히

    문서를 복원추출한 뒤 그 안의 조항을 전부 쓴다. **넓게 나오는 것이 정상이다** —
    문서 4~5개로는 r을 좁게 잡을 수 없고, 좁게 보고하면 나중에 상업 약관을 넣었을 때
    값이 구간 밖으로 튄다.
    """
    import random
    docs = sorted(by_doc)
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        pick = [by_doc[docs[rng.randrange(len(docs))]] for _ in docs]
        flat = [v for c in pick for v in c]
        if flat:
            out.append(sum(flat) / len(flat))
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def main() -> None:
    ap = argparse.ArgumentParser(description="r 측정 워크시트")
    ap.add_argument("--join", action="store_true", help="판단이 끝난 워크시트를 집계한다")
    ap.add_argument("--model-dir", default="models/_article_rNone")
    ap.add_argument("--gpu", type=int, default=1)
    a = ap.parse_args()
    join(a.model_dir, a.gpu) if a.join else build()


if __name__ == "__main__":
    main()
