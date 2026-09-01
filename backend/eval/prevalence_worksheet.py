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
import hashlib
import html
import json
import random
import re
from pathlib import Path

from backend.utils import PROJECT_ROOT, load_logger, save_json

logger = load_logger("prevalence_worksheet.log")

SRC_DIR = PROJECT_ROOT / "data/raw/real_tos"
OUT_DIR = PROJECT_ROOT / "data/eval/prevalence"
SHEET = OUT_DIR / "worksheet.csv"
PRED = OUT_DIR / "predictions.json"
REPORT = OUT_DIR / "prevalence_report.json"

# **평가셋은 얼린다.** 워크시트를 다시 만들 때마다 대상이 달라지면 이건 자산이 아니라
# 일회용이다. 얼린 셋(`EVALSET`)이 진실이고 CSV는 그걸 사람이 채우도록 편 표일 뿐이다.
EVALSET_VERSION = "v1"
_N_NEGATIVE = 50          # 표준계약서 음성 holdout에서 뽑을 건수
_NEG_SEED = 42
_SHUFFLE_SEED = 20260901  # 순서 단서 제거. 얼린 파일에 기록된다
EVALSET = OUT_DIR / f"evalset_{EVALSET_VERSION}.json"
# 판단은 **CSV가 아니라 여기** 쌓인다. CSV를 지우거나 다시 만들어도 판단이 살아남는다 —
# 오늘 워크시트를 재생성할 때 판단이 0건이라 손실이 없었던 것은 운이었다.
JUDGMENTS = OUT_DIR / "judgments.jsonl"

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


def _truncated_flags(texts: list[str], max_len: int = 256) -> tuple[list[bool], list[int]]:
    """모델 입력(`max_len` 토큰)에서 **잘리는** 조항 표시.

    실제 약관은 조 하나에 항이 여럿 붙어 학습 텍스트보다 2.4배 길고, 35.4%가 잘린다
    (`backend/eval/input_distribution_check.py`). 잘리는 구간에서 슬라이딩 윈도 OR이
    지목률을 42.9 → 60.0%로 올리는데, **그 17%p가 놓쳤던 위반인지 오경보인지 가릴 준거가
    없다.** 사람 판단이 이 구간을 덮으면 r과 함께 그 결정도 같이 난다.

    **블라인드를 깨지 않는다** — 모델 예측이 아니라 입력 길이다. 토크나이저를 못 읽으면
    글자 수로 근사한다(실측 2.3자/토큰).
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(PROJECT_ROOT / "models/article_v2"))
        n = [len(tok(t)["input_ids"]) for t in texts]
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"  토크나이저 없이 글자 수로 근사한다 ({e})")
        n = [int(len(t) / 2.3) for t in texts]
    return [x > max_len for x in n], n


def _uid(text: str) -> str:
    """조항의 **안정 ID**. 공백을 지운 본문의 해시라 재생성·정렬·중복제거 규칙이 바뀌어도 같다.

    예전 ID는 `kakao:3`처럼 **파일 안 순번**이었다. 원문에 조항 하나가 끼면 그 뒤가 전부
    밀려서, 어제 판단한 `kakao:3`과 오늘의 `kakao:3`이 다른 조항이 된다. 사람 판단을
    자산으로 쌓으려면 ID가 내용에 붙어 있어야 한다.
    """
    return hashlib.sha1(re.sub(r"\s+", "", text).encode("utf-8")).hexdigest()[:12]


def _load_judgments() -> dict[str, dict]:
    """`uid → 판단`. 파일이 없으면 빈 dict."""
    out: dict[str, dict] = {}
    if JUDGMENTS.exists():
        for line in JUDGMENTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["uid"]] = r
    return out


def harvest() -> int:
    """CSV에 채워진 판단을 **`judgments.jsonl`로 걷어 올린다.**

    이걸 돌려두면 워크시트를 다시 만들어도 판단이 안 날아간다. CSV는 입력 도구이고
    영속 저장소가 아니다 — 오늘 워크시트를 재생성할 수 있었던 것은 판단이 0건이었기
    때문이지 안전해서가 아니었다.
    """
    if not SHEET.exists():
        return 0
    have = _load_judgments()
    n = 0
    with open(SHEET, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            v = (r.get("violates_6_to_14") or "").strip()
            if v not in ("0", "1"):
                continue
            uid = r.get("uid") or _uid(r.get("clause_text", ""))
            rec = {"uid": uid, "violates_6_to_14": int(v),
                   "articles_judged": (r.get("articles_judged") or "").strip(),
                   "judged_by": (r.get("judged_by") or "").strip(),
                   "note": (r.get("note") or "").strip()}
            if have.get(uid) != rec:
                have[uid] = rec
                n += 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(JUDGMENTS, "w", encoding="utf-8") as fh:
        for uid in sorted(have):
            fh.write(json.dumps(have[uid], ensure_ascii=False) + "\n")
    return n


def _negative_holdout(n: int, seed: int) -> list[dict]:
    """표준계약서 음성 holdout에서 `n`건. **GPT 라벨로 거르지 않는다.**

    평가 코드는 지금까지 `not r["articles"]`(GPT가 빈 배열이라 한 것)만 음성 풀로 썼다.
    그건 채점에는 맞지만 **사람 판단 대상을 고를 때 쓰면 안 된다** — GPT가 고른 집합에서
    오경보를 세면 "GPT가 놓친 것"이 표본에서 통째로 빠진다. 156건 전부에서 뽑고,
    무엇이 음성인지는 **사람이 정한다.**
    """
    from backend.training.train_article import (
        exclude_gold_documents,
        load_article_records,
        load_ftc_gold,
        split_negative_holdout,
    )
    recs = exclude_gold_documents(
        load_article_records(PROJECT_ROOT / "data/fb_check/clean.jsonl"), load_ftc_gold("clean"))
    _, neg = split_negative_holdout(recs, 12, seed)
    rng = random.Random(seed)
    picked = rng.sample(sorted(neg, key=lambda r: _uid(r["text"])), min(n, len(neg)))
    return [{"text": r["text"], "source": "standard_contract", "cluster": r["group"]}
            for r in picked]


def _freeze(rows: list[dict]) -> dict:
    """평가셋을 얼려 저장한다. 이미 있으면 **읽기만 하고 바꾸지 않는다.**

    원천 약관을 더 모으면 조항 목록이 달라지는데, 그때 평가셋이 조용히 바뀌면
    "같은 셋으로 다음 모델을 평가한다"가 깨진다. 바꾸려면 버전을 올릴 것.

    ## 왜 실제 약관과 표준계약서를 **한 셋에** 넣나

    둘 다 같은 이진 판단("제6~14조 중 뭐라도 걸리나")이고, 나눠 얼리면 판단 세션이 둘로
    갈려 자산도 둘이 된다. 그리고 **섞어야 순서 단서가 사라진다** — 앞 99개가 실제 약관이면
    판단자가 뒤쪽 50개를 다른 눈으로 본다.
    """
    if EVALSET.exists():
        return json.loads(EVALSET.read_text(encoding="utf-8"))
    trunc, ntok = _truncated_flags([r["text"] for r in rows])
    items = [{"uid": _uid(r["text"]), "source": r["source"],
              "cluster": r.get("cluster") or r["source"], "text": r["text"],
              "n_tokens": t, "over_256_tokens": int(c)}
             for r, c, t in zip(rows, trunc, ntok)]
    random.Random(_SHUFFLE_SEED).shuffle(items)          # 순서 단서 제거
    data = {
        "version": EVALSET_VERSION,
        "created": "2026-09-01",
        "why": "gold(FTC 발췌문, 토큰 중앙 84)는 배포 분포(조문 전체, 토큰 중앙 184)를 "
               "대표하지 않는다. 이 셋은 **배포 분포 그 자체**이고 준거가 사람이라 "
               "순환도 없다 — 분포가 맞는 유일한 평가셋이다",
        "composition": "실제 약관(공공기관·포털) + 표준계약서 음성 holdout. 후자는 GPT 라벨로 "
                       "거르지 않고 156건에서 무작위 추출 — GPT가 고른 집합에서 오경보를 "
                       "세면 'GPT가 놓친 것'이 표본에서 빠진다",
        "blinding": "**판단 뷰(worksheet.csv)에 source를 넣지 않는다.** 출처를 보면 "
                    "'표준계약서는 공정하다'는 사전 지식이 판단에 들어가고, 그게 곧 "
                    "provenance가 라벨을 만드는 것이다 — 라벨링 파이프라인에서 금지한 바로 그것의 "
                    "사람 판단 판본. `over_256_tokens`는 입력 길이라 블라인드를 안 깨지만 "
                    "source는 깬다. 두 열을 같은 기준으로 두지 말 것",
        "shuffle_seed": _SHUFFLE_SEED,
        "negative_sample_seed": _NEG_SEED,
        "tokenizer": "models/article_v2",
        "dedup_ratio_threshold": 0.80,
        "clauses": items,
    }
    save_json(data, EVALSET)
    logger.info(f"  평가셋 얼림: {EVALSET} — 다음부터 이 파일이 대상을 정한다")
    return data


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

    neg = _negative_holdout(_N_NEGATIVE, _NEG_SEED)
    logger.info(f"  표준계약서 음성 holdout {len(neg)}건 추가 (156건에서 무작위, GPT 라벨로 안 거름)")
    rows = rows + neg
    frozen = _freeze(rows)

    # **얼린 셋이 대상을 정한다.** 원천을 더 모아도 이 버전의 평가셋은 안 바뀐다 —
    # 안 그러면 "같은 셋으로 다음 모델을 평가한다"가 성립하지 않는다.
    now = {_uid(r["text"]) for r in rows}
    old_ = {c["uid"] for c in frozen["clauses"]}
    if now != old_:
        logger.warning(f"  ⚠ 원천이 달라졌다 (추가 {len(now - old_)} · 사라짐 {len(old_ - now)}) — "
                       f"**평가셋 {frozen['version']}은 그대로 쓴다.** 반영하려면 "
                       f"EVALSET_VERSION을 올릴 것")

    harvested = harvest()                    # CSV를 덮기 **전에** 기존 판단을 걷어 올린다
    judged = _load_judgments()
    if harvested:
        logger.info(f"  기존 판단 {harvested}건을 {JUDGMENTS.name}로 걷어 올렸다")

    # **source는 판단 뷰에 넣지 않는다.** 출처가 판단을 끌면 라벨링에서 끊어낸 교락이
    # 사람 판단으로 되돌아온다. 저장 레코드(evalset)에는 남아 있고 uid로 이어 붙인다.
    cols = ["uid", "violates_6_to_14", "articles_judged", "judged_by", "note",
            "over_256_tokens", "n_tokens", "clause_text"]
    with open(SHEET, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for c in frozen["clauses"]:
            j = judged.get(c["uid"], {})
            w.writerow([c["uid"],
                        j.get("violates_6_to_14", ""), j.get("articles_judged", ""),
                        j.get("judged_by", ""), j.get("note", ""),
                        c["over_256_tokens"], c["n_tokens"], c["text"]])
    n, nt = len(frozen["clauses"]), sum(c["over_256_tokens"] for c in frozen["clauses"])
    logger.info(f"  워크시트 {n}개 조항 → {SHEET}   (평가셋 {frozen['version']}, 판단 {len(judged)}건 복원)")
    logger.info("  `violates_6_to_14`에 1(위반 의심)/0(아님), **`articles_judged`에 어느 조로 봤는지**를 "
                "적으세요 — 조를 같이 남겨야 나중에 모델의 조 예측과 대조할 수 있습니다")
    logger.info("  **모델 예측은 일부러 넣지 않았습니다** — 보고 판단하면 r이 오염됩니다.")
    logger.info(f"  `over_256_tokens` = article_v1이 잘랐던 조항({nt}/{n}건). **예측이 아니라 입력 "
                f"길이라 블라인드를 안 깬다** — 이 구간을 빠짐없이 판단해야 v1/v2 질문이 같이 풀린다")
    logger.info(f"  ★ 판단을 채운 뒤 `--harvest`를 돌리면 {JUDGMENTS.name}에 쌓입니다 — "
                f"CSV를 다시 만들어도 안 날아갑니다")
    save_json({"evalset_version": frozen["version"], "n_clauses": n, "n_truncated": nt,
               "n_judged": len(judged),
               "note": "얼린 평가셋. 판단은 judgments.jsonl에 쌓이고 CSV는 입력 도구일 뿐"}, REPORT)


def join(model_dir: str, gpu: int) -> None:
    """사람 판단이 끝난 워크시트에 모델 예측을 붙여 r과 일치율을 낸다."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from backend.model.electra import ArticleMultiLabelElectra
    from backend.training.train_article import ArticleDataset

    harvest()                                  # 집계 전에 항상 걷어 올린다
    rows = list(csv.DictReader(open(SHEET, encoding="utf-8-sig")))
    judged = [r for r in rows if (r.get("violates_6_to_14") or "").strip() in ("0", "1")]
    if not judged:
        raise SystemExit("판단된 행이 없다 — violates_6_to_14 열을 채우고 다시 실행할 것")

    # 판단 뷰에는 source가 없다(블라인드). 얼린 평가셋에서 uid로 되붙인다.
    meta = {c["uid"]: c for c in json.loads(EVALSET.read_text(encoding="utf-8"))["clauses"]}
    for r in judged:
        r["source"] = meta[r["uid"]]["source"]
        r["cluster"] = meta[r["uid"]]["cluster"]

    y = np.array([int(r["violates_6_to_14"]) for r in judged])
    real = [i for i, r in enumerate(judged) if r["source"] != "standard_contract"]
    std = [i for i, r in enumerate(judged) if r["source"] == "standard_contract"]
    logger.info("========== 유병률 r ==========")
    logger.info(f"  판단 완료 {len(judged)}/{len(rows)}개 조항 "
                f"(실제 약관 {len(real)} · 표준계약서 {len(std)})")

    # **r은 실제 약관에서만 낸다.** 표준계약서를 섞으면 배포 유병률이 아니라
    # "두 코퍼스를 이 비율로 섞었을 때의 값"이 된다 — 오늘 gold를 지층으로 가른 것과 같은 이유.
    for label, idx in (("실제 약관 (배포 분포 — 이것이 r)", real), ("표준계약서 (음성 준거)", std)):
        if not idx:
            continue
        by: dict[str, list[int]] = {}
        for i in idx:
            by.setdefault(judged[i]["cluster"], []).append(int(y[i]))
        lo, hi = cluster_ci(by)
        rate = float(y[idx].mean())
        logger.info(f"  {label}")
        logger.info(f"    {rate * 100:5.1f}%  95% CI [{lo * 100:.1f}%, {hi * 100:.1f}%]  "
                    f"(문서 {len(by)}개 클러스터 부트스트랩 — 조항 단위 Wilson은 거짓으로 좁다)")
        for sname, v in sorted(by.items()):
            logger.info(f"      {sname:<26}{sum(v):>3}/{len(v):<4} = {sum(v) / len(v) * 100:>5.1f}%")
    r_hat = float(y[real].mean()) if real else float(y.mean())
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
    # **max_len을 상수로 박지 않는다** — 체크포인트에서 읽는다(`Claude.md`의 원칙).
    _mc = json.loads((Path(model_dir) / "metrics.json").read_text(encoding="utf-8"))
    _ml = int((_mc.get("train_config") or {}).get("max_len", 256))
    ds = ArticleDataset([{"text": r["clause_text"], "articles": [], "group": "g"} for r in judged],
                        tok, _ml, m.article_names)
    P = []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=32):
            P.append(torch.sigmoid(m(input_ids=b["input_ids"].to(device),
                                     attention_mask=b["attention_mask"].to(device),
                                     token_type_ids=b["token_type_ids"].to(device))).cpu().numpy())
    P = np.vstack(P)
    thr = np.load(Path(model_dir) / "thresholds.npy")

    # ── 이 표의 왼쪽 두 칸이 오늘까지 없던 값이다 ──────────────────────────────
    # `disagree_with_gpt`는 음성 풀의 정답이 GPT 라벨 그 자체라 순환이었다. 여기서는
    # **사람이 정답이라 순환이 없다** — 이것이 이 프로젝트 최초의 독립 오경보율이다.
    logger.info("  ----- 사람 판단 대비 모델 -----")
    logger.info(f"    {'τ':>6}{'★오경보':>9}{'★재현':>8}{'정밀':>8}{'일치율':>8}{'모델 지목률':>12}")
    agree = {}
    thr_mean = float(np.mean(thr))
    trunc_idx = [i for i, r in enumerate(judged) if int(r.get("over_256_tokens") or 0)]
    for t in (0.15, 0.25, 0.35, 0.45, 0.65, thr_mean):
        pred = np.array([1 if any(v >= t for v in P[i]) else 0 for i in range(len(judged))])
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        rec = tp / (tp + fn) if tp + fn else 0.0
        pre = tp / (tp + fp) if tp + fp else 0.0
        fa = fp / (fp + tn) if fp + tn else 0.0          # **독립 오경보율**
        agree[f"{t:.2f}"] = {"false_alarm": fa, "recall": rec, "precision": pre,
                             "agreement": float((pred == y).mean()), "model_rate": float(pred.mean())}
        logger.info(f"    {t:>6.2f}{fa * 100:>8.1f}%{rec * 100:>7.1f}%{pre * 100:>7.1f}%"
                    f"{(pred == y).mean() * 100:>7.1f}%{pred.mean() * 100:>11.1f}%")
    logger.info("    ★ **오경보는 이제 사람 준거로 잰 값이다** — 음성 풀의 정답이 GPT 라벨이던 "
                "`disagree_with_gpt`의 순환이 여기엔 없다")

    # 절단 구간만 따로 — 결정 ③(v1 vs v2)이 여기서 갈린다. 두 모델로 각각 돌려 비교할 것.
    if trunc_idx:
        pred = np.array([1 if any(v >= thr_mean for v in P[i]) else 0 for i in range(len(judged))])
        yt, pt = y[trunc_idx], pred[trunc_idx]
        fp = int(((pt == 1) & (yt == 0)).sum()); tn = int(((pt == 0) & (yt == 0)).sum())
        tp = int(((pt == 1) & (yt == 1)).sum()); fn = int(((pt == 0) & (yt == 1)).sum())
        agree["truncated_subgroup"] = {
            "n": len(trunc_idx), "human_violation_rate": float(yt.mean()),
            "false_alarm": fp / (fp + tn) if fp + tn else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0}
        logger.info(f"  ----- 절단 구간만 (n={len(trunc_idx)}) — 결정 ③이 여기서 갈린다 -----")
        logger.info(f"    사람 위반율 {yt.mean() * 100:.1f}%  |  "
                    f"오경보 {agree['truncated_subgroup']['false_alarm'] * 100:.1f}%  "
                    f"재현 {agree['truncated_subgroup']['recall'] * 100:.1f}%")
        logger.info("    → `models/article_v1`로도 같은 명령을 돌려 두 값을 비교할 것. "
                    "v2의 지목률이 낮은 것이 **옳게 침묵한 것인지** 여기서 판정된다")

    save_json({"r_real_tos": r_hat, "n_judged": len(judged), "n_total": len(rows),
               "n_real": len(real), "n_standard_contract": len(std),
               "model_agreement": agree, "model_dir": model_dir,
               "false_alarm_note": "사람 준거. `disagree_with_gpt`(순환)를 대체하는 값이다",
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
    ap.add_argument("--harvest", action="store_true",
                    help="CSV에 채운 판단을 judgments.jsonl로 걷어 올린다. **판단하고 나면 이걸 먼저 돌릴 것** — "
                         "CSV는 입력 도구이고 영속 저장소가 아니다")
    ap.add_argument("--model-dir", default="models/article_v2")
    ap.add_argument("--gpu", type=int, default=1)
    a = ap.parse_args()
    if a.harvest:
        n = harvest()
        logger.info(f"  판단 {n}건 갱신 → {JUDGMENTS}  (누적 {len(_load_judgments())}건)")
    elif a.join:
        join(a.model_dir, a.gpu)
    else:
        build()


if __name__ == "__main__":
    main()
