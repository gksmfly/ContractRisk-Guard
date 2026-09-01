# backend/training/train_article.py
"""
약관규제법 조 multi-label 인코더 학습 — `train.py`(domain+risk single-label)의 형제.

## 왜 별도 모듈인가

`train.py`의 `DualHeadElectra` 경로는 9개 파일이 의존하고 그중 둘이 전량 라벨링 경로다.
제자리에서 갈아엎으면 라벨 생성이 먼저 깨진다. 그래서 헤드(`ArticleMultiLabelElectra`)도
학습 루프도 새로 두고, 공유 가능한 것(`split_by_document`, `_document_group`)만 재사용한다.
`train.py`에 헤드 선택 옵션은 두지 않았다 — 진입점까지 갈라야 서로를 안 깬다.

single-label과 다른 점이 손실 하나가 아니라 평가 전체다 — 임계값, 조별 지표, 체크포인트
선택 기준이 전부 달라서 한 함수에 분기를 넣으면 읽을 수 없어진다.

## 이 파일이 지키는 규칙 (전부 이 프로젝트에서 실제로 데인 것들)

1. **held-out gold를 학습·튜닝 어디에도 쓰지 않는다.** FTC 단일조항 사건은 외부 정답이라
   최종 채점 전용이다. 임계값을 여기서 고르면 그게 곧 과적합이다.
2. **문서(doc) 단위로 뺀다.** `chunk_id`만 빼면 같은 사건의 다른 조항이 학습에 남는다 —
   검색기반 비교를 뒤집었던 doc-level leakage와 같은 함정이다.
3. **상수 기준선을 매 epoch 함께 찍는다.** "항상 제6조만 찍기"를 못 넘으면 학습이 무의미하다.
   이걸 안 재서 같은 실수를 다섯 번 했다([[feedback_measure_constant_baseline]]).
4. **per-article F1에 support를 항상 병기한다.** 희소한 조는 F1이 0/1로 튀어 macro를 흔든다.
5. **제6조 제외 macro를 병기한다.** 제6조는 정답의 36%에 붙는 사실상 다수 클래스라,
   제6조만 잘 맞혀도 macro가 좋아 보이는 착시가 생긴다.
6. **증강은 기본 OFF.** evidence_span·원문 이중 증강은 예전 누수 경로 셋 중 하나였다.
   기준선을 먼저 세우고 증강은 나중에 A/B로 켠다 — 처음부터 켜면 "누수가 막힌 건지
   증강이 도운 건지" 못 가린다.
7. **체크포인트는 dev split으로 고른다.** 그리고 무엇으로 골랐는지 `metrics.json`에 남긴다.

실행:
    .venv/bin/python -m backend.training.train_article --gpu 1 --num-workers 0
    .venv/bin/python -m backend.training.train_article --dry-run     # 라벨 없이 형상만 검증
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from backend.labeling.articles import ARTICLE_IDS, normalize
from backend.model.electra import (
    ArticleMultiLabelElectra,
    article_labels,
    article_pos_weight,
)
from backend.training.train import BASE_MODEL, _document_group, split_by_document
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("train_article.log")

LABELED_PATH = Path(os.environ.get("ARTICLE_LABELS", str(PROJECT_ROOT / "data/fb_check/clean.jsonl")))
FTC_PATH     = Path(os.environ.get("FTC_PATH", str(PROJECT_ROOT / "data/raw/ftc_cases/ftc_cases_parsed.json")))

# 제6조는 정답의 36%에 붙는 다수 클래스라 macro를 혼자 끌어올린다 — 제외 macro를 병기한다.
_MAJORITY_ARTICLE = "제6조"
_MIN_CLAUSE_CHARS = 30


# ─────────────────────────────────────────────────────────────────────────────
# 데이터
# ─────────────────────────────────────────────────────────────────────────────

def load_article_records(path: Path, augment: bool = False,
                         label_source: str = "agreed") -> list[dict]:
    """라벨링 산출물에서 (조항 텍스트 → 조 목록) 레코드를 만든다.

    빈 배열도 유효한 라벨이다 — "제6~14조 위반 없음"이라는 음성 표본이라 버리면
    모델이 항상 뭔가를 지목하도록 학습된다.

    ## `label_source`를 지금 정하지 않는 이유

    FB-Check 레코드에 `forward_articles`와 `agreed_articles`(forward ∩ verify)가 **둘 다**
    남아 있어, 어느 쪽으로 학습할지는 재라벨링 없이 학습 시점에 고를 수 있다. 실측상
    두 라벨의 F1은 사실상 동률인데 **정밀/재현이 정확히 맞바뀐다**(같은 69건 기준
    forward 정밀 58.0/재현 68.8, agreed 정밀 64.5/재현 61.8).

    multi-label BCE에서 이 차이는 무해하지 않고 **해로운 방향이 서로 다르다**:

        forward (재현↑)  →  거짓 양성 라벨 ↑  →  없는 연관을 학습한다
        agreed  (정밀↑)  →  거짓 음성 라벨 ↑  →  "이 조항은 제9조가 아니다"를 적극 학습한다

    어느 쪽이 나은지는 재봐야 안다. 기본값은 `agreed`(교집합, 더 보수적)이고,
    `--label-source forward`로 A/B 할 수 있다.
    """
    if label_source not in ("forward", "agreed"):
        raise ValueError(f"알 수 없는 label_source: {label_source}")
    rows = load_jsonl(path)
    out: list[dict] = []
    for i, r in enumerate(rows):
        text = (r.get("text") or "").strip()
        if len(text) < _MIN_CLAUSE_CHARS:
            continue
        if label_source == "agreed":
            arts = r.get("agreed_articles")
            if arts is None:                      # 옛 레코드에는 없다
                arts = r.get("forward_articles") or []
        else:
            arts = r.get("forward_articles") or []
        arts = [a for a in arts if a in ARTICLE_IDS]
        out.append({
            "text": text,
            "articles": sorted(set(arts), key=lambda a: int(a.strip("제조"))),
            "group": _document_group(r.get("chunk_id"), f"row-{i}"),
            "source": r.get("source", ""),
        })

    if augment:
        # 기본 OFF. 켜면 evidence_span을 별도 예시로 추가하는데, 원문이 그 문구를
        # 부분 문자열로 담아 분할 누수가 된다(v8에서 실제로 발생). split_by_document가
        # 막긴 하지만 기준선을 먼저 세우기 전엔 켜지 않는다.
        extra = []
        for r, src in zip(out, rows):
            span = (src.get("evidence_span") or "").strip()
            if len(span) >= _MIN_CLAUSE_CHARS:
                extra.append({**r, "text": span})
        logger.info(f"  증강(evidence_span) {len(extra)}건 추가 — 누수 위험 구간이다")
        out += extra
    return out


def load_ftc_gold(stratum: str = "clean") -> list[dict]:
    """FTC 단일조항 사건 = **외부 정답**. 학습·튜닝 금지, 최종 채점 전용.

    조항이 1개인 사건만 쓴다 — 여러 개면 사건 단위 `근거_법령`을 어느 조항에 내려야
    할지 알 수 없어 라벨이 뭉개진다(Phase A에서 실제로 겪은 오염).

    ## 그런데 그 필터는 반쪽이었다 (2026-08-30)

    조항 1개로 걸러도 **`근거_법령`이 2개 이상인 사건이 22% 남는다.** 근거 법령은
    조항이 아니라 **사건 전체**에 붙으므로, 파서가 형제 조항을 놓친 사건에서는
    그 조들이 전부 우리가 가진 조항 하나에 귀속된다.

    원본에서 확인했다 — `근거_법령 2개+` 72건의 **100%** 가 그 경우다:

        구간          위반_유형 평균   추출 조항   근거_법령
        근거 2개+       2.89개        1.00개     2.67개   ← 의결서는 2.89개 위반을
                                                          서술했는데 조항은 1개만 찾았다
        근거 1개        1.08개        1.00개     1.00개   ← 대조군

    "여러 조를 실제로 위반하는 어려운 조항"이 아니라 **파서 누락의 부작용**이다.
    채점하면 그 구간의 최적 상수가 "조 4개를 항상 찍기"(F1 56.6%)가 되고, 조항 하나를
    읽고 1.4개를 내는 teacher(44.6%)·학생(43.0%)이 **둘 다 진다.** 모델의 실패가 아니라
    채점 대상의 성질이다.

    깨끗한 구간(n=255)에서는 **teacher `+9.0%p [+2.9,+15.4]`, 학생 `+6.4%p [+1.6,+11.4]`**
    로 둘 다 상수를 유의하게 이긴다. 섞어서 평균 내면 미판정으로 눌린다.

    ## 지우지 않고 구간으로 나눈다

        clean   근거 1개  n=255   ← 주 평가. 귀속이 모호할 여지가 없다
        noisy   근거 2개+ n=72    ← 사유와 함께 병기. 파서를 고쳐 형제 조항을 찾으면 복구 대상
        all     327건            ← 옛 정의. 과거 수치와 대조할 때만

    평가셋을 조용히 줄이는 것과, 줄인 사실과 이유를 같이 내보내는 것은 다르다.
    """
    if stratum not in ("clean", "noisy", "all"):
        raise ValueError(f"알 수 없는 stratum: {stratum}")
    cases = json.loads(FTC_PATH.read_text(encoding="utf-8"))["사례"]
    gold: list[dict] = []
    for c in cases:
        cl = [str(x).strip() for x in (c.get("조항_원문") or []) if len(str(x).strip()) >= _MIN_CLAUSE_CHARS]
        arts = {a for g in (c.get("근거_법령") or []) if (a := normalize(str(g)))}
        if len(cl) != 1 or not arts:
            continue
        if stratum == "clean" and len(arts) != 1:
            continue
        if stratum == "noisy" and len(arts) < 2:
            continue
        gold.append({
            "text": cl[0],
            "articles": sorted(arts, key=lambda a: int(a.strip("제조"))),
            "doc_id": str((c.get("셀_데이터") or {}).get("사건번호") or c.get("사건명", "")),
            "n_violation_types": len(c.get("위반_유형") or []),
        })
    return gold


def split_negative_holdout(records: list[dict], n_docs: int, seed: int) -> tuple[list[dict], list[dict]]:
    """표준계약서 문서 `n_docs`개를 학습에서 **완전히** 떼어 음성 평가셋으로 만든다.

    ## 왜 필요한가 — 고정 gold도 편향돼 있다

    `--negative-ratio` 팔들을 "고정된 held-out gold에서 비교하면 된다"고 적었는데,
    그것만으로는 부족하다. **held-out gold 365건은 전부 FTC이고 FTC gold는 절대
    비어 있지 않다.** 그러면 음성 표본을 많이 본 모델(빈 배열 쪽으로 기운 모델)은
    구조적으로 불리하고 `--negative-ratio 0.0`이 자동으로 이긴다 — `apply_negative_ratio`
    주석의 상수 표가 보여준 것과 똑같은 구조다.

    그래서 축이 둘이어야 한다:

        축 1  FTC gold          위반을 찾아내는가        (재현 쪽)   ← ratio 0.0이 유리
        축 2  표준계약서 held-out  올바르게 비어 있는가      (정밀 쪽)   ← ratio None이 유리

    실제 제품 조건(사용자가 올리는 계약서는 대부분 멀쩡하고 몇 개가 문제)에서는 둘 다 봐야
    한다. **어느 축에 무게를 둘지는 측정으로 안 정해진다 — 제품 판단이다.**

    ## 반드시 ratio보다 **먼저** 갈라야 한다

    고정 seed로, `--negative-ratio`와 무관하게 먼저 뗀다. 나중에 떼면 팔마다 평가셋이
    달라져 비교가 또 깨진다 — 이 파일에서 두 번째로 같은 실수를 하는 셈이 된다.
    문서 단위인 것도 같은 이유다(형제 조항 누수).
    """
    if n_docs <= 0:
        return records, []
    neg_docs = sorted({r["group"] for r in records if r.get("source") == "standard_contract"})
    if n_docs >= len(neg_docs):
        raise SystemExit(f"--negative-holdout {n_docs}: 표준계약서 문서가 {len(neg_docs)}개뿐이다")
    rng = random.Random(seed)          # ratio와 무관한 고정 시드 — 팔마다 같은 평가셋
    held = set(rng.sample(neg_docs, n_docs))
    train = [r for r in records if r["group"] not in held]
    holdout = [r for r in records if r["group"] in held]
    empty = sum(1 for r in holdout if not r["articles"]) / max(1, len(holdout))
    logger.info(f"  음성 held-out: 표준계약서 문서 {n_docs}개 / 조항 {len(holdout)}건 분리 "
                f"(빈 배열 {empty * 100:.1f}%) — 학습·임계값 튜닝에 쓰지 않는다")
    return train, holdout


def apply_negative_ratio(records: list[dict], ratio: float | None, seed: int) -> list[dict]:
    """표준계약서(음성 표본) 비중을 FTC 대비 `ratio`로 맞춘다. None이면 그대로 둔다.

    ## 왜 스윕이 필요한가

    시드 구성의 FTC 2,432 : 표준계약서 2,036은 **임의로 고른 비율**이고
    (`labeling/seed.py`의 `--contract-sample`), 실제 위반 비율을 반영하지 않는다.
    그런데 교락 측정에서 그 비율이 학습 난이도를 직접 정한다는 게 드러났다:

        무조건 상수 "항상 위반 없음"  →  건별 F1 61.6%
        표준계약서가 CLEAN의 56%, 그중 91.3%가 빈 배열이라 나오는 값이다

    multi-label BCE에서 **빈 라벨이 과반이면 모델은 아무것도 예측하지 않는 쪽으로 강하게
    끌린다.** `pos_weight`는 조별 불균형을 보정하지만 "라벨이 아예 없는 샘플이 절반"인
    구조는 보정 대상이 아니다 — 모든 조의 로짓을 함께 내리는 압력이라 조별 가중치로는
    상쇄되지 않는다.

    비율은 **학습 시점에** 정한다. 라벨은 양쪽 다 만들어뒀으므로 재라벨링이 필요 없다.
    문서 단위로 덜어낸다 — 조항 단위로 뽑으면 같은 문서의 형제 조항이 학습·검증에
    갈라져 누수가 된다(Phase 1에서 막은 그 경로).
    """
    if ratio is None:
        return records
    pos = [r for r in records if r.get("source") != "standard_contract"]
    neg = [r for r in records if r.get("source") == "standard_contract"]
    keep_n = int(len(pos) * ratio)
    if keep_n >= len(neg):
        logger.info(f"  --negative-ratio {ratio:g}: 표준계약서 {len(neg)}건 전량 유지 "
                    f"(목표 {keep_n}건이 보유량 이상)")
        return records

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in neg:
        by_doc[r["group"]].append(r)
    docs = sorted(by_doc)
    random.Random(seed).shuffle(docs)
    kept: list[dict] = []
    for d in docs:
        if len(kept) >= keep_n:
            break
        kept.extend(by_doc[d])
    kept_docs = len({r["group"] for r in kept})
    logger.info(f"  --negative-ratio {ratio:g}: 표준계약서 {len(neg)}→{len(kept)}건 "
                f"(문서 {len(by_doc)}→{kept_docs}개) | "
                f"FTC {len(pos)}건 대비 {len(kept) / max(1, len(pos)):.2f}")

    # ratio 간 dev 지표를 **직접 비교하면 안 된다** — 표본 구성이 바뀌면 상수 기준선이
    # 함께 움직인다. 실측(clean.jsonl 1,700건, label_source=agreed):
    #
    #     ratio   전체   빈라벨   "최적 상수" F1   최적 상수
    #     None    1700   61.6%        61.6%      (위반 없음)
    #     1.0     1534   57.8%        57.8%      (위반 없음)
    #     0.5     1130   46.7%        46.7%      (위반 없음)
    #     0.25     943   37.4%        37.4%      (위반 없음)
    #     0.0      753   24.3%        26.4%      제6·8·10조   ← 최적 상수의 종류가 뒤집힌다
    #
    # ratio 0.25에서 dev F1 55%가 나와도 상수(37.4%)를 +17.6%p 이긴 것이고,
    # ratio None에서 F1 62%는 상수(61.6%)를 +0.4%p 이긴 것이다. **절대값이 낮은 쪽이
    # 더 잘한 것이다.** 비교는 (a) 각 ratio의 상수 대비 초과분, 또는 (b) 고정된
    # held-out gold(`article_gold_eval`) 위에서 해야 한다.
    logger.warning("  ★ ratio 간 dev 지표를 직접 비교하지 말 것 — 표본이 바뀌면 상수 기준선도 "
                   "함께 움직인다. 상수 대비 초과분 또는 고정 gold로 비교할 것")
    if kept_docs < 15:
        logger.warning(f"  ★ 표준계약서 문서가 {kept_docs}개뿐이다 — 건수보다 **문서 다양성**이 "
                       f"먼저 줄었다. 이 팔의 결과는 문서 표집 분산이 크므로 시드를 여러 개 볼 것")
    return pos + kept


def exclude_gold_documents(records: list[dict], gold: list[dict]) -> list[dict]:
    """held-out gold와 **같은 문서**에서 나온 학습 레코드를 제거한다(doc 단위 누수 차단)."""
    gold_docs = {g["doc_id"] for g in gold if g["doc_id"]}
    gold_texts = {g["text"].strip() for g in gold}
    kept, dropped_doc, dropped_text = [], 0, 0
    for r in records:
        # group은 `source:문서ID` 형태라 사건번호가 그 안에 들어 있다
        if any(d and d in r["group"] for d in gold_docs):
            dropped_doc += 1
            continue
        if r["text"].strip() in gold_texts:
            dropped_text += 1
            continue
        kept.append(r)
    logger.info(f"  held-out gold 문서 배제: 문서 일치 {dropped_doc}건 · 텍스트 일치 {dropped_text}건 "
                f"→ 학습 후보 {len(kept)}건")
    return kept


class ArticleDataset(Dataset):
    """조항 텍스트 → multi-hot 조 라벨."""

    def __init__(self, records: list[dict], tokenizer: Any, max_len: int, names: list[str]) -> None:
        self.records, self.tokenizer, self.max_len = records, tokenizer, max_len
        self.index = {a: i for i, a in enumerate(names)}
        self.n = len(names)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        enc = self.tokenizer(rec["text"], max_length=self.max_len, padding="max_length",
                             truncation=True, return_tensors="pt")
        y = torch.zeros(self.n, dtype=torch.float)
        for a in rec["articles"]:
            if a in self.index:      # 헤드에서 접힌 조(support 부족)는 라벨에서도 뺀다
                y[self.index[a]] = 1.0
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros(self.max_len, dtype=torch.long)).squeeze(0),
            "labels":         y,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 평가
# ─────────────────────────────────────────────────────────────────────────────

def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def per_article_metrics(probs: np.ndarray, labels: np.ndarray, names: list[str],
                        thresholds: np.ndarray) -> dict:
    """조별 P/R/F1 + support. macro는 제6조 포함/제외 두 가지를 낸다."""
    rows = {}
    for j, a in enumerate(names):
        pred = probs[:, j] >= thresholds[j]
        true = labels[:, j] > 0.5
        tp = int((pred & true).sum()); fp = int((pred & ~true).sum()); fn = int((~pred & true).sum())
        p, r, f1 = _prf(tp, fp, fn)
        rows[a] = {"precision": p, "recall": r, "f1": f1, "support": int(true.sum()),
                   "threshold": float(thresholds[j])}
    # **support가 0인 조는 macro에서 뺀다.** 그 조의 F1은 측정값이 아니라 상수 0이라,
    # 포함시키면 체크포인트 선택 기준에 잡음이 상시로 섞인다(8개 중 1개가 항상 0이면
    # macro가 1/8만큼 눌리고, 모델이 그 조를 간혹 예측하면 전부 오탐이라 여전히 0이다).
    # 대신 **몇 개 조가 macro에 들어갔는지 함께 낸다** — 조 개수가 바뀌면 값이 점프하므로
    # 그걸 모르면 epoch 간 비교가 깨진다.
    scored   = {a: v for a, v in rows.items() if v["support"] > 0}
    f1s      = [v["f1"] for v in scored.values()]
    f1s_ex   = [v["f1"] for a, v in scored.items() if a != _MAJORITY_ARTICLE]
    return {
        "per_article": rows,
        "scored_articles": sorted(scored),
        "n_scored": len(scored),
        "zero_support_articles": sorted(a for a, v in rows.items() if v["support"] == 0),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "macro_f1_excl_majority": float(np.mean(f1s_ex)) if f1s_ex else 0.0,
    }


def tune_thresholds(probs: np.ndarray, labels: np.ndarray, names: list[str]) -> np.ndarray:
    """조별 임계값을 F1 최대가 되게 고른다. **dev split에서만 호출할 것.**

    0.5 고정은 불균형 multi-label에서 거의 항상 나쁘다(희소한 조는 확률이 0.5를 못 넘음).
    다만 이걸 held-out gold에서 고르면 그 자체가 과적합이라, 학습셋에서 떼어낸 dev로만 한다.
    """
    grid = np.arange(0.05, 0.96, 0.05)
    out = np.full(len(names), 0.5)
    for j in range(len(names)):
        best_f1, best_t = -1.0, 0.5
        true = labels[:, j] > 0.5
        for t in grid:
            pred = probs[:, j] >= t
            tp = int((pred & true).sum()); fp = int((pred & ~true).sum()); fn = int((~pred & true).sum())
            _, _, f1 = _prf(tp, fp, fn)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        out[j] = best_t
    return out


def source_conditional_baseline(records: list[dict], names: list[str]) -> dict:
    """**출처별 최빈 조 집합**을 찍는 상수 예측기 — 모델이 진짜로 넘어야 할 선.

    무조건 상수(top1/top3)는 하한일 뿐이다. 이 데이터는 출처 교락이 크다 —
    `ftc_case`는 거의 전부 조를 지목하고 `standard_contract`는 거의 전부 빈 배열이라,
    **출처만 알면 라벨을 상당 부분 맞힐 수 있다**(학습셋 77.8% / 평가셋 90.9% 전례).

    그래서 "상수를 이겼다"로는 부족하다. 이 값을 못 넘으면 모델이 배운 게 조항 내용이
    아니라 문체일 수 있다. 상세 측정은 `backend/eval/confound_articles.py`.
    """
    by_src: dict[str, Counter] = {}
    for r in records:
        key = frozenset(a for a in r["articles"] if a in names)
        by_src.setdefault(r.get("source", "?"), Counter())[key] += 1
    majority = {s: c.most_common(1)[0][0] for s, c in by_src.items()}

    f1s = []
    for r in records:
        g = {a for a in r["articles"] if a in names}
        p = set(majority.get(r.get("source", "?"), frozenset()))
        if not p and not g:
            f1s.append(1.0)
            continue
        inter = p & g
        pr = len(inter) / len(p) if p else 0.0
        rc = len(inter) / len(g) if g else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return {"f1": float(np.mean(f1s)) if f1s else 0.0,
            "per_source": {s: sorted(majority[s]) or ["(위반 없음)"] for s in majority}}


def constant_baseline_f1(records: list[dict], names: list[str], k: int) -> dict:
    """"항상 상위 k개 조를 찍기"의 per-sample F1. 모델은 이걸 넘어야 한다."""
    freq = Counter(a for r in records for a in r["articles"] if a in names)
    top = [a for a, _ in freq.most_common(k)]
    f1s = []
    for r in records:
        g = {a for a in r["articles"] if a in names}
        p = set(top)
        inter = p & g
        pr = len(inter) / len(p) if p else 0.0
        rc = len(inter) / len(g) if g else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return {"articles": top, "f1": float(np.mean(f1s)) if f1s else 0.0}


@torch.no_grad()
def infer(model, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    P, Y, loss_sum, n = [], [], 0.0, 0
    crit = nn.BCEWithLogitsLoss()
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                       batch["token_type_ids"].to(device))
        y = batch["labels"].to(device)
        loss_sum += crit(logits, y).item(); n += 1
        P.append(torch.sigmoid(logits).cpu().numpy()); Y.append(y.cpu().numpy())
    return np.concatenate(P), np.concatenate(Y), (loss_sum / n if n else 0.0)


# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="약관규제법 조 multi-label 인코더 학습")
    ap.add_argument("--labels", type=str, default=str(LABELED_PATH), help="라벨 JSONL 경로")
    ap.add_argument("--model-dir", type=str, default=str(PROJECT_ROOT / "models/_article"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--test-ratio", type=float, default=0.2, help="dev split 비율(학습셋에서 분리)")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--min-support", type=int, default=5, help="헤드에 둘 조의 최소 등장 횟수")
    ap.add_argument("--fulltext-augment", action="store_true",
                    help="evidence_span 증강을 켠다(기본 OFF — 누수 위험, 기준선 확보 후 A/B로 켤 것)")
    ap.add_argument("--train-subsample", type=int, default=0, metavar="건수",
                    help="학습 레코드를 이 개수로 문서 단위 부분표집한다(dev는 고정). "
                         "학습 곡선용 — dev까지 함께 줄이면 곡선이 학습량이 아니라 "
                         "dev 표본을 읽게 된다")
    ap.add_argument("--negative-holdout", type=int, default=0, metavar="문서수",
                    help="표준계약서 문서 N개를 학습에서 빼 음성 평가셋으로 쓴다. "
                         "--negative-ratio보다 먼저, 고정 시드로 갈라지므로 모든 팔이 같은 "
                         "평가셋을 쓴다. FTC gold만으로는 ratio 비교가 편향된다 "
                         "(split_negative_holdout 참고)")
    ap.add_argument("--negative-ratio", type=float, default=None, metavar="비율",
                    help="FTC 대비 표준계약서(음성 표본) 비율. 미지정이면 시드 구성 그대로. "
                         "빈 라벨이 과반이면 모델이 아무것도 예측하지 않는 쪽으로 끌리므로 "
                         "0.0 / 0.25 / 0.5 / 1.0 을 쓸어볼 것 (apply_negative_ratio 참고)")
    ap.add_argument("--label-source", choices=["agreed", "forward"], default="agreed",
                    help="어느 조 라벨로 학습할지. agreed=forward∩verify(정밀↑/거짓음성↑), "
                         "forward=forward만(재현↑/거짓양성↑). F1은 동률이라 재봐야 안다")
    ap.add_argument("--dry-run", action="store_true", help="라벨 없이 형상·손실만 검증")
    args = ap.parse_args(argv)

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")

    gold = load_ftc_gold()
    logger.info(f"========== 조 multi-label 학습 | device={device} | seed={args.seed} ==========")
    logger.info(f"  held-out gold(FTC 단일조항): {len(gold)}건 — 학습·임계값 튜닝에 쓰지 않는다")

    if args.dry_run:
        records = [{"text": g["text"], "articles": g["articles"], "group": f"dry-{i}", "source": "dry"}
                   for i, g in enumerate(gold)]
        logger.warning("  --dry-run: gold를 학습 데이터로 흉내 내 형상만 검증한다(성능 수치는 무의미)")
    else:
        records = load_article_records(Path(args.labels), augment=args.fulltext_augment,
                                       label_source=args.label_source)
        logger.info(f"  라벨 데이터 {len(records)}건 ({args.labels}) | label_source={args.label_source}")
        records = exclude_gold_documents(records, gold)
        # 순서가 중요하다 — 음성 held-out을 ratio보다 **먼저**, ratio와 무관한 고정 시드로
        # 갈라야 모든 팔이 같은 평가셋을 본다.
        records, neg_holdout = split_negative_holdout(records, args.negative_holdout, 42)
        records = apply_negative_ratio(records, args.negative_ratio, args.seed)

    if not records:
        raise SystemExit("학습 레코드가 없다 — 라벨 생성이 끝났는지 확인할 것")

    counts = Counter(a for r in records for a in r["articles"])
    names = article_labels(dict(counts), args.min_support)
    folded = sorted(set(ARTICLE_IDS) - set(names), key=lambda a: int(a.strip("제조")))
    logger.info(f"  조 분포: {dict(counts.most_common())}")
    logger.info(f"  헤드 라벨({len(names)}): {names}")
    logger.info(f"  접힌 조(support<{args.min_support}): {folded} — 표본이 없어 학습이 안 되고 macro만 흔든다")

    # 층화는 "조가 하나라도 있는가"로 — 조 조합은 경우의 수가 많아 층화가 불가능하다
    for r in records:
        r["has_article"] = bool([a for a in r["articles"] if a in names])
    train_recs, dev_recs = split_by_document(records, args.test_ratio, args.seed,
                                             stratify_key="has_article")

    # 학습 곡선용 부분표집 — **분할 뒤에** 한다. dev는 크기와 무관하게 고정돼야
    # 임계값·체크포인트 선택이 흔들리지 않는다. 문서 단위인 것은 형제 조항 누수 때문.
    if args.train_subsample and args.train_subsample < len(train_recs):
        by_doc: dict[str, list[dict]] = defaultdict(list)
        for r in train_recs:
            by_doc[r["group"]].append(r)
        docs = sorted(by_doc)
        random.Random(args.seed).shuffle(docs)
        picked: list[dict] = []
        for d in docs:
            if len(picked) >= args.train_subsample:
                break
            picked.extend(by_doc[d])
        logger.info(f"  --train-subsample: 학습 {len(train_recs)}→{len(picked)}건 "
                    f"(문서 {len(by_doc)}→{len({r['group'] for r in picked})}개) | dev {len(dev_recs)}건 고정")
        train_recs = picked
    logger.info(f"  학습 {len(train_recs)} / dev {len(dev_recs)}  "
                f"(dev는 임계값 튜닝·체크포인트 선택 전용, gold와 별개)")

    baselines = {}
    for k in (1, 3):
        b = constant_baseline_f1(dev_recs, names, k)
        baselines[f"top{k}"] = b
        logger.info(f"  [상수 기준선] 항상 {b['articles']} → dev per-sample F1 {b['f1']*100:.1f}%")
    cond = source_conditional_baseline(dev_recs, names)
    baselines["source_conditional"] = cond
    logger.info(f"  [상수 기준선] **출처별 최빈 조 집합** → dev per-sample F1 {cond['f1']*100:.1f}%"
                f"  ← 모델이 넘어야 할 진짜 선")
    for s, arts in cond["per_source"].items():
        logger.info(f"      {s:<20} → {arts}")
    _BAR = max(b["f1"] for b in baselines.values())

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tl = DataLoader(ArticleDataset(train_recs, tokenizer, args.max_len, names),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dl = DataLoader(ArticleDataset(dev_recs, tokenizer, args.max_len, names),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ArticleMultiLabelElectra(BASE_MODEL, names).to(device)
    pos_w = article_pos_weight(dict(counts), names, len(train_recs)).to(device)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    logger.info(f"  pos_weight: {dict(zip(names, [round(x, 1) for x in pos_w.tolist()]))}")

    model_dir = Path(args.model_dir)
    best, history = -1.0, []
    best_thresholds: dict[str, float] = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in tl:
            optim.zero_grad()
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                           batch["token_type_ids"].to(device))
            loss = crit(logits, batch["labels"].to(device))
            loss.backward(); optim.step()
            tot += loss.item(); nb += 1
        train_loss = tot / max(nb, 1)

        probs, labels, dev_loss = infer(model, dl, device)
        thr = tune_thresholds(probs, labels, names)      # dev에서만 튜닝
        m = per_article_metrics(probs, labels, names, thr)
        m.update({"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(dev_loss, 4)})
        history.append(m)

        detail = " ".join(f"{a}={v['f1']:.2f}(n={v['support']})" for a, v in m["per_article"].items())
        logger.info(f"  Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} val_loss={dev_loss:.4f} "
                    f"| macro_F1={m['macro_f1']:.4f} | 제6조제외={m['macro_f1_excl_majority']:.4f} "
                    f"| macro 대상 {m['n_scored']}개 조")
        logger.info(f"    {detail}")
        if m["zero_support_articles"]:
            logger.info(f"    macro 제외(dev support 0): {m['zero_support_articles']} "
                        f"— 측정값이 아니라 상수 0이라 뺀다")

        # 체크포인트 기준은 **제6조 제외 macro** — 다수 클래스로 부풀지 않는 값이다
        if m["macro_f1_excl_majority"] > best:
            best = m["macro_f1_excl_majority"]
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(model_dir); tokenizer.save_pretrained(model_dir)
            np.save(model_dir / "thresholds.npy", thr)
            best_thresholds = {a: float(t) for a, t in zip(names, thr)}
            logger.info(f"    모델 저장(best 제6조제외 macro={best:.4f}): {model_dir}")

    save_json({
        "base_model": BASE_MODEL, "article_names": names, "folded_articles": folded,
        "train_samples": len(train_recs), "dev_samples": len(dev_recs),
        "heldout_gold_samples": len(gold),
        "checkpoint_criterion": "dev macro_f1_excl_majority (제6조 제외) — held-out gold는 미사용",
        "threshold_tuned_on": "dev split only",
        # gold 채점은 이 값을 **그대로 얼려 써야 한다.** 채점 시점에 다시 튜닝하면
        # "임계값은 dev에서만"이라는 규칙을 우회하는 셈이 된다.
        "thresholds": best_thresholds,
        "augment": bool(args.fulltext_augment),
        "label_source": args.label_source,
        "negative_ratio": args.negative_ratio,
        "negative_holdout_docs": args.negative_holdout,
        "train_subsample": args.train_subsample,
        "best_macro_f1_excl_majority": best,
        "constant_baselines": baselines,
        "baseline_to_beat": _BAR,
        "train_config": {"batch_size": args.batch_size, "lr": args.lr, "max_len": args.max_len,
                         "epochs": args.epochs, "min_support": args.min_support},
        "history": history,
    }, model_dir / "metrics.json")

    try:
        from backend.training.plot_history import plot_history
        plot_history(model_dir)
    except Exception as e:
        logger.warning(f"  학습 곡선 생성 실패(학습 결과에는 영향 없음): {e}")

    logger.info(f"========== 완료 | best 제6조제외 macro={best:.4f} ==========")
    logger.info("  ※ held-out gold 채점은 별도 스크립트로 — 학습 루프에서 보면 그 순간 오염된다")


if __name__ == "__main__":
    main()
