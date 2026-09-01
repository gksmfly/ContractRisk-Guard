# backend/training/train.py
"""
KoELECTRA 분류 모델 학습 스크립트

계약 조항 텍스트 → (domain, risk_level) 분류 모델을 학습한다.

데이터 소스 (--data-source):
    seed  - data/labels/seed_labeled.jsonl 원본 (1차 모델, FB-Check 검증 이전)
            텍스트가 FTC 판례 발췌문 전체(평균 600자+, 최대 2만자)라서
            실제 서비스가 넣는 짧은 개별 조항(100~150자)과 길이 분포가 다르고,
            길이와 risk_level이 상관돼 있어(Low가 유의하게 짧음) 모델이
            "내용"이 아니라 "길이"로 지름길 학습을 할 위험이 있다.
    clean - data/fb_check/clean.jsonl (FB-Check CLEAN, 2차 모델/Data Flywheel)
            text 대신 evidence_span(평균 49자, Consistency Verification으로
            "이 근거만으로도 같은 라벨이 재현됨"이 검증된 짧은 인용구)을 학습
            텍스트로 사용해 실제 서비스 입력 길이에 맞춘다.

출력:
    models/{v1,v2}/            - 모델 가중치 및 토크나이저
    models/{v1,v2}/metrics.json - 평가 지표 (논문 실험 테이블용)

사용법:
    python -m backend.training.train --data-source seed
    python -m backend.training.train --data-source clean --epochs 10 --batch-size 32 --gpu 1
"""

import argparse
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer  # 백본 교체 비교를 위해 계열 비의존으로 로드한다

from backend.model.electra import DOMAIN_MAP, DOMAIN_NAMES, RISK_SCHEMES, DualHeadElectra, risk_scheme
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("train_koelectra.log")

SEED_PATH  = Path(os.environ.get("SEED_PATH",  str(PROJECT_ROOT / "data/labels/seed_labeled.jsonl")))
CLEAN_PATH = Path(os.environ.get("CLEAN_PATH", str(PROJECT_ROOT / "data/fb_check/clean.jsonl")))
GT3_PATH   = Path(os.environ.get("GT3_PATH",   str(PROJECT_ROOT / "data/eval/ground_truth_3class.jsonl")))
BASE_MODEL = os.environ.get("BASE_MODEL", "monologg/koelectra-base-v3-discriminator")


def _document_group(chunk_id: str | None, fallback: str) -> str:
    """`chunk_id`(`source:문서ID:순번`)에서 문서 식별자(`source:문서ID`)를 뽑는다.

    예: `ftc_case:2021약관0294:0` → `ftc_case:2021약관0294`
    구조가 다르면 chunk_id를 그대로(=조항 단위로) 쓴다 — 문서 단위로 묶지 못하더라도
    최소한 증강 쌍은 붙어 있게 된다.
    """
    if not chunk_id:
        return fallback
    parts = chunk_id.split(":")
    return ":".join(parts[:2]) if len(parts) >= 3 else chunk_id


def load_records(data_source: str, fulltext_augment: bool = True,
                 allowed_risk: set[str] | None = None) -> list[dict]:
    """학습 레코드를 로드하고 {text, domain, risk_level, group} 형태로 정규화한다.

    `group`은 **원 문서 식별자**이며 학습/검증 분할의 단위다. 두 단계의 누수를 동시에 막는다:

    1. **증강 쌍 누수** — fulltext 증강이 조항 하나를 레코드 둘(evidence_span, 원문)로
       늘리는데, 이 둘이 갈라지면 검증 세트가 학습 문장을 부분 문자열로 그대로 담는다
       (원문 ⊃ evidence_span). 검증 점수가 일반화가 아니라 암기를 재게 된다.
    2. **형제 조항 누수** — `chunk_id`는 `source:문서ID:순번` 구조라 한 문서에서 여러 조항이
       나온다(694조항 → 361문서, 72.3%가 형제 있음). 그리고 **문서 내 라벨이 전부 같은
       문서가 83.4%**다. 형제가 갈라지면 모델이 당사자명·서식·문체로 문서를 식별해
       라벨을 맞힐 수 있다.

    그래서 조항 단위가 아니라 **문서 단위**로 묶는다. 실제로 이 누수를 둔 채 학습했을 때
    내부 검증 F1은 0.92~0.97인데 외부 평가셋 정확도는 45.7%였다.

    allowed_risk를 주면 그 라벨만 남긴다 — 2class 학습에서 Medium을 제외하는 용도
    (제외 근거는 `backend/model/electra.py`의 RISK_SCHEMES 주석 참고).
    """
    if data_source == "gt3":
        # data/eval/ground_truth_3class.jsonl — clean.jsonl과 **다른 라벨 정의**를 쓴다
        # (High = FTC 케이스의 위반유형 개수 2개 이상 / Low = 표준계약서 provenance).
        # 두 정의 중 어느 쪽이 일반화되는지 교차 학습으로 비교하기 위한 소스다.
        # evidence_span이 없으므로 증강도 없다.
        rows = load_jsonl(GT3_PATH)
        out = []
        for i, r in enumerate(rows):
            if r.get("domain") not in DOMAIN_MAP:
                continue
            if allowed_risk is not None and r.get("risk_level") not in allowed_risk:
                continue
            out.append({"text": r.get("text") or "", "domain": r["domain"],
                        "risk_level": r["risk_level"],
                        "group": _document_group(r.get("chunk_id"), r.get("doc_id") or f"gt3-{i}")})
        return [r for r in out if r["text"].strip()]

    if data_source == "seed":
        rows = load_jsonl(SEED_PATH)
        # seed.py가 키워드 도메인 필터를 뗀 뒤로 risk_level이 None인 레코드가 섞인다
        # (해지·책임제한 정규식이 못 다루는 유형 — LLM 판정 대기). 학습에는 쓸 수 없다.
        pending = sum(1 for r in rows if not r.get("risk_level"))
        if pending:
            logger.info(f"  seed 중 risk_level 미판정 {pending}건 제외 (LLM 라벨링 대기)")
        rows = [r for r in rows if r.get("risk_level")]
        rows = [r for r in rows if allowed_risk is None or r.get("risk_level") in allowed_risk]
        for i, r in enumerate(rows):
            r.setdefault("group", _document_group(r.get("chunk_id"), f"seed-{i}"))
        return rows

    raw = load_jsonl(CLEAN_PATH)
    records = []
    for row_idx, r in enumerate(raw):
        group = _document_group(r.get("chunk_id"), r.get("fb_id") or f"clean-{row_idx}")
        evidence_span = r.get("evidence_span", "")
        if not evidence_span:
            continue
        # 드물게 GPT가 domain 문자열을 깨뜨려 반환한다(예: "해 책임제한_조항") — 유효하지
        # 않은 값은 DOMAIN_MAP에서 KeyError를 내므로 학습 대상에서 제외한다.
        domain = r["forward_domain"]
        if domain not in DOMAIN_MAP:
            continue
        risk_level = r.get("final_label", r["forward_label"])
        if allowed_risk is not None and risk_level not in allowed_risk:
            continue
        records.append({
            "text":       evidence_span,
            "domain":     domain,
            # final_label은 forward/verify/backward 2/3 다수결 결과 (forward_label과 다를 수 있음).
            # 옛 clean.jsonl(다수결 로직 반영 전)엔 final_label이 없어 forward_label로 폴백한다.
            "risk_level": risk_level,
            "group":      group,
        })
        if not fulltext_augment:
            continue
        # evidence_span(평균 40자)만 학습하면 실서비스에서 evidence_span을 못 뽑은 조항엔
        # 원문(평균 230자+)이 그대로 들어가는데, 모델이 그 길이의 입력을 한 번도 본 적이
        # 없어 정확도가 폭락한다(ground truth 평가에서 evidence_span 69.6% vs 원문
        # 대체 23.0%로 확인됨) — 같은 라벨로 원문도 별도 학습 예시로 추가해 두 길이
        # 분포 모두에 대응하게 한다.
        full_text = r.get("text", "")
        if full_text and full_text != evidence_span:
            records.append({
                "text":       full_text,
                "domain":     domain,
                "risk_level": risk_level,
                "group":      group,   # evidence_span 레코드와 반드시 같은 쪽으로 가야 한다
            })
    return records


_MIN_LEAK_LEN = 30  # 이보다 짧은 문구는 우연히 겹치므로 누수 판정에서 제외


def _count_contained(val_recs: list[dict], train_texts: set[str]) -> list[int]:
    """검증 레코드 중 학습 문장을 통째로 품고 있는 것들의 인덱스."""
    long_train = [t for t in train_texts if len(t) >= _MIN_LEAK_LEN]
    return [i for i, r in enumerate(val_recs) if any(t in r["text"] for t in long_train)]


def split_by_document(records: list[dict], test_ratio: float, seed: int,
                      stratify_key: str = "risk_level") -> tuple[list[dict], list[dict]]:
    """학습/검증을 **문서 단위로** 나누고, 남은 텍스트 누수까지 제거한다.

    이 데이터에는 누수 경로가 세 개 있고 하나씩 막아야 한다:

    1. **증강 쌍** — 조항 1개가 레코드 2개(evidence_span, 원문)가 되는데 원문이
       evidence_span을 부분 문자열로 담는다 → 문서 단위 묶기로 해결
    2. **형제 조항** — `chunk_id`가 `source:문서ID:순번`이라 한 문서에서 여러 조항이 나오고,
       문서 내 라벨이 전부 같은 문서가 83.4%다. 갈라지면 모델이 문서를 식별해 맞힐 수 있다
       → 문서 단위 묶기로 해결
    3. **문서 간 텍스트 중복** — 표준계약서들이 같은 조항 문구를 공유한다. 전체 1,388건 중
       중복이 769건(55.4%), 한 문구가 최대 49회. 문서가 안 겹쳐도 문장이 겹치므로 문서
       단위 분할로는 **못 막는다**(실측: 검증 260건 중 135건이 학습에 동일 텍스트 보유)
       → 중복 제거 + 잔여분 이동으로 해결

    중복 제거는 라벨을 바꾸지 않는다(동일 텍스트에 다른 라벨이 붙은 충돌 0건 확인).
    다만 **클래스 균형은 바뀐다** — 중복이 Low를 과대표집하고 있었다:
    Low 848/High 368/Medium 172 → Low 358/High 318/Medium 105. 중복은 새 정보를 주지
    않으면서 다수 클래스만 부풀리므로 제거가 맞다.
    """
    # (3) 문서 간 중복 제거 — 같은 텍스트는 한 번만 남긴다
    unique: dict[str, dict] = {}
    for r in records:
        unique.setdefault(r["text"].strip(), r)
    deduped = list(unique.values())
    if len(deduped) != len(records):
        logger.info(f"  텍스트 중복 제거: {len(records)} → {len(deduped)}건 "
                    f"(문서 간 공유 문구 제거 — 남겨두면 검증 세트로 새어 들어간다)")

    # (1)(2) 문서 단위 분할. 대표 라벨은 문서 내 최빈값(형제 조항 라벨이 다를 수 있음)
    groups = sorted({r["group"] for r in deduped})
    per_group: dict[str, list[str]] = {}
    for r in deduped:
        per_group.setdefault(r["group"], []).append(str(r.get(stratify_key)))
    group_label = {g: Counter(v).most_common(1)[0][0] for g, v in per_group.items()}

    train_g, val_g = train_test_split(
        groups, test_size=test_ratio, random_state=seed,
        stratify=[group_label[g] for g in groups],
    )
    train_g, val_g = set(train_g), set(val_g)
    train_recs = [r for r in deduped if r["group"] in train_g]
    val_recs   = [r for r in deduped if r["group"] in val_g]

    # 잔여 부분 포함(문서는 달라도 보일러플레이트 조각이 겹치는 경우)은 검증에서 학습으로
    # 옮긴다. 옮긴 레코드가 새로운 학습 문장이 되어 또 걸릴 수 있으므로 수렴할 때까지 반복.
    moved = 0
    for _ in range(5):
        train_texts = {r["text"].strip() for r in train_recs}
        leaking = _count_contained(val_recs, train_texts)
        if not leaking:
            break
        moved += len(leaking)
        leak_set = set(leaking)
        train_recs += [r for i, r in enumerate(val_recs) if i in leak_set]
        val_recs = [r for i, r in enumerate(val_recs) if i not in leak_set]
    if moved:
        logger.info(f"  잔여 텍스트 누수 {moved}건을 검증→학습으로 이동(문서 간 보일러플레이트)")

    logger.info(f"  문서 {len(groups)}개를 {1 - test_ratio:.0%}:{test_ratio:.0%}로 분할(문서 단위)")
    logger.info(f"    학습 {len(train_recs)}건/{len(train_g)}문서 | 검증 {len(val_recs)}건/{len(val_g)}문서")
    # 라벨 분포는 층화에 쓴 키로 찍는다 — 조 multi-label 경로에는 `risk_level`이 없다.
    logger.info(f"    학습 라벨 {Counter(str(r.get(stratify_key)) for r in train_recs).most_common()}")
    logger.info(f"    검증 라벨 {Counter(str(r.get(stratify_key)) for r in val_recs).most_common()}")

    # --- 누수 검증: 학습 로그에 항상 남긴다. 분할 로직을 누가 건드려도 즉시 드러나게 ---
    overlap = len(train_g & val_g)
    train_texts = {r["text"].strip() for r in train_recs}
    exact = sum(1 for r in val_recs if r["text"].strip() in train_texts)
    contained = len(_count_contained(val_recs, train_texts))
    level = logger.warning if (overlap or exact or contained) else logger.info
    level(f"    누수 점검 — 문서 교집합 {overlap} / 동일 텍스트 {exact} / 부분 포함 {contained} "
          f"(셋 다 0이어야 정상)")
    return train_recs, val_recs


class ClauseDataset(Dataset):
    """계약 조항 분류 데이터셋."""

    def __init__(self, records: list[dict], tokenizer: Any, max_len: int,
                 risk_map: dict[str, int]) -> None:
        self.records   = records
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.risk_map  = risk_map

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["text"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros(self.max_len, dtype=torch.long)).squeeze(0),
            "domain_label":   torch.tensor(DOMAIN_MAP[rec["domain"]], dtype=torch.long),
            "risk_label":     torch.tensor(self.risk_map[rec["risk_level"]], dtype=torch.long),
        }


def train_epoch(
    model: DualHeadElectra,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    domain_criterion: nn.CrossEntropyLoss,
    risk_criterion: nn.CrossEntropyLoss,
    device: torch.device,
    accum_steps: int = 1,
) -> float:
    """accum_steps > 1이면 그만큼 기울기를 모아서 한 번 갱신한다(기울기 누적).

    큰 백본(337M~568M)은 GPU 여유가 적을 때 배치 32가 안 들어간다. 배치만 줄이면
    유효 배치가 달라져 base급 모델과 비교가 흐려지므로, 배치를 줄인 만큼 누적해
    **실효 배치를 동일하게 유지**한다(예: 배치 8 × 누적 4 = 실효 32).
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(loader, start=1):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        d_logits, r_logits = model(input_ids, attention_mask, token_type_ids)
        loss = domain_criterion(d_logits, batch["domain_label"].to(device)) + \
               risk_criterion(r_logits, batch["risk_label"].to(device))
        total_loss += loss.item()
        (loss / accum_steps).backward()
        if step % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    if len(loader) % accum_steps != 0:   # 마지막 자투리 배치도 반영
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(loader)


def evaluate(
    model: DualHeadElectra,
    loader: DataLoader,
    device: torch.device,
    domain_criterion: nn.Module | None = None,
    risk_criterion: nn.Module | None = None,
) -> tuple[list[int], list[int], list[int], list[int], float | None]:
    """검증 예측과 **검증 손실**을 함께 낸다.

    검증 손실이 없으면 과적합을 볼 수 없다 — F1만 보면 "아직 오르는 중"과 "이미 외우기
    시작했는데 운 좋게 F1이 유지되는 중"을 구분할 수 없기 때문이다. 학습 손실과 같은
    criterion(클래스 가중치 포함)을 써야 두 곡선을 같은 축에서 비교할 수 있다.
    """
    model.eval()
    d_preds, d_labels, r_preds, r_labels = [], [], [], []
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            d_logits, r_logits = model(input_ids, attention_mask, token_type_ids)
            if domain_criterion is not None and risk_criterion is not None:
                loss = (domain_criterion(d_logits, batch["domain_label"].to(device))
                        + risk_criterion(r_logits, batch["risk_label"].to(device)))
                total_loss += loss.item()
                n_batches += 1
            d_preds.extend(d_logits.argmax(dim=-1).cpu().tolist())
            d_labels.extend(batch["domain_label"].tolist())
            r_preds.extend(r_logits.argmax(dim=-1).cpu().tolist())
            r_labels.extend(batch["risk_label"].tolist())
    val_loss = total_loss / n_batches if n_batches else None
    return d_preds, d_labels, r_preds, r_labels, val_loss


def compute_metrics(
    d_preds: list[int],
    d_labels: list[int],
    r_preds: list[int],
    r_labels: list[int],
    risk_names: list[str],
) -> dict[str, Any]:
    return {
        "domain_macro_f1": round(f1_score(d_labels, d_preds, average="macro"), 4),
        "risk_macro_f1":   round(f1_score(r_labels, r_preds, average="macro"), 4),
        "domain_report":   classification_report(d_labels, d_preds, target_names=DOMAIN_NAMES, output_dict=True),
        "risk_report":     classification_report(
            r_labels, r_preds, target_names=risk_names,
            labels=list(range(len(risk_names))), output_dict=True, zero_division=0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="KoELECTRA 분류 모델 학습")
    parser.add_argument("--data-source", choices=["seed", "clean", "gt3"], default="seed",
                        help="학습 데이터 소스 (seed=1차 원본, clean=FB-Check CLEAN/2차 모델, 기본 seed)")
    parser.add_argument("--model-dir",   type=str,   default=None,
                        help="모델 저장 경로 (기본: seed→models/v1, clean→models/v2)")
    parser.add_argument("--epochs",     type=int,   default=5,    help="학습 에폭 수")
    parser.add_argument("--batch-size", type=int,   default=16,   help="배치 크기")
    parser.add_argument("--lr",         type=float, default=3e-5, help="학습률")
    parser.add_argument("--max-len",    type=int,   default=256,  help="최대 토큰 길이")
    parser.add_argument("--test-ratio", type=float, default=0.2,  help="검증 세트 비율")
    parser.add_argument("--gpu",        type=int,   default=1,    help="사용할 GPU 인덱스 (기본 1)")
    parser.add_argument("--seed",       type=int,   default=42,   help="재현성을 위한 랜덤 시드")
    parser.add_argument("--accum-steps", type=int, default=1,
                        help="기울기 누적 스텝. GPU 여유가 적어 배치를 줄여야 할 때 "
                             "batch_size × accum_steps 가 다른 실험과 같아지도록 맞춘다")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader 워커 수. 0이면 메인 프로세스에서 로드한다 — 워커 교착이 "
                             "발생하면(진행 없이 do_poll/futex에서 멈춤) 0으로 두면 원인 자체가 사라진다. "
                             "이 데이터셋(1,216건)은 작아서 0이어도 속도 차이가 거의 없다")
    parser.add_argument("--risk-scheme", choices=list(RISK_SCHEMES), default="3class",
                        help="risk 라벨 체계. 2class는 Medium을 제외하고 High/Low만 학습한다 "
                             "(근거: backend/model/electra.py의 RISK_SCHEMES 주석)")
    parser.add_argument("--no-fulltext-augment", action="store_true",
                        help="evidence_span 외 원문(full text) 증강 학습 예시를 추가하지 않음 (v4 등 이전 버전 재현용)")
    args = parser.parse_args()

    # 분류기 헤드 초기화·배치 셔플·dropout이 전부 랜덤이라, 시드를 고정 안 하면
    # 같은 데이터로 다시 학습해도 결과가 달라진다 (특히 데이터가 작을수록 크게 흔들림).
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    default_dir = {"clean": "models/v2", "gt3": "models/_gt3"}.get(args.data_source, "models/v1")
    MODEL_DIR = Path(args.model_dir or os.environ.get("MODEL_DIR", str(PROJECT_ROOT / default_dir)))

    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    risk_map, _, risk_names = risk_scheme(args.risk_scheme)
    logger.info(f"========== KoELECTRA 학습 시작 | device={device} | data_source={args.data_source} "
                f"| seed={args.seed} | risk={args.risk_scheme}({'/'.join(risk_names)}) ==========")

    records = load_records(args.data_source, fulltext_augment=not args.no_fulltext_augment,
                           allowed_risk=set(risk_names))
    logger.info(f"  데이터: {len(records)}건 로드 (source={args.data_source})")
    logger.info(f"  라벨 분포: {Counter(r['risk_level'] for r in records).most_common()}")

    train_recs, val_recs = split_by_document(records, args.test_ratio, args.seed)

    tokenizer    = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_loader = DataLoader(ClauseDataset(train_recs, tokenizer, args.max_len, risk_map), batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(ClauseDataset(val_recs,   tokenizer, args.max_len, risk_map), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    domain_labels_arr = np.array([DOMAIN_MAP[r["domain"]] for r in train_recs])
    risk_labels_arr   = np.array([risk_map[r["risk_level"]] for r in train_recs])
    domain_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.unique(domain_labels_arr), y=domain_labels_arr),
        dtype=torch.float,
    ).to(device)
    risk_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.unique(risk_labels_arr), y=risk_labels_arr),
        dtype=torch.float,
    ).to(device)
    logger.info(f"  domain class_weight: {domain_weights.tolist()}")
    logger.info(f"  risk class_weight:   {risk_weights.tolist()}")

    model          = DualHeadElectra(BASE_MODEL, num_risk_labels=len(risk_map)).to(device)
    optimizer      = torch.optim.AdamW(model.parameters(), lr=args.lr)
    domain_criterion = nn.CrossEntropyLoss(weight=domain_weights)
    risk_criterion   = nn.CrossEntropyLoss(weight=risk_weights)

    best_risk_f1 = 0.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, domain_criterion, risk_criterion,
                                 device, accum_steps=args.accum_steps)
        d_preds, d_labels, r_preds, r_labels, val_loss = evaluate(
            model, val_loader, device, domain_criterion, risk_criterion
        )
        metrics = compute_metrics(d_preds, d_labels, r_preds, r_labels, risk_names)
        metrics.update({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4) if val_loss is not None else None,
        })
        history.append(metrics)
        logger.info(
            f"  Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | domain_f1={metrics['domain_macro_f1']:.4f} "
            f"| risk_f1={metrics['risk_macro_f1']:.4f}"
        )
        if metrics["risk_macro_f1"] > best_risk_f1:
            best_risk_f1 = metrics["risk_macro_f1"]
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            logger.info(f"  모델 저장 (best risk_f1={best_risk_f1:.4f}): {MODEL_DIR}")

    save_json(
        {
            "base_model": BASE_MODEL, "risk_scheme": args.risk_scheme, "risk_names": risk_names,
            "train_samples": len(train_recs), "val_samples": len(val_recs),
            "epochs": args.epochs, "best_risk_macro_f1": best_risk_f1,
            "train_config": {"batch_size": args.batch_size, "lr": args.lr, "max_len": args.max_len},
            "history": history,
        },
        MODEL_DIR / "metrics.json",
    )

    # 학습 곡선 PNG를 체크포인트 옆에 남긴다 — 숫자만으로는 과적합 시작 지점을 못 본다.
    # 그리기 실패가 학습 결과를 날리면 안 되므로 예외는 로그만 남기고 삼킨다.
    try:
        from backend.training.plot_history import diagnose, plot_history
        out = plot_history(MODEL_DIR)
        if out:
            logger.info(f"  학습 곡선: {out}")
            logger.info(f"  과적합 진단: {diagnose(history)['verdict']}")
    except Exception as e:
        logger.warning(f"  학습 곡선 생성 실패(학습 결과에는 영향 없음): {e}")

    logger.info(f"========== 학습 완료 | best_risk_f1={best_risk_f1:.4f} ==========")


if __name__ == "__main__":
    main()
