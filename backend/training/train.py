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
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import ElectraTokenizerFast

from backend.model.electra import DualHeadElectra, DOMAIN_MAP, RISK_MAP, DOMAIN_NAMES, RISK_NAMES
from backend.utils import load_jsonl, load_logger, save_json, PROJECT_ROOT

logger = load_logger("train_koelectra.log")

SEED_PATH  = Path(os.environ.get("SEED_PATH",  str(PROJECT_ROOT / "data/labels/seed_labeled.jsonl")))
CLEAN_PATH = Path(os.environ.get("CLEAN_PATH", str(PROJECT_ROOT / "data/fb_check/clean.jsonl")))
BASE_MODEL = os.environ.get("BASE_MODEL", "monologg/koelectra-base-v3-discriminator")


def load_records(data_source: str) -> list[dict]:
    """학습 레코드를 로드하고 {text, domain, risk_level} 형태로 정규화한다."""
    if data_source == "seed":
        return load_jsonl(SEED_PATH)

    raw = load_jsonl(CLEAN_PATH)
    records = []
    for r in raw:
        evidence_span = r.get("evidence_span", "")
        if not evidence_span:
            continue
        records.append({
            "text":       evidence_span,
            "domain":     r["forward_domain"],
            "risk_level": r["forward_label"],
        })
    return records


class ClauseDataset(Dataset):
    """계약 조항 분류 데이터셋."""

    def __init__(self, records: list[dict], tokenizer: ElectraTokenizerFast, max_len: int) -> None:
        self.records   = records
        self.tokenizer = tokenizer
        self.max_len   = max_len

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
            "risk_label":     torch.tensor(RISK_MAP[rec["risk_level"]], dtype=torch.long),
        }


def train_epoch(
    model: DualHeadElectra,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    domain_criterion: nn.CrossEntropyLoss,
    risk_criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        optimizer.zero_grad()
        d_logits, r_logits = model(input_ids, attention_mask, token_type_ids)
        loss = domain_criterion(d_logits, batch["domain_label"].to(device)) + \
               risk_criterion(r_logits, batch["risk_label"].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(
    model: DualHeadElectra,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[int], list[int]]:
    model.eval()
    d_preds, d_labels, r_preds, r_labels = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            d_logits, r_logits = model(input_ids, attention_mask, token_type_ids)
            d_preds.extend(d_logits.argmax(dim=-1).cpu().tolist())
            d_labels.extend(batch["domain_label"].tolist())
            r_preds.extend(r_logits.argmax(dim=-1).cpu().tolist())
            r_labels.extend(batch["risk_label"].tolist())
    return d_preds, d_labels, r_preds, r_labels


def compute_metrics(
    d_preds: list[int],
    d_labels: list[int],
    r_preds: list[int],
    r_labels: list[int],
) -> dict[str, Any]:
    return {
        "domain_macro_f1": round(f1_score(d_labels, d_preds, average="macro"), 4),
        "risk_macro_f1":   round(f1_score(r_labels, r_preds, average="macro"), 4),
        "domain_report":   classification_report(d_labels, d_preds, target_names=DOMAIN_NAMES, output_dict=True),
        "risk_report":     classification_report(r_labels, r_preds, target_names=RISK_NAMES,   output_dict=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="KoELECTRA 분류 모델 학습")
    parser.add_argument("--data-source", choices=["seed", "clean"], default="seed",
                        help="학습 데이터 소스 (seed=1차 원본, clean=FB-Check CLEAN/2차 모델, 기본 seed)")
    parser.add_argument("--model-dir",   type=str,   default=None,
                        help="모델 저장 경로 (기본: seed→models/v1, clean→models/v2)")
    parser.add_argument("--epochs",     type=int,   default=5,    help="학습 에폭 수")
    parser.add_argument("--batch-size", type=int,   default=16,   help="배치 크기")
    parser.add_argument("--lr",         type=float, default=3e-5, help="학습률")
    parser.add_argument("--max-len",    type=int,   default=256,  help="최대 토큰 길이")
    parser.add_argument("--test-ratio", type=float, default=0.2,  help="검증 세트 비율")
    parser.add_argument("--gpu",        type=int,   default=1,    help="사용할 GPU 인덱스 (기본 1)")
    args = parser.parse_args()

    default_dir = "models/v2" if args.data_source == "clean" else "models/v1"
    MODEL_DIR = Path(args.model_dir or os.environ.get("MODEL_DIR", str(PROJECT_ROOT / default_dir)))

    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    logger.info(f"========== KoELECTRA 학습 시작 | device={device} | data_source={args.data_source} ==========")

    records = load_records(args.data_source)
    logger.info(f"  데이터: {len(records)}건 로드 (source={args.data_source})")

    train_recs, val_recs = train_test_split(
        records, test_size=args.test_ratio, random_state=42,
        stratify=[r["risk_level"] for r in records],
    )
    logger.info(f"  학습: {len(train_recs)}건 | 검증: {len(val_recs)}건")

    tokenizer    = ElectraTokenizerFast.from_pretrained(BASE_MODEL)
    train_loader = DataLoader(ClauseDataset(train_recs, tokenizer, args.max_len), batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(ClauseDataset(val_recs,   tokenizer, args.max_len), batch_size=args.batch_size, shuffle=False, num_workers=2)

    domain_labels_arr = np.array([DOMAIN_MAP[r["domain"]] for r in train_recs])
    risk_labels_arr   = np.array([RISK_MAP[r["risk_level"]] for r in train_recs])
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

    model          = DualHeadElectra(BASE_MODEL).to(device)
    optimizer      = torch.optim.AdamW(model.parameters(), lr=args.lr)
    domain_criterion = nn.CrossEntropyLoss(weight=domain_weights)
    risk_criterion   = nn.CrossEntropyLoss(weight=risk_weights)

    best_risk_f1 = 0.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, domain_criterion, risk_criterion, device)
        d_preds, d_labels, r_preds, r_labels = evaluate(model, val_loader, device)
        metrics = compute_metrics(d_preds, d_labels, r_preds, r_labels)
        metrics.update({"epoch": epoch, "train_loss": round(train_loss, 4)})
        history.append(metrics)
        logger.info(
            f"  Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} "
            f"| domain_f1={metrics['domain_macro_f1']:.4f} | risk_f1={metrics['risk_macro_f1']:.4f}"
        )
        if metrics["risk_macro_f1"] > best_risk_f1:
            best_risk_f1 = metrics["risk_macro_f1"]
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            logger.info(f"  모델 저장 (best risk_f1={best_risk_f1:.4f}): {MODEL_DIR}")

    save_json(
        {
            "base_model": BASE_MODEL, "train_samples": len(train_recs), "val_samples": len(val_recs),
            "epochs": args.epochs, "best_risk_macro_f1": best_risk_f1,
            "train_config": {"batch_size": args.batch_size, "lr": args.lr, "max_len": args.max_len},
            "history": history,
        },
        MODEL_DIR / "metrics.json",
    )
    logger.info(f"========== 학습 완료 | best_risk_f1={best_risk_f1:.4f} ==========")


if __name__ == "__main__":
    main()
