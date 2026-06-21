# backend/domain/config.py
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parents[3]  # ContractRisk-Guard/

LAW_DIR    = Path(os.environ.get("LAW_RAW_DIR",  str(_ROOT / "data/raw/law")))
PREC_DIR   = Path(os.environ.get("PREC_RAW_DIR", str(_ROOT / "data/raw/case")))
EXPC_DIR   = Path(os.environ.get("EXPC_RAW_DIR", str(_ROOT / "data/raw/commentary")))
DOMAIN_DIR = Path(os.environ.get("DOMAIN_DIR",   str(_ROOT / "data/domain")))

ALLOWED_LAW_NAMES: set[str] = {
    "민법",
    "상법",
    "약관의 규제에 관한 법률",
    "할부거래에 관한 법률",
    "방문판매 등에 관한 법률",
    "전자상거래 등에서의 소비자보호에 관한 법률",
    "소비자기본법",
}

# benchmark.py 전용 — 판례 임베딩 모델 비교 시 사용
PREC_DENSE_QUERIES: list[str] = [
    "약관 불공정 조항 소비자 계약 해지 효력",
    "책임제한 약관 조항 무효 소비자 보호",
    "약관의 규제에 관한 법률 위반 해지 조항",
    "위약금 약관 조항 효력 계약 해제",
    "표준약관 소비자 해지권 제한 유효",
    "전자상거래 방문판매 약관 불공정 손해배상 면책",
]

# 판례 키워드 필터 — 사건종류명 제외 후 적용
PREC_KEYWORDS: list[str] = [
    "약관", "계약 해지", "계약해지", "책임제한", "위약금",
    "면책 조항", "불공정 조항", "배상책임 제한", "해지 조항",
]

