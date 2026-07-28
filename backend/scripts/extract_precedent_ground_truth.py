# backend/scripts/extract_precedent_ground_truth.py
"""
법원 판례에서 약관 조항 무효/유효 판결을 근거로 삼는 제3의(사법부) ground-truth 후보 추출.

FTC(행정부, 시정조치·표준계약서)만으로는 Medium 신호가 약해서(위반_유형 개수
프록시, 스팟체크에서 반례 확인됨), 완전히 독립된 권위 있는 기관인 법원 판결을
추가한다.

data/domain/case/(1,995건) 중 약관규제법을 직접 다루고, 해지·책임제한 도메인
키워드가 있고, "무효" 여부가 쟁점인 판례는 23건이었다. 정규식으로 판결 결론을
판별해봤지만("무효라고 볼 수 없다" 류 부정 표현 탐지) 표현이 너무 다양해서
23건 중 1건만 잡혔다 — 이 정도 표본 크기(23건)에서는 일반화된 파서보다 직접
읽고 판별하는 게 낫다고 판단해, 아래 _CURATED_VERDICTS를 수작업으로 만들었다.

수작업 기준:
  - 파기환송(대법원이 원심을 뒤집었지만 하급심에서 최종 결론이 다시 나야 하는
    경우, 예: 2건)은 "확정"이 아니므로 제외 — 이 프로젝트의 다른 라벨(FTC
    시정조치, 표준계약서)이 전부 "확정된" 권위 신호인 것과 일관성을 맞춤
  - 한 판례 안에 조항이 여러 개이고 결론이 갈리는 경우(예: 177679/177678/176454
    — 이자반환 면책조항은 원칙 무효, 위약금 해당분은 유효로 갈림) 제외
  - 무효/유효 판단 자체가 아니라 다른 쟁점(적용법조 배제, 사실관계 다툼)인
    경우 제외
  - 법원이 "무효"라고 확정한 케이스는 보수적으로 Medium(개별 사건 판단 —
    FTC 시정조치의 확정 위반만큼 광범위한 것은 아님)으로 잡는다. High로 볼 만한
    강한 표현(예: 연 60% 연체료)이 있어도 이 규칙을 예외 없이 유지한다 —
    임의로 High를 주면 그 자체가 새로운 자의적 프록시가 되기 때문.

실행: python -m backend.scripts.extract_precedent_ground_truth
"""

import json
from pathlib import Path

from backend.labeling.seed import classify_domain
from backend.utils import PROJECT_ROOT, load_logger, save_jsonl

logger = load_logger("extract_precedent_ground_truth.log")

CASE_DIR = PROJECT_ROOT / "data/domain/case"
OUT_PATH = PROJECT_ROOT / "data/eval/candidates/precedent_candidates.jsonl"

# case_id -> (risk_level, 분쟁 조항 텍스트, 비고)
# 판시사항/판결요지를 직접 읽고 수작업으로 확정한 매핑 — 재현을 위해 case_id를 남긴다.
_CURATED_VERDICTS: dict[str, tuple[str, str, str]] = {
    "174951": ("Low", "월 단위로 가입하여 이용하는 VOD 서비스는 가입 후 이용자가 해지할 때까지는 서비스가 제공되며, 이에 따른 요금이 부과된다", "종합유선방송 VOD 해지조항, 무효 아님(원고 패소 확정)"),
    "161640": ("Low", "리스계약에서 정한 계약해지사유가 발생하면 甲 회사의 요청에 따라 乙이 리스물건의 상태 및 존재 유무에 상관없이 리스계약에서 정한 규정손해금을 매입대금으로 하여 무조건 리스물건을 매수하여야 한다", "리스 재매입약정 해지조항, 무효로 보기 어렵다(확정)"),
    "185523": ("Low", "고객의 티머니 카드 분실 또는 도난 시 기 저장된 금액과 카드 값은 지급 받으실 수 없습니다", "선불카드 분실면책조항, 약관법 위반 아님(청구기각 확정)"),
    "177820": ("Medium", "甲은 상조계약을 해제할 수 없고 계약이 해제되었다고 하더라도 약관에 따른 해약환급금만 지급할 의무가 있다", "상조계약 해약환급금조항, 무효 확정"),
    "164520": ("Medium", "위탁자는 수탁자의 동의 없이 신탁계약을 해제할 수 없다", "저작권신탁계약 해지권 배제조항, 무효 확정"),
    "140172": ("Medium", "임차인의 월차임 연체에 대하여 월 5%(연 60%)의 연체료를 부담시킨 계약조항 및 임차인의 월차임 연체 등을 이유로 계약을 해지한 경우 임차인에게 임대차보증금의 10%를 위약금으로 지급하도록 한 계약조항", "임대차 연체료·해지위약금조항, 무효 확정"),
    "172304": ("Medium", "분양계약 해제 시 상가개발비를 어떠한 경우에도 반환하지 않는다고 정한 조항", "상가분양 개발비 미반환조항, 무효 확정"),
    "194064": ("Medium", "가스공급기간을 지키지 않은 때에 가스공급자가 부담한 시설비의 2배에 해당하는 금액을 가스공급자에게 배상하여야 한다", "LPG 공급계약 손해배상액예정조항, 약관법 제8조 해당(무효 취지) 확정"),
    "85770": ("Medium", "임대차목적물의 명도 또는 원상복구 지연에 따른 배상금 조항", "임대차 명도지연 배상금조항, 무효 확정"),
}

_DOMAIN_OVERRIDE = {
    "161640": "해지_조항", "174951": "해지_조항", "177820": "해지_조항", "164520": "해지_조항",
    "185523": "책임제한_조항", "194064": "책임제한_조항", "85770": "책임제한_조항",
}


def _build_record(case_id: str, risk_level: str, text: str, note: str, meta: dict) -> dict | None:
    domain = _DOMAIN_OVERRIDE.get(case_id) or classify_domain(text)
    if not domain:
        logger.warning(f"  {case_id}: 도메인 분류 실패, 제외 — {text[:40]}")
        return None
    return {
        "chunk_id":   f"precedent:{case_id}:0",
        "source":     "precedent",
        "doc_id":     case_id,
        "text":       text,
        "domain":     domain,
        "risk_level": risk_level,
        "risk_basis": "court_ruling_validity",
        "metadata":   {**meta, "비고": note},
    }


def extract_precedent_candidates(case_dir: Path) -> list[dict]:
    records: list[dict] = []
    for case_id, (risk_level, text, note) in _CURATED_VERDICTS.items():
        path = case_dir / f"{case_id}.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        p = data.get("PrecService", data)
        meta = {"사건명": p.get("사건명", ""), "법원명": p.get("법원명", ""), "선고일자": p.get("선고일자", "")}
        record = _build_record(case_id, risk_level, text, note, meta)
        if record:
            records.append(record)
    return records


def main() -> None:
    logger.info("========== 판례 기반 ground-truth 후보 추출 시작(수작업 검증 목록 기반) ==========")
    records = extract_precedent_candidates(CASE_DIR)

    by_risk: dict[str, int] = {}
    for r in records:
        by_risk[r["risk_level"]] = by_risk.get(r["risk_level"], 0) + 1
        logger.info(f"  [{r['risk_level']}] {r['chunk_id']} | {r['metadata']['비고']}")

    save_jsonl(records, OUT_PATH)
    logger.info(f"  추출: {len(records)}건 | 분포: {by_risk}")
    logger.info(f"  저장: {OUT_PATH}")
    logger.info("========== 완료 ==========")


if __name__ == "__main__":
    main()
