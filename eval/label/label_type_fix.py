# eval/label/label_type_fix.py
"""라벨 문제 유형을 정답으로 고치는 규칙 (paper_plan §7-27, 2026-10-13) — 순수 함수, 무거운 import 없음.

유형 (AI Hub 조항 하나의 Qwen 첫 판정 F, 전문가 정답 ref, 위반 여부 viol):
    fp   잘못된 라벨  — viol=False 인데 F가 비어 있지 않다           → 고치면 빈 라벨
    fn   누락된 라벨  — viol=True 인데 F가 비어 있다                 → 고치면 정답 조
    mix  혼합된 라벨  — viol=True, F가 비어 있지 않고 F != ref       → 고치면 정답 조
    ok   이미 맞음
팔은 어떤 유형을 정답으로 바꾸는지만 다르다. 같은 조항(텍스트)·같은 개수라 입력은 모든 팔에서 같고 라벨만 다르다.
"""

ARMS: dict[str, frozenset[str]] = {
    "OQ": frozenset(),                              # Qwen 라벨 그대로
    "OFP": frozenset({"fp"}),                       # 잘못된 라벨만 고침
    "OFN": frozenset({"fn"}),                       # 누락된 라벨만 고침
    "OMX": frozenset({"mix"}),                      # 혼합된 라벨만 고침
    "OG": frozenset({"fp", "fn", "mix"}),           # 전부 고침 = 전문가 정답
}


def noise_kind(F: set[str], ref: set[str], viol: bool) -> str:
    if not viol:
        return "fp" if F else "ok"
    if not F:
        return "fn"
    return "ok" if F == ref else "mix"


def fixed_labels(F: set[str], ref: set[str], viol: bool, fix: frozenset[str]) -> list[str]:
    """`fix`에 든 유형이면 정답으로 바꾸고, 아니면 Qwen 라벨 그대로 둔다. 순서는 조 번호순."""
    kind = noise_kind(F, ref, viol)
    out = set(ref) if kind in fix and kind != "fp" else (set() if kind == "fp" and "fp" in fix else set(F))
    return sorted(out, key=lambda a: int(a.strip("제조")))
