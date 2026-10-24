# tests/model/test_label_type_fix.py
"""라벨 문제 유형별 정답 수정 규칙 (paper_plan §7-27) — 팔마다 어떤 유형만 바뀌는지 고정한다."""

from eval.label.label_type_fix import ARMS, fixed_labels, noise_kind

# (Qwen 라벨 F, 전문가 정답 ref, 위반 여부)
FP = ({"제8조"}, set(), False)             # 공정 조항에 위반 라벨
FN = (set(), {"제6조"}, True)              # 위반인데 빈 라벨
MX = ({"제9조", "제6조"}, {"제9조"}, True)  # 맞는 조에 틀린 조 덧붙임
OK = ({"제7조"}, {"제7조"}, True)
FAIR_OK = (set(), set(), False)


def test_noise_kind():
    assert [noise_kind(*x) for x in (FP, FN, MX, OK, FAIR_OK)] == ["fp", "fn", "mix", "ok", "ok"]


def test_each_arm_changes_only_its_own_type():
    for arm, fix in ARMS.items():
        got = {k: fixed_labels(*x, fix) for k, x in {"fp": FP, "fn": FN, "mix": MX, "ok": OK}.items()}
        assert got["ok"] == ["제7조"], arm                                           # 맞는 라벨은 어느 팔에서도 그대로
        assert got["fp"] == ([] if "fp" in fix else ["제8조"]), arm
        assert got["fn"] == (["제6조"] if "fn" in fix else []), arm
        assert got["mix"] == (["제9조"] if "mix" in fix else ["제6조", "제9조"]), arm


def test_gold_arm_equals_expert_labels_and_baseline_equals_qwen():
    for F, ref, viol in (FP, FN, MX, OK, FAIR_OK):
        assert fixed_labels(F, ref, viol, ARMS["OG"]) == sorted(ref, key=lambda a: int(a.strip("제조")))
        assert fixed_labels(F, ref, viol, ARMS["OQ"]) == sorted(F, key=lambda a: int(a.strip("제조")))
