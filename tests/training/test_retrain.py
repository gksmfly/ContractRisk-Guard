"""
backend/training/retrain.py 단위 테스트 — GPU·모델 없이 도는 부분(데이터 확인·판정 규칙·인자 복원).

학습과 채점(`predict`, `evaluate_pair`)은 체크포인트와 GPU가 필요해 여기서 다루지 않는다.
"""

import json

import numpy as np
import pytest

from backend.training import retrain
from backend.training.train_article import split_negative_holdout

_LABELER = {"forward_model": "gpt-4o", "forward_prompt": "fwd-v5",
            "verify_model": "gpt-4o", "verify_prompt": "ver-v5"}


def _row(cid: str, source: str, arts: list[str], **over) -> dict:
    return {"chunk_id": cid, "source": source, "text": "가" * 40,
            "agreed_articles": arts, "forward_articles": arts, **_LABELER, **over}


def _write(tmp_path, rows):
    p = tmp_path / "clean.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return p


# ── 스냅숏 · 라벨러 · 분포 이동 ──────────────────────────────────────────────

def test_snapshot_counts_and_shares(tmp_path):
    p = _write(tmp_path, [
        _row("ftc_case:1:1", "ftc_case", ["제6조"]),
        _row("ftc_case:2:1", "ftc_case", ["제6조", "제9조"]),
        _row("standard_contract:3:1", "standard_contract", []),
        _row("standard_contract:3:2", "standard_contract", []),
    ])
    s = retrain.snapshot(p)
    assert s["records"] == 4
    assert s["empty_ratio"] == pytest.approx(0.5)
    assert s["article_share"] == {"제6조": 0.5, "제9조": 0.25}
    assert s["labelers"] == {"gpt-4o|fwd-v5|gpt-4o|ver-v5": 4}
    assert len(s["sha256"]) == 64


def test_mixed_labelers_are_blocked(tmp_path):
    """.env의 mini로 라벨링된 행이 섞이면 멈춰야 한다 — 실제로 있었던 사고 형태."""
    p = _write(tmp_path, [
        _row("ftc_case:1:1", "ftc_case", ["제6조"]),
        _row("ftc_case:2:1", "ftc_case", ["제6조"], forward_model="gpt-4o-mini"),
    ])
    problems = retrain.check_labelers(retrain.snapshot(p), None)
    assert len(problems) == 1 and "2개" in problems[0]


def test_labeler_change_against_history_is_blocked(tmp_path):
    p = _write(tmp_path, [_row("ftc_case:1:1", "ftc_case", ["제6조"], forward_prompt="fwd-v6")])
    prev = {"labelers": {"gpt-4o|fwd-v5|gpt-4o|ver-v5": 10}}
    problems = retrain.check_labelers(retrain.snapshot(p), prev)
    assert any("이전 기록" in x for x in problems)


def test_single_labeler_passes():
    snap = {"labelers": {"a|b|c|d": 3}}
    assert retrain.check_labelers(snap, {"labelers": {"a|b|c|d": 1}}) == []


def test_drift_warns_only_above_threshold():
    prev = {"empty_ratio": 0.60, "article_share": {"제6조": 0.20, "제9조": 0.10}}
    small = {"empty_ratio": 0.62, "article_share": {"제6조": 0.22, "제9조": 0.10}}
    big = {"empty_ratio": 0.45, "article_share": {"제6조": 0.20, "제9조": 0.18, "제13조": 0.06}}
    assert retrain.drift_warnings(small, prev) == []
    w = retrain.drift_warnings(big, prev)
    assert any("빈 라벨" in x for x in w)
    assert any("제9조" in x for x in w) and any("제13조" in x for x in w)
    assert retrain.drift_warnings(big, None) == []


# ── 판정 규칙 ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("recall_ci, fpr_ci, passed", [
    ((+0.4, -3.9, +4.7), (+0.3, -1.0, +1.5), True),     # 둘 다 미판정 → 통과
    ((-5.0, -9.0, -1.0), (0.0, -1.0, +1.0), False),     # 재현이 유의하게 하락
    ((0.0, -3.0, +3.0), (+3.0, +0.5, +5.5), False),     # 오경보가 유의하게 상승
    ((+6.0, +2.0, +10.0), (-2.0, -4.0, -0.5), True),    # 둘 다 좋아짐도 '통과'일 뿐
])
def test_harm_verdict(recall_ci, fpr_ci, passed):
    v = retrain.harm_verdict(recall_ci, fpr_ci)
    assert v["passed"] is passed
    assert len(v["reasons"]) == (0 if passed else 1)


def test_paired_ci_detects_consistent_shift_and_not_noise():
    rng = np.random.default_rng(0)
    worse = -(rng.random(300) < 0.15).astype(float)            # 15% 건에서 1→0
    d, lo, hi = retrain.paired_ci(worse)
    assert hi < 0
    noise = rng.choice([-1.0, 0.0, 1.0], size=300, p=[0.05, 0.9, 0.05])
    _, lo, hi = retrain.paired_ci(noise)
    assert lo < 0 < hi


def test_cluster_bootstrap_is_wider_than_row_bootstrap():
    """한 문서에 몰린 차이는 문서 단위로 재표집하면 구간이 넓어져야 한다(형제 조항)."""
    diff = np.zeros(120)
    diff[:10] = 1.0                                             # 전부 문서 0에서 발생
    groups = [f"doc{i // 10}" for i in range(120)]
    _, lo_r, hi_r = retrain.paired_ci(diff)
    _, lo_c, hi_c = retrain.paired_ci(diff, groups)
    assert (hi_c - lo_c) > (hi_r - lo_r)


def test_f1_rows_matches_reported_definition():
    pred = np.array([[1, 0], [0, 0], [1, 1]], dtype=bool)
    lab = np.array([[1, 0], [0, 1], [1, 0]], dtype=bool)
    np.testing.assert_allclose(retrain.f1_rows(pred, lab), [1.0, 0.0, 2 / 3])


def test_cumulative_warning_uses_first_record():
    history = [{"eval": {"base": {"dir": "models/article_v2", "gold_recall": 0.784, "holdout_fpr": 0.053}}}]
    assert retrain.cumulative_warnings(history, {"gold_recall": 0.77, "holdout_fpr": 0.06}) == []
    w = retrain.cumulative_warnings(history, {"gold_recall": 0.74, "holdout_fpr": 0.09})
    assert len(w) == 2
    assert retrain.cumulative_warnings([], {"gold_recall": 0.1, "holdout_fpr": 0.9}) == []


# ── 인자 복원 · 홀드아웃 고정 ────────────────────────────────────────────────

_BASE = {"train_config": {"epochs": 10, "batch_size": 16, "lr": 3e-5, "max_len": 512, "min_support": 5},
         "label_source": "agreed", "negative_ratio": None, "negative_holdout_docs": 12,
         "augment": False, "train_subsample": 0}


def test_build_train_argv_reproduces_base_config(tmp_path):
    argv = retrain.build_train_argv(_BASE, tmp_path / "l.jsonl", tmp_path / "out", tmp_path / "h.json", 1)
    kv = dict(zip(argv[::2], argv[1::2]))
    assert kv["--max-len"] == "512" and kv["--lr"] == "3e-05" and kv["--seed"] == "42"
    assert kv["--negative-holdout-docs"] == str(tmp_path / "h.json")
    assert "--negative-ratio" not in argv and "--fulltext-augment" not in argv


def test_build_train_argv_refuses_unfit_base(tmp_path):
    with pytest.raises(SystemExit):
        retrain.build_train_argv({**_BASE, "train_subsample": 500}, tmp_path, tmp_path, tmp_path, 1)
    with pytest.raises(SystemExit):
        retrain.build_train_argv({**_BASE, "negative_holdout_docs": 0}, tmp_path, tmp_path, tmp_path, 1)


def test_fixed_holdout_survives_new_documents():
    """문서가 늘면 시드 추첨은 다른 문서를 뽑는다 — 고정 목록은 그대로여야 한다."""
    recs = [{"group": f"standard_contract:{i}", "source": "standard_contract", "articles": []} for i in range(20)]
    _, before = split_negative_holdout(recs, 3, 42)
    grown = recs + [{"group": f"standard_contract:{i}", "source": "standard_contract", "articles": []}
                    for i in range(100, 130)]
    _, after_random = split_negative_holdout(grown, 3, 42)
    fixed = sorted({r["group"] for r in before})
    _, after_fixed = split_negative_holdout(grown, 3, 42, fixed)
    assert {r["group"] for r in after_fixed} == set(fixed)
    assert {r["group"] for r in after_random} != set(fixed)


def test_fixed_holdout_missing_doc_raises():
    recs = [{"group": "standard_contract:1", "source": "standard_contract", "articles": []}]
    with pytest.raises(SystemExit):
        split_negative_holdout(recs, 1, 42, ["standard_contract:999"])
