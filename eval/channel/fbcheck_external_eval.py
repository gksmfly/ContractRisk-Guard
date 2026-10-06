# backend/eval/fbcheck_external_eval.py
"""FB-Check가 더 나은 라벨을 고르는가 — **외부 준거(FTC 근거_법령)** 로 CLEAN vs NOISE를 채점한다.

## 왜 이 스크립트가 있나

README 헤드라인 "+8.8%p [+5.1, +12.7] (n=630/395)"는 커밋 37105f1에서 README에만 등장했고
`reported_numbers.md`에도 3,500줄 측정 로그에도 근거가 없었다. 데이터가 5배(CLEAN 1,786 →
3,352)가 된 뒤에도 그대로 남아 있었다. 2026-09-06 두 세션이 **서로 다른 코드로** 재계산해
소수점까지 같은 값을 얻었으나(+7.7%p [+5.2,+10.4], n=1,158/816, MW p=2.7e-09), 그 코드는
둘 다 레포 밖에 있었다 — +8.8%p가 출처를 잃은 경로 그대로다.

그래서 **한 번 실행에 네 변형을 전부 계산해 한 JSON에 쓴다.** 누가 하나만 다시 뽑아
헤드라인으로 올리는 일이 없게 하기 위해서다.

## 네 변형 — 서로 다른 질문이다

    ① agreed 라벨   CLEAN=agreed_articles / NOISE=forward_articles     ← **헤드라인**
                    "학습셋에 실제로 들어가는 라벨이 더 좋은가" (train_article 기본 label_source=agreed)
    ② forward 라벨  둘 다 forward_articles
                    "같은 라벨러의 라벨이 필터를 통과하면 더 믿을 만한가" — 필터의 선별력 자체
    ③ clean 지층    조항 1개·근거_법령 1개인 사건만 (gold clean과 같은 조건)
                    귀속 모호성이 없는 층. 규칙 8대로 MDE를 같이 낸다
    ④ 사유별·방향별  label_mismatch를 forward→verify 방향으로 쪼갠다
                    게이트가 어느 방향에서 값을 하는가

①이 헤드라인인 이유: README의 주장은 "FB-Check filters hallucinated labels"이고 그 산출물은
CLEAN 학습셋이며 그 라벨은 agreed다. 주장에 대응하는 수치를 고르면 ①이다. ①(+3.8)이
②(+7.7)보다 **불리**하다 — 주장에 맞춰 골랐더니 자기에게 불리한 값이 나온 사례로 남긴다.

## 채점 규칙

- 준거: 공정위 의결서 `근거_법령` → `normalize()`로 약관규제법 조만 추출. 사건 단위이므로
  다조항 사건은 각 조항에 사건 전체의 조가 붙는다(귀속 잡음). CLEAN·NOISE 양쪽에 같은
  잡음이 들어가므로 **상대 비교는 유효**하고, 잡음 없는 층은 ③이 따로 본다.
- non-empty predictions: 빈 예측은 양쪽에서 뺀다. FTC 준거는 비어 있지 않으므로 빈 예측은
  구성상 0점이고, CLEAN에 빈 배열이 21%(FTC 기준) 몰려 있어 포함하면 버려진 쪽이 유리해
  보인다. "빈 포함" 값도 참고로 같이 낸다(다른 질문: 학습셋 평균이 나빠지나).
- 검정: 평균차 부트스트랩 95% CI(5,000회, seed 42) 주, Mann-Whitney(단측) 보조.
- MDE: ③에서 관측 d로 80% 검정력에 필요한 NOISE n, 그리고 ②의 효과(+7.7%p)를 이 층에서
  보려면 필요한 n.

실행:
    .venv/bin/python -m eval.fbcheck_external_eval
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from backend.labeling.articles import ARTICLES, normalize
from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("fbcheck_external_eval.log")

CLEAN_PATH = PROJECT_ROOT / "data/fb_check/clean.jsonl"
NOISE_PATH = PROJECT_ROOT / "data/fb_check/noise.jsonl"
FTC_PATH   = PROJECT_ROOT / "data/raw/ftc_cases/ftc_cases_parsed.json"
OUT_PATH   = PROJECT_ROOT / "data/eval/fbcheck_external_report.json"

BOOT_N = 5000
SEED   = 42
Z_ALPHA, Z_POWER = 1.96, 0.84   # 양측 5%, 검정력 80%


def _f1(pred: set, ref: set) -> float:
    if not pred and not ref:
        return 1.0
    i = len(pred & ref)
    p = i / len(pred) if pred else 0.0
    r = i / len(ref) if ref else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def _labels(v) -> set[str]:
    """라벨은 이미 '제6조' 형태다. `normalize()`는 의결서 문자열용이라 여기엔 쓰지 않는다."""
    return {x for x in (v or []) if x in ARTICLES}


def _ref_by_doc() -> tuple[dict[str, set[str]], dict[str, tuple[int, int]]]:
    cases = json.loads(FTC_PATH.read_text(encoding="utf-8"))["사례"]
    ref, meta = {}, {}
    for c in cases:
        doc = str((c.get("셀_데이터") or {}).get("사건번호") or c.get("사건명", ""))
        arts = {a for g in (c.get("근거_법령") or []) if (a := normalize(str(g)))}
        if doc and arts:
            ref[f"ftc_case:{doc}"] = arts
            meta[f"ftc_case:{doc}"] = (len(c.get("조항_원문") or []), len(arts))
    return ref, meta


def _doc_of(chunk_id: str) -> str:
    parts = str(chunk_id).split(":")
    return ":".join(parts[:2]) if len(parts) >= 3 else str(chunk_id)


def _load(path: Path, ref: dict) -> list[dict]:
    out = []
    for r in load_jsonl(path):
        if r.get("source") != "ftc_case":
            continue
        doc = _doc_of(r.get("chunk_id", ""))
        if doc not in ref:
            continue
        out.append({"doc": doc,
                    "forward": _labels(r.get("forward_articles")),
                    "agreed": _labels(r.get("agreed_articles")),
                    "reason": str(r.get("noise_reason") or "")})
    return out


def _compare(a: list[float], b: list[float], rng: np.random.Generator) -> dict:
    """a(CLEAN) − b(NOISE). 양수면 CLEAN이 낫다."""
    a, b = np.array(a), np.array(b)
    if len(a) < 2 or len(b) < 2:
        return {"n_clean": len(a), "n_noise": len(b), "verdict": "표본 부족"}
    boot = np.sort([a[rng.integers(0, len(a), len(a))].mean()
                    - b[rng.integers(0, len(b), len(b))].mean() for _ in range(BOOT_N)])
    lo, hi = boot[int(BOOT_N * 0.025)] * 100, boot[int(BOOT_N * 0.975)] * 100
    _, p = mannwhitneyu(a, b, alternative="greater")
    d = (a.mean() - b.mean()) * 100
    verdict = "CLEAN이 낫다(유의)" if lo > 0 else ("미판정" if hi > 0 else "NOISE가 낫다(유의)")
    return {"n_clean": int(len(a)), "n_noise": int(len(b)),
            "clean_f1": float(a.mean()), "noise_f1": float(b.mean()),
            "diff_pp": float(d), "ci95_pp": [float(lo), float(hi)],
            "mw_p": float(p), "verdict": verdict}


def _mde(a: list[float], b: list[float], target_pp: float | None = None) -> dict:
    """관측 효과크기 d와, 80% 검정력에 필요한 소집단(NOISE) n. target_pp를 주면 그 효과 기준."""
    a, b = np.array(a), np.array(b)
    sd = float(np.sqrt((a.var() * len(a) + b.var() * len(b)) / (len(a) + len(b))))
    ratio = len(a) / len(b)
    k = (1 + 1 / ratio) * (Z_ALPHA + Z_POWER) ** 2
    d_obs = float((a.mean() - b.mean()) / sd) if sd else 0.0
    out = {"d_observed": d_obs, "pooled_sd": sd,
           "n_noise_needed_at_observed_d": float(k / d_obs ** 2) if d_obs > 0 else None}
    if target_pp is not None and sd:
        out["n_noise_needed_at_target"] = {"target_pp": target_pp,
                                           "n": float(k / (target_pp / 100 / sd) ** 2)}
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    ref, meta = _ref_by_doc()
    clean, noise = _load(CLEAN_PATH, ref), _load(NOISE_PATH, ref)
    logger.info(f"ftc_case & 준거 있음: CLEAN {len(clean)} · NOISE {len(noise)} "
                f"(원본 {sum(1 for _ in open(CLEAN_PATH, encoding='utf-8'))} / "
                f"{sum(1 for _ in open(NOISE_PATH, encoding='utf-8'))})")

    def scores(rows, key, cond=lambda d: True, nonempty=True):
        return [_f1(r[key], ref[r["doc"]]) for r in rows
                if cond(r["doc"]) and (r[key] or not nonempty)]

    report = {"conditions": {
        "reference": "FTC 의결서 근거_법령(사건 단위) → 약관규제법 조. GPT 무관",
        "population": "source=ftc_case 이고 사건에 근거_법령이 있는 레코드",
        "n_clean": len(clean), "n_noise": len(noise),
        "test": f"평균차 부트스트랩 95% CI({BOOT_N}회, seed {SEED}) 주 · Mann-Whitney 단측 보조",
    }}

    # ① 헤드라인: agreed(CLEAN) vs forward(NOISE)
    report["1_agreed_headline"] = {
        "question": "학습셋에 실제로 들어가는 라벨(agreed)이 버려진 라벨(forward)보다 좋은가",
        "nonempty": _compare(scores(clean, "agreed"), scores(noise, "forward"), rng),
        "all_incl_empty": _compare(scores(clean, "agreed", nonempty=False),
                                   scores(noise, "forward", nonempty=False), rng),
    }
    # ② forward vs forward
    report["2_forward_both"] = {
        "question": "같은 라벨러(forward)의 라벨이 필터를 통과하면 더 믿을 만한가 — 선별력 자체",
        "nonempty": _compare(scores(clean, "forward"), scores(noise, "forward"), rng),
        "all_incl_empty": _compare(scores(clean, "forward", nonempty=False),
                                   scores(noise, "forward", nonempty=False), rng),
    }
    # ③ clean 지층
    is_clean = lambda d: meta[d] == (1, 1)
    c1, n1 = scores(clean, "agreed", is_clean), scores(noise, "forward", is_clean)
    c2, n2 = scores(clean, "forward", is_clean), scores(noise, "forward", is_clean)
    report["3_clean_stratum"] = {
        "condition": "사건의 조항_원문 1개 AND 근거_법령 1개 — 귀속 모호성 없음 (gold clean과 동일)",
        "agreed": _compare(c1, n1, rng), "forward": _compare(c2, n2, rng),
        "mde_forward": _mde(c2, n2, target_pp=report["2_forward_both"]["nonempty"]["diff_pp"]),
        "mde_agreed": _mde(c1, n1, target_pp=report["1_agreed_headline"]["nonempty"]["diff_pp"]),
    }
    # ④ 사유별·방향별 (forward 라벨, CLEAN forward 대비)
    base = scores(clean, "forward")
    by_reason, by_dir = defaultdict(list), defaultdict(list)
    for r in noise:
        if not r["forward"]:
            continue
        s = _f1(r["forward"], ref[r["doc"]])
        head = r["reason"].split(":")[0].strip()
        by_reason[head].append(s)
        if head == "label_mismatch":
            by_dir[r["reason"].split(":", 1)[1].strip().replace(" ", "")].append(s)
    report["4_by_reason"] = {
        "basis": "forward 라벨, non-empty, CLEAN(forward) 대비. 양수면 버린 쪽이 나쁘다",
        "reason": {k: _compare(base, v, rng) for k, v in sorted(by_reason.items())},
        "label_mismatch_direction": {k: _compare(base, v, rng)
                                     for k, v in sorted(by_dir.items(), key=lambda kv: -len(kv[1]))},
    }
    report["provenance"] = {
        "supersedes": "+8.8%p [+5.1,+12.7] (n=630/395) — README에만 있었고 근거 없음. 폐기",
        "independent_reproduction": "2026-09-06 두 세션이 다른 코드로 ② +7.7%p [+5.2,+10.4] "
                                    "n=1,158/816 MW p=2.7e-09 를 소수점까지 동일하게 얻음",
    }

    def line(tag, c):
        if "diff_pp" not in c:
            return f"    {tag:<34} {c}"
        return (f"    {tag:<34} CLEAN {c['n_clean']:>5} {c['clean_f1']*100:5.1f}%  "
                f"NOISE {c['n_noise']:>4} {c['noise_f1']*100:5.1f}%  "
                f"{c['diff_pp']:+5.1f}%p [{c['ci95_pp'][0]:+.1f},{c['ci95_pp'][1]:+.1f}]  "
                f"MW p={c['mw_p']:.1e}  {c['verdict']}")
    logger.info("  ① agreed(헤드라인)");  logger.info(line("non-empty", report["1_agreed_headline"]["nonempty"]))
    logger.info(line("빈 포함(참고)", report["1_agreed_headline"]["all_incl_empty"]))
    logger.info("  ② forward 둘 다");     logger.info(line("non-empty", report["2_forward_both"]["nonempty"]))
    logger.info(line("빈 포함(참고)", report["2_forward_both"]["all_incl_empty"]))
    logger.info("  ③ clean 지층");        logger.info(line("agreed", report["3_clean_stratum"]["agreed"]))
    logger.info(line("forward", report["3_clean_stratum"]["forward"]))
    m = report["3_clean_stratum"]["mde_forward"]
    logger.info(f"    MDE(forward)  d={m['d_observed']:.2f}  관측 d로 필요 NOISE n≈{m['n_noise_needed_at_observed_d'] or float('nan'):.0f}"
                f"  ②효과({m['n_noise_needed_at_target']['target_pp']:+.1f}%p) 기준 n≈{m['n_noise_needed_at_target']['n']:.0f}")
    logger.info("  ④ 사유별")
    for k, v in report["4_by_reason"]["reason"].items():
        logger.info(line(k, v))
    logger.info("  ④ label_mismatch 방향별 (forward→verify)")
    for k, v in report["4_by_reason"]["label_mismatch_direction"].items():
        logger.info(line(k, v))
    save_json(report, OUT_PATH)
    logger.info(f"  저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
