# eval/label/card_detect.py
"""요건 카드 기반 누락 위반 탐지 — 파일럿 (사전 등록: eval/paper_plan.md §7-25, 2026-10-13, 학습·API 0)

첫 판정이 빈 라벨이고 인코더(CNA 시드 7)도 못 짚은 조항에서, 요건 카드(V) 또는 조 번호·제목만(A, 통제)을 로컬 Qwen에게 읽혀
위반 후보를 표시하게 한다. 표시 수(충족 호 수 / 해당 조 수)를 점수로 쓴다.

    --build  : 개발용(학습 절반)에서 양성 100 + 음성 100 표본을 한 번 뽑아 고정한다
    --run V|A: 표본에 대해 Qwen을 호출해 raw 파일에 이어 쓴다 (이어서 실행 가능)
    --analyze: 파일럿 판정 지표(오경보·재현·형식 오류·AUROC·운영점별 곡선·정밀도 두 가지 구성)
평가용(평가 절반)은 `--half eval`로만 열며, 파일럿 통과 전에는 쓰지 않는다.
"""

import argparse
import json
import os
import re

import numpy as np

from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json

logger = load_logger("card_detect.log")
DIR = PROJECT_ROOT / "data/eval/label/card_detect"
CARDS = json.loads((PROJECT_ROOT / "data/eval/requirement_cards.json").read_text(encoding="utf-8"))
SEED = 42
PROMPT_VERSION = "card-v0"
MODEL = "local:qwen2.5-14b"

RULES = """## 판단 규칙
- 입력은 약관 조항 하나입니다. 이 조항이 아래 목록의 항목에 **실질적으로 해당하는지**만 판단하세요.
- 용어나 주제가 비슷하다는 이유만으로 해당한다고 하지 마세요. 조항의 문언에서 직접 확인되는 것만 고르세요.
- "부당하게", "상당한 이유 없이" 같은 한정어는 항목의 일부입니다. 조항에 고객 보호 장치(단서, 사전 고지, 합리적 기한 등)가 보이면 반영하세요.
- 해당하는 항목이 없으면 빈 목록을 내세요. 억지로 고르지 마세요.
- reason은 조항에서 직접 인용한 구절을 포함해 한 문장으로 쓰세요."""


def ids_v() -> list[tuple[str, str]]:
    out = []
    for c in CARDS:
        n = c["article"].replace("제", "").replace("조", "")
        for k, it in enumerate(c["items"], 1):
            out.append((f"{n}-{k}", f"{c['article']} {it['ho']} — {it['check']}"))
    return out


def system_prompt(arm: str) -> str:
    if arm == "V":
        listing = "\n".join(f"[{i}] {t}" for i, t in ids_v())
        return ("당신은 한국 약관 심사 전문가입니다. 「약관의 규제에 관한 법률」 제6~14조의 요건 목록이 있습니다.\n\n" + RULES +
                f"\n\n## 요건 목록\n{listing}\n\n다음 JSON 하나만 출력하세요: "
                '{"reason": "<한 문장>", "satisfied": ["<해당하는 항목 번호>", ...]}')
    listing = "\n".join(f"[{c['article'].replace('제', '').replace('조', '')}] {c['article']} ({c['title']})" for c in CARDS)
    return ("당신은 한국 약관 심사 전문가입니다. 「약관의 규제에 관한 법률」 제6~14조의 목록이 있습니다.\n\n" + RULES +
            f"\n\n## 조 목록\n{listing}\n\n다음 JSON 하나만 출력하세요: "
            '{"reason": "<한 문장>", "satisfied": ["<해당하는 조 번호>", ...]}')


def population(half: str) -> list[dict]:
    from eval.label.fbcheck_aihub import population as pop
    from eval.model.aihub_eval import CACHE, testset
    items = testset()
    idx = {(it["group"], it["text"]): i for i, it in enumerate(items)}
    train_files = set(json.loads((PROJECT_ROOT / "data/eval/model/aihub_split.json").read_text(encoding="utf-8"))["train_files"])
    raw = {r["key"]: r for r in load_jsonl(PROJECT_ROOT / "data/eval/label/fbcheck_aihub_raw.jsonl")}
    z = np.load(CACHE / "CNA_s7.npz")
    flag = (z["P"] >= z["thr"]).any(1)
    out = []
    for p in pop():
        in_train = p["group"] in train_files
        if (half == "dev") != in_train:
            continue
        r = raw[p["key"]]
        if r["forward"] or flag[idx[(p["group"], p["text"])]]:
            continue                       # 첫 판정이 라벨을 붙였거나 인코더가 이미 표시한 조항은 대상이 아니다
        out.append({"key": p["key"], "group": p["group"], "text": p["text"], "pos": p["label"] == 1, "articles": p["articles"]})
    return out


def build() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    pop = population("dev")
    pos = [x for x in pop if x["pos"]]
    neg = [x for x in pop if not x["pos"]]
    rng = np.random.default_rng(SEED)
    sel = [pos[i] for i in rng.choice(len(pos), 100, replace=False)] + [neg[i] for i in rng.choice(len(neg), 100, replace=False)]
    (DIR / "pilot_sample.json").write_text(json.dumps(sel, ensure_ascii=False), encoding="utf-8")
    logger.info(f"개발용 모집단 양성 {len(pos)} · 음성 {len(neg)} → 파일럿 표본 {len(sel)}건 고정")


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return d if isinstance(d.get("satisfied"), list) else None


def run(arm: str, half: str) -> None:
    from backend.fb_check import local_llm
    os.environ.setdefault("LOCAL_LLM_ALLOW_DOWNLOAD", "0")
    sample = json.loads((DIR / "pilot_sample.json").read_text(encoding="utf-8")) if half == "dev" else population("eval")
    path = DIR / f"raw_{half}_{arm}.jsonl"
    done = {r["key"] for r in load_jsonl(path)} if path.exists() else set()
    sysmsg = system_prompt(arm)
    valid = {i for i, _ in ids_v()} if arm == "V" else {c["article"].replace("제", "").replace("조", "") for c in CARDS}
    with open(path, "a", encoding="utf-8") as f:
        for n, x in enumerate(sample):
            if x["key"] in done:
                continue
            msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": f"약관 조항:\n{x['text'][:1500]}"}]
            out = local_llm.chat_json(MODEL, msgs, max_new_tokens=160)
            ok = out is not None and isinstance(out.get("satisfied"), list)
            sat = sorted({str(s) for s in out["satisfied"] if str(s) in valid}) if ok else []
            f.write(json.dumps({"key": x["key"], "pos": x["pos"], "parse_ok": ok, "satisfied": sat,
                                "score": len(sat), "reason": (out or {}).get("reason", "")}, ensure_ascii=False) + "\n")
            f.flush()
            if (n + 1) % 25 == 0:
                logger.info(f"  [{arm}/{half}] {n + 1}/{len(sample)}")


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def analyze(half: str = "dev", eval_pos_neg: tuple[int, int] | None = None) -> None:
    rep = {"rule": "eval/paper_plan.md §7-25", "half": half, "prompt": PROMPT_VERSION, "arms": {}}
    for arm in ("V", "A"):
        path = DIR / f"raw_{half}_{arm}.jsonl"
        if not path.exists():
            continue
        rows = load_jsonl(path)
        pos = np.array([r["score"] for r in rows if r["pos"]], dtype=float)
        neg = np.array([r["score"] for r in rows if not r["pos"]], dtype=float)
        fmt_err = 1 - float(np.mean([r["parse_ok"] for r in rows]))
        curve = []
        for k in range(1, int(max(pos.max(), neg.max())) + 2):
            curve.append({"k": k, "recall_pct": round(float((pos >= k).mean()) * 100, 1), "false_alarm_pct": round(float((neg >= k).mean()) * 100, 1)})
        ok_pts = [c for c in curve if c["false_alarm_pct"] <= 5.0]
        best = max(ok_pts, key=lambda c: c["recall_pct"]) if ok_pts else None
        a = {"n_pos": len(pos), "n_neg": len(neg), "format_error_pct": round(fmt_err * 100, 1), "auroc": round(auroc(pos, neg), 3),
             "curve": curve, "operating_point_fa_le_5": best}
        if best:
            R, F = best["recall_pct"] / 100, best["false_alarm_pct"] / 100
            a["precision_at_pilot_mix(50:50)_pct"] = round(R / (R + F) * 100, 1) if R + F > 0 else None
            if eval_pos_neg:
                pi = eval_pos_neg[0] / sum(eval_pos_neg)
                a["precision_at_eval_mix_pct"] = round(R * pi / (R * pi + F * (1 - pi)) * 100, 1) if R * pi + F * (1 - pi) > 0 else None
        rep["arms"][arm] = a
    v = rep["arms"].get("V")
    if v and half == "dev":
        b = v["operating_point_fa_le_5"]
        rep["pilot_gates"] = {
            "1_오경보<=5%": bool(b and b["false_alarm_pct"] <= 5.0),
            "2_재현>=20%": bool(b and b["recall_pct"] >= 20.0),
            "3_형식오류<=10%": v["format_error_pct"] <= 10.0,
            "보조_V의_AUROC가_A보다_높음": bool(rep["arms"].get("A") and v["auroc"] > rep["arms"]["A"]["auroc"])}
        rep["pilot_decision"] = "평가용으로 진행" if all(list(rep["pilot_gates"].values())[:3]) and rep["pilot_gates"]["보조_V의_AUROC가_A보다_높음"] else "중단"
    save_json(rep, DIR / f"analysis_{half}.json")
    logger.info(json.dumps(rep, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", choices=["V", "A"])
    ap.add_argument("--half", default="dev", choices=["dev", "eval"])
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    if a.run:
        run(a.run, a.half)
    if a.analyze:
        pop = population("eval")
        mix = (sum(x["pos"] for x in pop), sum(not x["pos"] for x in pop))
        analyze(a.half, mix)


if __name__ == "__main__":
    main()
