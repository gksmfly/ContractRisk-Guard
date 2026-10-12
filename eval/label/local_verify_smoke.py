# eval/label/local_verify_smoke.py
"""로컬 Qwen verify(`VERIFY_MODEL=local:qwen2.5-14b`) 동작 확인 — 운영 FB-Check 레코드 20건을 GPT verify와 대조 (2026-10-12, API 0)

    HF_HOME=/dev/shm/gksmfly_hf HF_HUB_OFFLINE=1 LOCAL_LLM_DEVICE=cuda:0 .venv/bin/python -m eval.label.local_verify_smoke
"""

import random
import time

from backend.fb_check.consistency_verification import run_verify
from backend.utils import PROJECT_ROOT, load_jsonl

N = 20


def main() -> None:
    rows = [r for r in load_jsonl(PROJECT_ROOT / "data/fb_check/clean.jsonl") + load_jsonl(PROJECT_ROOT / "data/fb_check/noise.jsonl")
            if r.get("verify_mode") in ("span", "clause") and r.get("verify_articles") is not None]
    random.Random(1).shuffle(rows)
    rows = rows[:N]
    agree = fails = 0
    t0 = time.time()
    for r in rows:
        inp = r["evidence_span"] if r["verify_mode"] == "span" else r["text"]
        out = run_verify(None, inp, mode=r["verify_mode"], model="local:qwen2.5-14b")
        if out is None:
            fails += 1
            continue
        g, q = set(r["verify_articles"]), set(out["articles"])
        agree += bool(g) == bool(q)
        print(f"[{r['verify_mode']}] GPT {sorted(g)} {r.get('verify_label')} | Qwen {out['articles']} {out['risk_level']}")
    print(f"== {N}건: 파싱 실패 {fails}, 위반 여부 GPT와 일치 {agree}/{N - fails}, 건당 {(time.time() - t0) / N:.1f}초 (모델 로딩 포함)")


if __name__ == "__main__":
    main()
