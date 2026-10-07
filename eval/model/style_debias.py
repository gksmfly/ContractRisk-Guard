# eval/model/style_debias.py
"""말투 편향 줄이기 — 말투를 뒤집은 사본으로 학습 (사전 등록: eval/paper_plan.md §7-2)

    --generate   Qwen2.5-14B로 학습 후보 조항의 말투를 뒤집는다 (API 0, 8건씩 묶어 생성)
                 공정한 표준계약서 → 회사 이용약관 말투 / 위반 공정위 조항 → 표준계약서 공문서 말투
    --build      자동 검사 통과분으로 학습 라벨 파일 두 개를 만든다
                 data/eval/style_aug/labels_SN.jsonl  (clean.jsonl + 표준계약서 → 회사 말투 사본)
                 data/eval/style_aug/labels_SS.jsonl  (위 + 공정위 위반 → 공문서 말투 사본)
    --analyze    KSN·KSS·RSN·RSS vs C·RB 비교 (B4 말투 오경보 주 지표, 학습 변동 포함 CI)

사본의 chunk_id는 `원본#sty`라 `_document_group`이 원본과 같은 문서로 묶는다 → dev 분할에서 원본과 갈라지지 않는다.
"""

import argparse
import json

import numpy as np

from backend.utils import PROJECT_ROOT, load_jsonl, load_logger, save_json, save_jsonl
from eval.channel.paraphrase_consistency import quality

logger = load_logger("style_debias.log")

CLEAN   = PROJECT_ROOT / "data/fb_check/clean.jsonl"
HOLDOUT = PROJECT_ROOT / "backend/training/negative_holdout_docs.json"
AUG_DIR = PROJECT_ROOT / "data/eval/style_aug"
RAW     = AUG_DIR / "raw.jsonl"
BATCH   = 8

_RULES = """규칙:
1. 당사자 관계, 권리와 의무, 조건, 기한, 금액·비율·횟수 같은 숫자, 예외(단서)를 하나도 바꾸거나 빼거나 더하지 마세요.
2. 새로운 의무·면책·제한·불이익을 넣지 마세요. 내용이 더 유리해지거나 불리해지면 안 됩니다.
3. 숫자는 원문에 적힌 그대로(아라비아 숫자는 아라비아 숫자로) 쓰세요.
4. 설명이나 평가를 덧붙이지 마세요.

반드시 다음 JSON 형식으로만 답하세요: {"rewrite": "다시 쓴 조항 전문"}"""

TO_COMMERCIAL = ("당신은 국내 인터넷 서비스·플랫폼 기업의 이용약관을 쓰는 담당자입니다. 주어진 계약 조항을 "
                 "**그 회사 이용약관의 말투**로 다시 쓰세요. '갑/을' 대신 '회사/회원(이용자)'을 쓰고, "
                 "'회사는 ~할 수 있습니다', '회원은 ~하여야 합니다' 같은 경어체로 씁니다.\n\n" + _RULES)
TO_FORMAL = ("당신은 정부(공정거래위원회 등)가 배포하는 표준계약서를 작성하는 담당자입니다. 주어진 약관 조항을 "
             "**표준계약서의 공문서 말투**로 다시 쓰세요. '회사/회원' 대신 '갑/을'을 쓰고, "
             "'갑은 을에게 ~한다', '을은 ~하여야 한다' 같은 평서체로 씁니다.\n\n" + _RULES)


def _doc(cid: str) -> str:
    return ":".join(str(cid).split(":")[:2])


def sources() -> list[dict]:
    held = set(json.loads(HOLDOUT.read_text(encoding="utf-8")))
    out = []
    for r in load_jsonl(CLEAN):
        if _doc(r["chunk_id"]) in held or len(r.get("text", "")) > 1500:
            continue
        if r["source"] == "standard_contract" and not r.get("agreed_articles"):
            out.append({"chunk_id": r["chunk_id"], "dir": "to_commercial", "text": r["text"]})
        elif r["source"] == "ftc_case" and r.get("agreed_articles"):
            out.append({"chunk_id": r["chunk_id"], "dir": "to_formal", "text": r["text"]})
    return out


# ── 생성 ─────────────────────────────────────────────────────────────────────

def generate(gpu: int) -> None:
    import torch

    from eval.label.e9_local_labeling import _extract_json, load_model

    AUG_DIR.mkdir(parents=True, exist_ok=True)
    done = {(r["chunk_id"], r["dir"]) for r in load_jsonl(RAW)} if RAW.exists() else set()
    todo = [s for s in sources() if (s["chunk_id"], s["dir"]) not in done]
    todo.sort(key=lambda s: len(s["text"]))                      # 길이가 비슷한 것끼리 묶어 패딩 낭비를 줄인다
    logger.info(f"=== 말투 뒤집기 남음 {len(todo)} (완료 {len(done)}) ===")
    if not todo:
        return
    model, tok, _ = load_model("qwen2.5-14b", gpu)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = next(model.parameters()).device
    with open(RAW, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            prompts = [tok.apply_chat_template(
                [{"role": "system", "content": TO_COMMERCIAL if s["dir"] == "to_commercial" else TO_FORMAL},
                 {"role": "user", "content": f"계약 조항:\n{s['text']}"}],
                tokenize=False, add_generation_prompt=True) for s in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
            n_in = max(len(tok(s["text"])["input_ids"]) for s in chunk)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=min(1500, int(n_in * 1.8) + 64),
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            for s, seq in zip(chunk, out):
                text = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                j = _extract_json(text)
                rw = j.get("rewrite") if isinstance(j, dict) and isinstance(j.get("rewrite"), str) else None
                row = {"chunk_id": s["chunk_id"], "dir": s["dir"], "rewrite": rw}
                if rw:
                    row["quality"] = quality(s["text"], rw)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if (i // BATCH) % 25 == 0:
                logger.info(f"  [{i + len(chunk)}/{len(todo)}]")


# ── 라벨 파일 ────────────────────────────────────────────────────────────────

def build() -> None:
    clean = load_jsonl(CLEAN)
    by_id = {r["chunk_id"]: r for r in clean}
    raw = [r for r in load_jsonl(RAW) if r.get("rewrite") and r["quality"]["pass"]]
    aug = {"to_commercial": [], "to_formal": []}
    for r in raw:
        o = dict(by_id[r["chunk_id"]])
        o.update({"chunk_id": r["chunk_id"] + "#sty", "fb_id": str(o.get("fb_id", "")) + "#sty",
                  "text": r["rewrite"], "style_aug": r["dir"]})
        aug[r["dir"]].append(o)
    sn = clean + aug["to_commercial"]
    ss = sn + aug["to_formal"]
    save_jsonl(sn, AUG_DIR / "labels_SN.jsonl")
    save_jsonl(ss, AUG_DIR / "labels_SS.jsonl")
    n_src = {d: sum(s["dir"] == d for s in sources()) for d in aug}
    stats = {"sources": n_src, "generated": {d: sum(r["dir"] == d for r in load_jsonl(RAW) if r.get("rewrite")) for d in aug},
             "passed_quality": {d: len(v) for d, v in aug.items()}, "SN": len(sn), "SS": len(ss)}
    save_json(stats, AUG_DIR / "build_stats.json")
    logger.info(json.dumps(stats, ensure_ascii=False))


# ── 분석 ─────────────────────────────────────────────────────────────────────

ARMS = {"C": "KoELECTRA 기존", "KSN": "KoELECTRA +회사말투 음성", "KSS": "KoELECTRA +양방향",
        "RB": "RoBERTa 기존", "RSN": "RoBERTa +회사말투 음성", "RSS": "RoBERTa +양방향"}
PAIRS = [("KSN", "C"), ("KSS", "C"), ("KSS", "KSN"), ("RSN", "RB"), ("RSS", "RB"), ("RSS", "RSN")]


def analyze(gpu: int) -> None:
    from backend.training.retrain import eval_sets, f1_rows, predict
    from backend.training.train_article import load_article_records, split_negative_holdout
    from eval.channel.paraphrase_consistency import _items

    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    gold, neg = eval_sets(holdout)
    recs = load_article_records(CLEAN, label_source="agreed")
    _, h = split_negative_holdout(recs, 12, 42)
    b4o = [r for r in h if r.get("source") == "standard_contract" and not r.get("articles")]
    b4x = json.loads((PROJECT_ROOT / "data/eval/channel/style_transfer_fpr.cache.json").read_text(encoding="utf-8"))
    items = {r["key"]: r for r in _items()}
    para = [{**items[c["key"]], "para": c["paraphrase"]} for c in load_jsonl(PROJECT_ROOT / "data/eval/channel/paraphrase_raw.jsonl")
            if c.get("paraphrase") and c["quality"]["pass"] and c["key"].startswith("gold")]
    real = [c["text"] for f in sorted((PROJECT_ROOT / "data/eval/real_tos").glob("*.json"))
            for c in json.loads(f.read_text(encoding="utf-8"))["clauses"]]

    sets = {"gold": [g["text"] for g in gold], "hold": [r["text"] for r in neg], "b4o": [r["text"] for r in b4o],
            "b4x": b4x, "para_o": [p["text"] for p in para], "para_p": [p["para"] for p in para], "real": real}
    groups = {"gold": [g["doc_id"] for g in gold], "hold": [_doc(r["chunk_id"]) for r in neg],
              "b4x": [r["group"] for r in b4o], "para": [p["group"] for p in para]}
    labels = None
    per = {}                                                     # arm -> seed -> {set: flags}
    for arm in ARMS:
        for s in (42, 7, 123):
            d = PROJECT_ROOT / f"models/_e8_{arm}_s{s}"
            if not (d / "metrics.json").exists():
                continue
            fl = {}
            for k, texts in sets.items():
                P, thr, names = predict(d, texts, gpu)
                fl[k] = P >= thr
            if labels is None:
                labels = np.array([[a in g["articles"] for a in names] for g in gold])
            fl["f1"] = f1_rows(fl["gold"], labels)
            per.setdefault(arm, {})[s] = fl

    def metric(fl: dict) -> dict:
        return {"gold_recall": fl["gold"].any(1), "hold_fpr": fl["hold"].any(1), "b4x_fpr": fl["b4x"].any(1),
                "b4o_fpr": fl["b4o"].any(1), "para_recall": fl["para_p"].any(1), "real_flag": fl["real"].any(1),
                "gold_f1": fl["f1"]}
    grp_of = {"gold_recall": "gold", "gold_f1": "gold", "hold_fpr": "hold", "b4x_fpr": "b4x", "b4o_fpr": "b4x",
              "para_recall": "para", "real_flag": None}
    summary = {}
    for arm, seeds in per.items():
        ms = [metric(fl) for fl in seeds.values()]
        summary[arm] = {k: {"mean": round(float(np.mean([m[k].mean() for m in ms])) * 100, 2),
                            "seeds": [round(float(m[k].mean()) * 100, 1) for m in ms]} for k in ms[0]}

    rng = np.random.default_rng(42)
    contrasts = {}
    for x, y in PAIRS:
        if x not in per or y not in per:
            continue
        mx = [metric(fl) for fl in per[x].values()]
        my = [metric(fl) for fl in per[y].values()]
        row = {}
        for k, g in grp_of.items():
            ax = np.mean([m[k].astype(float) for m in mx], axis=0)
            ay = np.mean([m[k].astype(float) for m in my], axis=0)
            diff = ax - ay
            if g is None:
                boot_var = 0.0
            else:
                idx: dict[str, list[int]] = {}
                for i, gg in enumerate(groups[g]):
                    idx.setdefault(gg, []).append(i)
                cl = [np.array(v) for v in idx.values()]
                b = [diff[np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])].mean() for _ in range(2000)]
                boot_var = float(np.var(b))
            sx = np.var([m[k].mean() for m in mx], ddof=1) / len(mx) if len(mx) > 1 else 0.0
            sy = np.var([m[k].mean() for m in my], ddof=1) / len(my) if len(my) > 1 else 0.0
            se = float(np.sqrt(boot_var + sx + sy))
            pt = float(diff.mean())
            lo, hi = pt - 1.96 * se, pt + 1.96 * se
            row[k] = {"diff_pp": round(pt * 100, 2), "ci95_pp": [round(lo * 100, 2), round(hi * 100, 2)],
                      "sig": "감소" if hi < 0 else ("증가" if lo > 0 else "미판정")}
        contrasts[f"{x}-{y}"] = row
    # 판정 (사전 등록): 1 주 지표 b4x_fpr 감소, 2 해 없음 = gold 재현 유의 감소 없음 & 홀드아웃 오경보 유의 증가 없음
    verdict = {}
    for c, row in contrasts.items():
        if c.endswith("-C") or c.endswith("-RB"):
            verdict[c] = {"style_bias_reduced": row["b4x_fpr"]["sig"] == "감소",
                          "harm": row["gold_recall"]["sig"] == "감소" or row["hold_fpr"]["sig"] == "증가"}
    rep = {"arms": ARMS, "summary": summary, "contrasts": contrasts, "verdict": verdict,
           "n": {k: len(v) for k, v in sets.items()},
           "note": "b4x = GPT-4o-mini가 회사 약관 말투로 바꾼 공정한 표준계약서(학습 증강은 Qwen이라 다른 모델). para는 Qwen 변형이라 증강과 같은 모델 — 과대평가 가능"}
    save_json(rep, PROJECT_ROOT / "data/eval/model/style_debias.json")
    for arm, s in summary.items():
        logger.info(f"  {arm:<4} {ARMS[arm]:<18} 말투오경보 {s['b4x_fpr']['mean']:5.1f}% (원문 {s['b4o_fpr']['mean']:4.1f}) "
                    f"홀드아웃 {s['hold_fpr']['mean']:4.1f}% 재현 {s['gold_recall']['mean']:4.1f}% "
                    f"변형재현 {s['para_recall']['mean']:4.1f}% 조F1 {s['gold_f1']['mean']:4.1f}% 실제약관 {s['real_flag']['mean']:4.1f}%")
    for c, row in contrasts.items():
        logger.info(f"  {c:<8} " + " | ".join(f"{k} {v['diff_pp']:+.1f} {v['ci95_pp']} {v['sig']}" for k, v in row.items()))
    logger.info(f"  판정: {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    if args.generate:
        generate(args.gpu)
    if args.build:
        build()
    if args.analyze:
        analyze(args.gpu)


if __name__ == "__main__":
    main()
