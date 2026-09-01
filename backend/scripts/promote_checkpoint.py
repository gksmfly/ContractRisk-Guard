# backend/scripts/promote_checkpoint.py
"""실험 체크포인트를 **운영 이름으로 승격**한다 — 검증·지문·기록을 한 번에.

## 왜 스크립트인가

`models/article_v1`은 `cp -r models/_article_rNone`으로 만들어졌다. 검증도 지문도 없이.
그래서 **프로덕션이 가중치 검사가 꺼진 채로 돌았다** — 가드가 꺼져 있어도 되는 곳이
있다면 프로덕션이 마지막인데 정확히 반대로 돼 있었다.

손으로 하면 다음 승격 때 똑같이 빠진다. 절차를 코드로 굳힌다:

    1. 인코더 왕복 확인   safetensors 키와 재로드된 state_dict의 차이가 0인가
    2. 지문 계산·기록     **`model._compute_fingerprint()`** — `save()`가 쓰는 바로 그 함수
    3. 승격 기록          metrics.json에 출처·사유·측정치
    4. load() 왕복 확인   기록한 지문으로 실제 통과하는가

## 지문을 손으로 박지 않는다

`heads["fingerprint_f64"] = 5325.83...` 처럼 상수를 넣으면 **"저장 형식을 아는 곳이
한 군데"라는 불변식이 깨진다.** `save()`가 만들지 않은 지문 경로가 생기고, 그러면
가드 자체가 약해진다. 같은 함수로 계산해서 쓴다.

## 원자적 쓰기 + 백업

서빙 아티팩트다. 중간에 죽으면 반쪽 `heads.pt`가 남고 서빙이 안 뜬다.
`clean.jsonl`에 적용한 원칙 그대로 — 백업 후 tmp에 쓰고 `os.replace`.

실행:
    .venv/bin/python -m backend.scripts.promote_checkpoint models/_article_rNone models/article_v1
    .venv/bin/python -m backend.scripts.promote_checkpoint --stamp-only models/article_v1
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch

from backend.model.electra import ArticleMultiLabelElectra
from backend.utils import load_logger

logger = load_logger("promote_checkpoint.log")


def verify_encoder(path: Path) -> tuple[int, float]:
    """저장된 safetensors와 **재로드된 인코더**를 키 단위로 대조한다.

    `from_pretrained`는 가중치가 없거나 config가 어긋나도 경고만 찍고 랜덤 초기화로
    진행한다. 실제로 그 사고가 났고(학습 로그 macro 0.73 vs 채점 F1 5.8%) 아무도 몰랐다.
    """
    from safetensors.torch import load_file
    from transformers import AutoModel

    saved = load_file(str(path / "model.safetensors"))
    live = dict(AutoModel.from_pretrained(str(path)).state_dict())
    missing = [k for k in live if k not in saved]
    extra = [k for k in saved if k not in live]
    if missing or extra:
        raise SystemExit(f"인코더 키 불일치 — 없는 키 {missing[:5]} / 남는 키 {extra[:5]}")
    diff = sum(float((live[k].float() - saved[k].float()).norm()) for k in live)
    return len(live), diff


def stamp(path: Path, note: dict | None = None) -> float:
    """지문을 계산해 `heads.pt`에 기록한다. 원자적 쓰기 + 백업."""
    model = ArticleMultiLabelElectra.load(path)          # 지문 없어도 로드는 된다(경고만)
    fp = model._compute_fingerprint()                            # save()가 쓰는 그 함수

    heads = torch.load(path / "heads.pt", map_location="cpu", weights_only=False)
    heads["fingerprint_f64"] = fp
    bak, tmp = path / "heads.pt.bak", path / "heads.pt.tmp"
    shutil.copy2(path / "heads.pt", bak)
    torch.save(heads, tmp)
    os.replace(tmp, path / "heads.pt")
    logger.info(f"  지문 기록 {fp:.6f}  (백업: {bak.name})")

    if note:
        mp = path / "metrics.json"
        m = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
        m["promoted"] = {**m.get("promoted", {}), **note, "fingerprint_f64": fp}
        mtmp = path / "metrics.json.tmp"
        mtmp.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(mtmp, mp)
    return fp


def main() -> None:
    ap = argparse.ArgumentParser(description="체크포인트 승격")
    ap.add_argument("src", nargs="?", help="원본(실험) 체크포인트. --stamp-only면 생략")
    ap.add_argument("dst", help="운영 이름")
    ap.add_argument("--stamp-only", action="store_true",
                    help="이미 복사된 디렉터리에 검증·지문·기록만 얹는다(복사 안 함)")
    ap.add_argument("--why", default="", help="승격 사유 한 줄")
    a = ap.parse_args()
    dst = Path(a.dst)

    if not a.stamp_only:
        src = Path(a.src)
        if dst.exists():
            raise SystemExit(f"{dst} 가 이미 있다 — 덮어쓰지 않는다. --stamp-only를 쓰거나 지울 것")
        logger.info(f"========== 승격 {src} → {dst} ==========")
        shutil.copytree(src, dst)      # 복사이지 이동이 아니다 — 실험 기록을 남긴다
    else:
        logger.info(f"========== 검증·지문 기록 {dst} ==========")

    n_keys, diff = verify_encoder(dst)
    logger.info(f"  ① 인코더 왕복  키 {n_keys}개 · 가중치 차이 {diff:.6f}  "
                f"{'OK' if diff < 1e-6 else '★ 0이 아니다 — 확인할 것'}")
    if diff >= 1e-6:
        raise SystemExit("인코더가 완전히 복원되지 않는다 — 승격 중단")

    fp = stamp(dst, {
        "name": dst.name,
        "source_checkpoint": str(a.src) if a.src else "(stamp-only)",
        "why": a.why or "실험 폴더(models/_*)를 운영이 가리키지 않게 한다",
        "verified": {"encoder_keys": n_keys, "encoder_weight_diff": diff},
    })
    logger.info(f"  ② 지문 기록   {fp:.6f} (save()와 같은 함수로 계산)")

    reloaded = ArticleMultiLabelElectra.load(dst)        # 여기서 지문 검사가 실제로 돈다
    logger.info(f"  ③ load() 왕복  통과 — 조 {len(reloaded.article_names)}개")
    logger.info(f"  승격 완료: {dst}")


if __name__ == "__main__":
    main()
