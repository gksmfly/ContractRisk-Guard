# backend/training/plot_history.py
"""
학습 곡선 시각화 — 과적합을 눈으로 확인하기 위한 PNG를 체크포인트 옆에 남긴다.

## 왜 필요한가

`metrics.json`의 `history`에 epoch별 수치가 쌓이지만, 숫자만 봐서는 **"아직 학습 중"과
"이미 외우기 시작했다"를 구분할 수 없다.** 특히 이 프로젝트는 학습 데이터가 수백 건대라
과적합이 빠르게 오는데, 지금까지 `best_risk_macro_f1` 하나만 보고 체크포인트를 골라 왔다.

과적합의 표준 신호는 **검증 손실이 바닥을 치고 다시 오르는 지점**이다. 그래서
`train.py::evaluate()`가 검증 손실도 함께 기록하도록 고쳤다(그전에는 학습 손실만 있었다).

## 무엇을 그리나

세로로 두 패널, x축은 epoch 공유:

  1. Loss      — train vs validation. 두 곡선이 벌어지기 시작하는 지점이 과적합 시작
  2. Macro F1  — domain / risk. 실제로 우리가 고르는 기준

**두 축을 한 그림에 겹치지 않는다**(loss와 F1은 척도가 달라 이중축을 쓰면 아무 관계나
있어 보이게 만들 수 있다). 세로선 두 개로 (a) 체크포인트가 저장된 epoch와
(b) 검증 손실 최저 epoch를 표시하는데, **이 둘이 어긋나면 F1 기준으로 고른 체크포인트가
이미 과적합 구간에 있다는 뜻**이다.

차트 라벨은 영문이다 — 이 환경에 한글 폰트가 없어 한글을 쓰면 글자가 깨진다.

실행:
    .venv/bin/python -m backend.training.plot_history models/_fixedsplit/seed42
    .venv/bin/python -m backend.training.plot_history models/_fixedsplit/*   # 여러 개
"""

import argparse
import json
from pathlib import Path
from typing import Any

from backend.utils import load_logger

logger = load_logger("plot_history.log")

# dataviz 스킬의 검증된 기본 팔레트(light) — 카테고리 슬롯 1·2·3.
# `validate_palette.js`로 확인: 명도대·채도·색각 분리·명도 대비 전부 통과
# (aqua는 배경 대비 2.74:1이라 "직접 라벨 필요" 경고 → 선 끝에 라벨을 단다).
_BLUE, _ORANGE, _AQUA = "#2a78d6", "#eb6834", "#1baf7a"
_INK, _MUTED, _GRID = "#0b0b0b", "#52514e", "#e3e2df"
_SURFACE = "#fcfcfb"


def _line(ax, xs, ys, color, label):
    """2px 선 + 8px 마커 + 선 끝 직접 라벨(범례에만 의존하지 않게)."""
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
            markeredgecolor=_SURFACE, markeredgewidth=1, label=label, zorder=3)
    ax.annotate(label, xy=(xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                color=color, fontsize=9, va="center", fontweight="medium")


def _style(ax, ylabel):
    ax.set_facecolor(_SURFACE)
    ax.grid(True, color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.set_ylabel(ylabel, color=_INK, fontsize=10)


def _mark_epoch(ax, epoch, color, text, va="top", last_epoch=None):
    """세로선 + 라벨. 선이 오른쪽 끝에 가까우면 라벨을 왼쪽으로 눕힌다.

    마지막 epoch가 최고점인 경우(5시드 중 2개가 그랬다) 라벨이 축 밖으로 나가
    범례·선 끝 라벨과 겹친다 — 실제로 seed123에서 발생했다.
    """
    ax.axvline(epoch, color=color, linewidth=1.2, linestyle="--", alpha=0.7, zorder=1)
    y = ax.get_ylim()[1] if va == "top" else ax.get_ylim()[0]
    near_right = last_epoch is not None and epoch >= last_epoch - 0.5
    ax.annotate(
        text, xy=(epoch, y),
        xytext=(-4 if near_right else 3, -10 if va == "top" else 10),
        textcoords="offset points", color=color, fontsize=8, va=va,
        ha="right" if near_right else "left",
    )


def diagnose(history: list[dict]) -> dict[str, Any]:
    """체크포인트 선택 epoch와 검증손실 최저 epoch를 비교해 과적합 여부를 판정한다."""
    epochs = [h["epoch"] for h in history]
    risk_f1 = [h.get("risk_macro_f1") for h in history]
    val_loss = [h.get("val_loss") for h in history]

    best_f1_epoch = epochs[risk_f1.index(max(risk_f1))]
    have_val = all(v is not None for v in val_loss)
    best_loss_epoch = epochs[val_loss.index(min(val_loss))] if have_val else None

    verdict = "검증 손실이 기록되지 않아 과적합 판정 불가(train.py 재학습 필요)"
    if have_val:
        if best_loss_epoch < best_f1_epoch:
            verdict = (f"과적합 구간에서 체크포인트를 골랐다 — 검증 손실은 epoch "
                       f"{best_loss_epoch}에서 최저인데 저장은 epoch {best_f1_epoch}")
        elif best_loss_epoch == epochs[-1]:
            verdict = "검증 손실이 마지막까지 내려간다 — 아직 과적합 전, 더 학습해볼 여지가 있다"
        else:
            verdict = f"검증 손실 최저(epoch {best_loss_epoch})와 체크포인트 선택이 어긋나지 않는다"
    return {
        "best_f1_epoch": best_f1_epoch,
        "best_val_loss_epoch": best_loss_epoch,
        "has_val_loss": have_val,
        "verdict": verdict,
    }


def plot_history(model_dir: Path, out_path: Path | None = None) -> Path | None:
    """`model_dir/metrics.json` → `model_dir/training_curve.png`."""
    import matplotlib
    matplotlib.use("Agg")          # 헤드리스 환경(학습 서버)에서 실행되므로 GUI 백엔드 금지
    import matplotlib.pyplot as plt

    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        logger.warning(f"  {metrics_path} 없음 — 건너뜀")
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = metrics.get("history") or []
    if len(history) < 2:
        logger.warning(f"  {model_dir.name}: epoch가 {len(history)}개뿐 — 곡선을 그릴 수 없다")
        return None

    diag = diagnose(history)
    epochs = [h["epoch"] for h in history]

    fig, (ax_loss, ax_f1) = plt.subplots(
        2, 1, figsize=(8.5, 6.6), sharex=True, gridspec_kw={"hspace": 0.18}
    )
    fig.patch.set_facecolor(_SURFACE)

    # ── 1. Loss — 과적합 신호 ──
    _line(ax_loss, epochs, [h["train_loss"] for h in history], _BLUE, "train")
    if diag["has_val_loss"]:
        _line(ax_loss, epochs, [h["val_loss"] for h in history], _ORANGE, "validation")
        # 계열이 둘일 때만 범례를 둔다 — 하나뿐이면 선 끝 라벨로 충분하고,
        # 우상단 범례가 "saved (best F1)" 세로선 라벨과 겹친다.
        ax_loss.legend(frameon=False, fontsize=9, labelcolor=_MUTED, loc="center right")
    _style(ax_loss, "Loss")

    # ── 2. Macro F1 — 체크포인트 선택 기준 ──
    _line(ax_f1, epochs, [h["domain_macro_f1"] for h in history], _BLUE, "domain")
    _line(ax_f1, epochs, [h["risk_macro_f1"] for h in history], _AQUA, "risk")
    _style(ax_f1, "Macro F1")
    ax_f1.set_xlabel("Epoch", color=_INK, fontsize=10)
    ax_f1.set_xticks(epochs)
    ax_f1.legend(frameon=False, fontsize=9, labelcolor=_MUTED, loc="lower right")

    # 두 세로선이 어긋나 있으면 F1로 고른 체크포인트가 이미 과적합 구간이라는 뜻이다.
    # 세로선은 두 패널에 다 긋되 **글자는 위 패널에만** 단다 — 아래 패널에 또 쓰면
    # F1 곡선(0.9 부근)과 겹친다.
    ax_f1.axvline(diag["best_f1_epoch"], color=_MUTED, linewidth=1.2, linestyle="--",
                  alpha=0.7, zorder=1)
    _mark_epoch(ax_loss, diag["best_f1_epoch"], _MUTED,
                f"saved (best F1) e{diag['best_f1_epoch']}", last_epoch=epochs[-1])
    if diag["has_val_loss"] and diag["best_val_loss_epoch"] != diag["best_f1_epoch"]:
        _mark_epoch(ax_loss, diag["best_val_loss_epoch"], _ORANGE,
                    f"min val loss e{diag['best_val_loss_epoch']}", va="bottom",
                    last_epoch=epochs[-1])

    subtitle = (f"{metrics.get('base_model', '?')} · {metrics.get('risk_scheme', '?')} · "
                f"train {metrics.get('train_samples', '?')} / val {metrics.get('val_samples', '?')}")
    fig.suptitle(f"Training curve — {model_dir.name}", x=0.06, ha="left",
                 fontsize=13, color=_INK, fontweight="semibold", y=0.98)
    fig.text(0.06, 0.935, subtitle, ha="left", fontsize=9, color=_MUTED)
    fig.subplots_adjust(top=0.89, right=0.90, left=0.09, bottom=0.09)

    out_path = out_path or (model_dir / "training_curve.png")
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)

    logger.info(f"  {model_dir.name}: {out_path}")
    logger.info(f"    {diag['verdict']}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="학습 곡선 PNG 생성(과적합 확인용)")
    ap.add_argument("model_dirs", nargs="+", help="metrics.json이 있는 체크포인트 디렉터리")
    args = ap.parse_args()
    for d in args.model_dirs:
        plot_history(Path(d))


if __name__ == "__main__":
    main()
