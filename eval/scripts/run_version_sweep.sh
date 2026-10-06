#!/usr/bin/env bash
# GPU 0/1 병렬 버전 스윕 실험
# 사용: bash backend/eval/run_version_sweep.sh
set -euo pipefail

PYTHON=".venv/bin/python"
LOG_DIR="logs/version_sweep"
mkdir -p "$LOG_DIR"

echo "[$(date '+%H:%M:%S')] === 버전 스윕 시작 ==="

# ── 단계 1: 신규 모델 병렬 다운로드 ───────────────────────────────────────
echo "[$(date '+%H:%M:%S')] 신규 모델 다운로드 시작 (백그라운드)"
$PYTHON - <<'EOF' &
from huggingface_hub import snapshot_download
import concurrent.futures, sys

models = [
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

def dl(mid):
    print(f"  DL start: {mid}", flush=True)
    snapshot_download(mid, ignore_patterns=["*.msgpack", "*.h5", "flax_*", "tf_*"])
    print(f"  DL done:  {mid}", flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
    list(ex.map(dl, models))

print("  모든 다운로드 완료", flush=True)
EOF
DOWNLOAD_PID=$!

# ── 실행 헬퍼 ───────────────────────────────────────────────────────────────
run() {
    local gpu=$1 tag=$2 base=$3 seed=$4
    local extra="${5:-}"
    local log="$LOG_DIR/${tag}.log"
    echo "[$(date '+%H:%M:%S')] GPU${gpu} START: ${tag}"
    $PYTHON -m eval.model.sllm_lora_eval \
        --gpu "$gpu" --tag "$tag" --base "$base" --seed "$seed" $extra \
        > "$log" 2>&1
    echo "[$(date '+%H:%M:%S')] GPU${gpu} DONE:  ${tag}  (F1: $(grep 'F1(전체' $log | tail -1 | grep -oP '\d+\.\d+%' | head -1))"
}

# ── GPU 0 큐: Qwen1.5B → Llama3.1 추가시드 → EXAONE2.4B ──────────────────
gpu0_queue() {
    run 0 qwen25_1b5_s42  Qwen/Qwen2.5-1.5B-Instruct  42
    run 0 qwen25_1b5_s7   Qwen/Qwen2.5-1.5B-Instruct   7
    run 0 qwen25_1b5_s123 Qwen/Qwen2.5-1.5B-Instruct 123

    run 0 llama31_s7      meta-llama/Llama-3.1-8B-Instruct  7
    run 0 llama31_s123    meta-llama/Llama-3.1-8B-Instruct 123

    # EXAONE-3.5-2.4B: 다운로드 기다리기
    echo "[$(date '+%H:%M:%S')] GPU0: EXAONE-2.4B 다운로드 대기 중..."
    wait $DOWNLOAD_PID 2>/dev/null || true
    run 0 exaone35_2b4_s42  LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct  42
    run 0 exaone35_2b4_s7   LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct   7
    run 0 exaone35_2b4_s123 LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct 123
}

# ── GPU 1 큐: Qwen7B → Llama3.2-1B → Llama3.2-3B ─────────────────────────
gpu1_queue() {
    run 1 qwen25_7b_s42  Qwen/Qwen2.5-7B-Instruct  42
    run 1 qwen25_7b_s7   Qwen/Qwen2.5-7B-Instruct   7
    run 1 qwen25_7b_s123 Qwen/Qwen2.5-7B-Instruct 123

    # Llama3.2는 다운로드 완료 후 실행
    wait $DOWNLOAD_PID 2>/dev/null || true
    run 1 llama32_1b_s42  meta-llama/Llama-3.2-1B-Instruct  42
    run 1 llama32_1b_s7   meta-llama/Llama-3.2-1B-Instruct   7
    run 1 llama32_1b_s123 meta-llama/Llama-3.2-1B-Instruct 123

    run 1 llama32_3b_s42  meta-llama/Llama-3.2-3B-Instruct  42
    run 1 llama32_3b_s7   meta-llama/Llama-3.2-3B-Instruct   7
    run 1 llama32_3b_s123 meta-llama/Llama-3.2-3B-Instruct 123
}

# ── 병렬 실행 ──────────────────────────────────────────────────────────────
gpu0_queue &
GPU0_PID=$!
gpu1_queue &
GPU1_PID=$!

wait $GPU0_PID $GPU1_PID

echo ""
echo "[$(date '+%H:%M:%S')] === 전체 완료 ==="
echo ""
echo "── 결과 요약 ──────────────────────────────────────────────────────"
for log in "$LOG_DIR"/*.log; do
    tag=$(basename "$log" .log)
    f1=$(grep 'F1(전체' "$log" 2>/dev/null | tail -1 | grep -oP '\d+\.\d+%' | head -1 || echo "?")
    fpr=$(grep '오경보' "$log" 2>/dev/null | tail -1 | grep -oP '\d+\.\d+%' | head -1 || echo "?")
    echo "  $tag: F1=$f1  FPR=$fpr"
done
