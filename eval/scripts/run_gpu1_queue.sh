#!/usr/bin/env bash
# GPU 1 단일 큐: qwen25_7b_s7 완료 후 나머지 전부 실행
set -a && source .env && set +a

PYTHON=".venv/bin/python"
LOG_DIR="logs/version_sweep"

run() {
    local gpu=$1 tag=$2 base=$3 seed=$4 extra="${5:-}"
    local log="$LOG_DIR/${tag}.log"
    echo "[$(date '+%H:%M:%S')] GPU${gpu} START: ${tag}"
    $PYTHON -m eval.model.sllm_lora_eval \
        --gpu "$gpu" --tag "$tag" --base "$base" --seed "$seed" $extra \
        > "$log" 2>&1
    local f1=$(grep 'F1(전체' "$log" | tail -1 | grep -oP '\d+\.\d+%' | head -1)
    echo "[$(date '+%H:%M:%S')] GPU${gpu} DONE:  ${tag}  (F1: ${f1})"
}

# qwen25_7b_s7 완료 대기 (PID 4159037)
echo "[$(date '+%H:%M:%S')] qwen25_7b_s7 완료 대기 중 (PID 4159037)..."
while kill -0 4159037 2>/dev/null; do sleep 15; done
echo "[$(date '+%H:%M:%S')] GPU1 확보. 잔여 큐 시작."

# ── Qwen2.5-7B 잔여 시드 ─────────────────────────────────────────────────
run 1 qwen25_7b_s123 Qwen/Qwen2.5-7B-Instruct 123

# ── Qwen2.5-1.5B 잔여 시드 (GPU 0 OOM으로 실패했던 것) ──────────────────
run 1 qwen25_1b5_s123 Qwen/Qwen2.5-1.5B-Instruct 123

# ── Llama-3.1-8B 잔여 시드 ───────────────────────────────────────────────
run 1 llama31_s7   meta-llama/Llama-3.1-8B-Instruct  7
run 1 llama31_s123 meta-llama/Llama-3.1-8B-Instruct 123

# ── EXAONE-3.5-2.4B 전체 ─────────────────────────────────────────────────
run 1 exaone35_2b4_s42  LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct  42
run 1 exaone35_2b4_s7   LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct   7
run 1 exaone35_2b4_s123 LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct 123

# ── Llama-3.2 다운로드 후 실행 ───────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Llama-3.2 다운로드 시작..."
$PYTHON -c "
from huggingface_hub import snapshot_download
import os
for mid in ['meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct']:
    print(f'  DL: {mid}', flush=True)
    snapshot_download(mid, token=os.environ['HF_TOKEN'],
                      ignore_patterns=['*.msgpack','*.h5','flax_*','tf_*'])
    print(f'  OK: {mid}', flush=True)
"

run 1 llama32_1b_s42  meta-llama/Llama-3.2-1B-Instruct  42
run 1 llama32_1b_s7   meta-llama/Llama-3.2-1B-Instruct   7
run 1 llama32_1b_s123 meta-llama/Llama-3.2-1B-Instruct 123
run 1 llama32_3b_s42  meta-llama/Llama-3.2-3B-Instruct  42
run 1 llama32_3b_s7   meta-llama/Llama-3.2-3B-Instruct   7
run 1 llama32_3b_s123 meta-llama/Llama-3.2-3B-Instruct 123

echo "[$(date '+%H:%M:%S')] === 전체 GPU1 큐 완료 ==="
