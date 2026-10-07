#!/usr/bin/env bash
# Llama-3.2 스윕: resume sweep의 GPU1 큐 완료 대기 후 실행
# .env에서 HF_TOKEN 로드
set -a && source .env && set +a

PYTHON=".venv/bin/python"
LOG_DIR="logs/version_sweep"

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

# qwen25_7b_s123 완료 대기 (마지막 GPU1 resume 작업)
echo "[$(date '+%H:%M:%S')] Llama-3.2 스윕: qwen25_7b_s123 완료 대기 중..."
until [ -f data/eval/model/sllm_lora_qwen25_7b_s123.json ]; do sleep 30; done
echo "[$(date '+%H:%M:%S')] GPU1 확보. Llama-3.2 다운로드 시작..."

$PYTHON -c "
from huggingface_hub import snapshot_download
import os
for mid in ['meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct']:
    print(f'  DL: {mid}')
    snapshot_download(mid, token=os.environ['HF_TOKEN'],
                      ignore_patterns=['*.msgpack','*.h5','flax_*','tf_*'])
    print(f'  OK: {mid}')
"

echo "[$(date '+%H:%M:%S')] 다운로드 완료. Llama-3.2 학습 시작."
run 1 llama32_1b_s42  meta-llama/Llama-3.2-1B-Instruct  42
run 1 llama32_1b_s7   meta-llama/Llama-3.2-1B-Instruct   7
run 1 llama32_1b_s123 meta-llama/Llama-3.2-1B-Instruct 123
run 1 llama32_3b_s42  meta-llama/Llama-3.2-3B-Instruct  42
run 1 llama32_3b_s7   meta-llama/Llama-3.2-3B-Instruct   7
run 1 llama32_3b_s123 meta-llama/Llama-3.2-3B-Instruct 123

echo "[$(date '+%H:%M:%S')] === Llama-3.2 스윕 완료 ==="
