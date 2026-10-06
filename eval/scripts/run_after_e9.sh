#!/usr/bin/env bash
# e9_local_labeling 종료 후 OOM 실패분 + Llama-3.2 전체 실행
# 대상: qwen25_7b_s123, llama31_s7, llama31_s123, llama32_1b ×3, llama32_3b ×3
set -a && source .env && set +a

PYTHON=".venv/bin/python"
LOG_DIR="logs/version_sweep"

run() {
    local gpu=$1 tag=$2 base=$3 seed=$4 extra="${5:-}"
    local log="$LOG_DIR/${tag}.log"
    # 이미 성공한 결과 건너뜀
    if [ -f "data/eval/sllm_lora_${tag}.json" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP: ${tag} (결과 이미 있음)"
        return
    fi
    echo "[$(date '+%H:%M:%S')] GPU${gpu} START: ${tag}"
    $PYTHON -m eval.model.sllm_lora_eval \
        --gpu "$gpu" --tag "$tag" --base "$base" --seed "$seed" $extra \
        > "$log" 2>&1
    local f1=$(grep 'F1(전체' "$log" | tail -1 | grep -oP '\d+\.\d+%' | head -1)
    echo "[$(date '+%H:%M:%S')] GPU${gpu} DONE:  ${tag}  (F1: ${f1})"
}

echo "[$(date '+%H:%M:%S')] GPU1 e9 프로세스 종료 대기 중..."
while pgrep -f "e9_local_labeling.*--gpu 1" > /dev/null 2>&1; do sleep 30; done
echo "[$(date '+%H:%M:%S')] GPU1 e9 종료 확인. 메모리 해제 대기 (60s)..."
sleep 60

run 1 qwen25_7b_s123 Qwen/Qwen2.5-7B-Instruct          123
run 1 llama31_s7     meta-llama/Llama-3.1-8B-Instruct    7
run 1 llama31_s123   meta-llama/Llama-3.1-8B-Instruct  123

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
" >> "$LOG_DIR/llama32_download.log" 2>&1 && echo "[$(date '+%H:%M:%S')] 다운로드 완료" || echo "[$(date '+%H:%M:%S')] 다운로드 실패 — logs/version_sweep/llama32_download.log 확인"

run 1 llama32_1b_s42  meta-llama/Llama-3.2-1B-Instruct  42
run 1 llama32_1b_s7   meta-llama/Llama-3.2-1B-Instruct   7
run 1 llama32_1b_s123 meta-llama/Llama-3.2-1B-Instruct 123
run 1 llama32_3b_s42  meta-llama/Llama-3.2-3B-Instruct  42
run 1 llama32_3b_s7   meta-llama/Llama-3.2-3B-Instruct   7
run 1 llama32_3b_s123 meta-llama/Llama-3.2-3B-Instruct 123

echo "[$(date '+%H:%M:%S')] === 전체 후속 재실행 완료 ==="
