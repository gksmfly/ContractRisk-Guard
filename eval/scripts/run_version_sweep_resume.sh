#!/usr/bin/env bash
# 중단된 스윕 복구: Llama-3.2 제외, 남은 실험만 실행
PYTHON=".venv/bin/python"
LOG_DIR="logs/version_sweep"
mkdir -p "$LOG_DIR"

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

# GPU 0: qwen25_1b5_s7 완료 대기 (PID 4152692) → 나머지 큐
gpu0_resume() {
    echo "[$(date '+%H:%M:%S')] GPU0: qwen25_1b5_s7 완료 대기 중 (PID 4152692)..."
    while kill -0 4152692 2>/dev/null; do sleep 10; done
    echo "[$(date '+%H:%M:%S')] GPU0: 재개"

    run 0 qwen25_1b5_s123 Qwen/Qwen2.5-1.5B-Instruct 123
    run 0 llama31_s7      meta-llama/Llama-3.1-8B-Instruct   7
    run 0 llama31_s123    meta-llama/Llama-3.1-8B-Instruct 123
    run 0 exaone35_2b4_s42  LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct  42
    run 0 exaone35_2b4_s7   LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct   7
    run 0 exaone35_2b4_s123 LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct 123
}

# GPU 1: qwen25_7b_s42 완료 대기 (PID 4146884) → s7, s123
gpu1_resume() {
    echo "[$(date '+%H:%M:%S')] GPU1: qwen25_7b_s42 완료 대기 중 (PID 4146884)..."
    while kill -0 4146884 2>/dev/null; do sleep 10; done
    echo "[$(date '+%H:%M:%S')] GPU1: 재개"

    run 1 qwen25_7b_s7   Qwen/Qwen2.5-7B-Instruct   7
    run 1 qwen25_7b_s123 Qwen/Qwen2.5-7B-Instruct 123
}

gpu0_resume &
gpu1_resume &
wait

echo ""
echo "[$(date '+%H:%M:%S')] === 복구 완료 ==="
echo ""
echo "── 신규 실험 결과 요약 ─────────────────────────────────────────────"
for tag in qwen25_1b5_s42 qwen25_1b5_s7 qwen25_1b5_s123 \
           qwen25_7b_s42 qwen25_7b_s7 qwen25_7b_s123 \
           llama31_s7 llama31_s123 \
           exaone35_2b4_s42 exaone35_2b4_s7 exaone35_2b4_s123; do
    f="data/eval/sllm_lora_${tag}.json"
    if [ -f "$f" ]; then
        python3 -c "
import json
d = json.load(open('$f'))
print(f\"  ${tag:28s}  F1={d.get('f1_all',0)*100:.1f}%  FPR={d.get('fpr_intact',0)*100:.1f}%  재현={d.get('recall_any',0)*100:.1f}%\")
"
    else
        echo "  ${tag:28s}  (결과 없음)"
    fi
done
