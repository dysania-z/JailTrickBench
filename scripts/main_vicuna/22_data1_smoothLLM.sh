#!/usr/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

LOG_DIR="./exp_logs/smoothLLM/AutoDAN/"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

python -u main.py \
    --target_model_path /home/ubuntu/data/models/vicuna-7b-v1.5 \
    --defense_type smoothLLM \
    --attack AutoDAN \
    --instructions_path ./data/harmful_bench_10.csv \
    --save_result_path ./exp_results/main_vicuna/smoothLLM/autoDAN \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name main_vicuna_smoothllm \
    2>&1 | tee -ai "${LOG_DIR}${TIMESTAMP}.txt"

