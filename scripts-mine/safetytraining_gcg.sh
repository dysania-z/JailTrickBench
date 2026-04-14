#!/usr/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=6
export CUDA_DEVICE_ORDER=PCI_BUS_ID

LOG_DIR="./exp_logs/safety_training/GCG/"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

python -u main.py \
    --target_model_path /home/ubuntu/data/models/defense/tuning/Llama-2-7b-chat-hf_safety_training_lora_8ep \
    --defense_type safety_tuning \
    --attack GCG \
    --instructions_path ./data/harmful_bench_1.csv \
    --save_result_path ./exp_results \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name safetytraining-gcg \
    2>&1 | tee -ai "${LOG_DIR}${TIMESTAMP}.txt"


