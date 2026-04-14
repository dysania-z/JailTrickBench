#!/usr/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=6
export CUDA_DEVICE_ORDER=PCI_BUS_ID

LOG_DIR="./exp_logs/safety_training/PAIR/"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

python -u /home/ubuntu/data/JailTrickBench/main.py \
    --target_model_path /home/ubuntu/data/models/defense/tuning/Llama-2-7b-chat-hf_safety_training_lora_8ep \
    --defense_type safety_tuning \
    --attack PAIR \
    --attack_model /home/ubuntu/data/models/vicuna-7b-v1.5 \
    --instructions_path ./data/harmful_bench_1.csv \
    --save_result_path ./exp_results \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name safetytraining_pair \
    2>&1 | tee -ai "${LOG_DIR}${TIMESTAMP}.txt"

