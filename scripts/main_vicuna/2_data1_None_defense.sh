#!/usr/bin/bash
set -e

LOG_DIR="./exp_logs/None_defense/PAIR/"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

python -u main.py \
    --target_model_path /home/ubuntu/data/models/vicuna-7b-v1.5 \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model /home/ubuntu/data/models/vicuna-7b-v1.5 \
    --instructions_path ./data/harmful_bench_10.csv \
    --save_result_path ./exp_results/main_vicuna/no_defense/PAIR \
    --agent_evaluation \
    --resume_exp \
    --agent_recheck \
    --exp_name main_vicuna_none_defense \
    2>&1 | tee -ai "${LOG_DIR}${TIMESTAMP}.txt"


