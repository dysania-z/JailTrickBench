#!/usr/bin/bash
set -e

# 定义总的并行进程数（显卡数）
TOTAL_NUM=8

for i in {0..7}
do
    # 为每个进程分配一张显卡
    export CUDA_VISIBLE_DEVICES=$i
    
    # 启动进程
    python -u main.py \
        --target_model_path /home/ubuntu/data/models/vicuna-7b-v1.5 \
        --defense_type None_defense \
        --attack GCG \
        --instructions_path ./data/harmful_bench_10.csv \
        --save_result_path ./exp_results/main_vicuna/no_defense/GCG \
        --agent_evaluation \
        --resume_exp \
        --agent_recheck \
        --exp_name main_vicuna_none_defense \
        --data_split \
        --data_split_total_num $TOTAL_NUM \
        --data_split_idx $i \
        2>&1 | tee -ai ./exp_logs/main_vicuna/harmful_bench_50/None_defense/GCG/main_vicuna_none_defense/1_split0_harmful_bench_50_None_defense_GCG_vicuna-13b-v1.5_$(date +\%Y\%m\%d_\%H\%M\%S).txt &

    echo "Started process for GPU $i (Task index $i/$TOTAL_NUM)"
done

# 等待所有后台任务完成
wait
echo "All parallel tasks finished!"