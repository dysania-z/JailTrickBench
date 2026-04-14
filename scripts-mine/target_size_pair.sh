!/usr/bin/env bash

# ============================================================================
# 原始脚本备份
# ============================================================================

export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID


python -u main.py \
    --target_model_path /home/ubuntu/data/models/defense/Llama-2-7b-chat-hf \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model /home/ubuntu/data/models/vicuna-7b-v1.5 \
    --instructions_path ./data/harmful_bench_1.csv \
    --save_result_path ./exp_results/target_size_pair/ \
    --agent_recheck \
    --n_iterations 10 \
    --exp_name size_llama-7b_pair \
    --target_system_message null

python -u main.py \
    --target_model_path /home/ubuntu/data/models/Llama-2-13b-chat-hf \
    --defense_type None_defense \
    --attack PAIR \
    --attack_model /home/ubuntu/data/models/vicuna-7b-v1.5 \
    --instructions_path ./data/harmful_bench_1.csv \
    --save_result_path ./exp_results/target_size_pair/ \
    --agent_recheck \
    --n_iterations 10 \
    --exp_name system_llama2-13b_pair \
    --target_system_message null

# python -u main.py \
#     --target_model_path lmsys/vicuna-7b-v1.5 \
#     --defense_type None_defense \
#     --attack PAIR \
#     --attack_model lmsys/vicuna-13b-v1.5 \
#     --instructions_path ./data/harmful_bench_50.csv \
#     --save_result_path ./exp_results/trick_target_system_pair/ \
#     --agent_evaluation \
#     --resume_exp \
#     --agent_recheck \
#     --n_iterations 12 \
#     --exp_name system_vicuna-7b_null \
#     --target_system_message null

# python -u main.py \
#     --target_model_path meta-llama/Llama-2-7b-chat-hf \
#     --defense_type None_defense \
#     --attack PAIR \
#     --attack_model lmsys/vicuna-13b-v1.5 \
#     --instructions_path ./data/harmful_bench_50.csv \
#     --save_result_path ./exp_results/trick_target_system_pair/ \
#     --agent_evaluation \
#     --resume_exp \
#     --agent_recheck \
#     --n_iterations 12 \
#     --exp_name system_llama-7b_safe \
#     --target_system_message safe

# python -u main.py \
#     --target_model_path meta-llama/Meta-Llama-3-8B-Instruct \
#     --defense_type None_defense \
#     --attack PAIR \
#     --attack_model lmsys/vicuna-13b-v1.5 \
#     --instructions_path ./data/harmful_bench_50.csv \
#     --save_result_path ./exp_results/trick_target_system_pair/ \
#     --agent_evaluation \
#     --resume_exp \
#     --agent_recheck \
#     --n_iterations 12 \
#     --exp_name system_llama3-8b_safe \
#     --target_system_message safe

# python -u main.py \
#     --target_model_path lmsys/vicuna-7b-v1.5 \
#     --defense_type None_defense \
#     --attack PAIR \
#     --attack_model lmsys/vicuna-13b-v1.5 \
#     --instructions_path ./data/harmful_bench_50.csv \
#     --save_result_path ./exp_results/trick_target_system_pair/ \
#     --agent_evaluation \
#     --resume_exp \
#     --agent_recheck \
#     --n_iterations 12 \
#     --exp_name system_vicuna-7b_safe \
#     --target_system_message safe

# ============================================================================
# 新脚本（当前生效）
# ============================================================================

# 目标模型规模对攻击成功率（AdvBench 子集）：Llama2-7B/13B 与 Vicuna-7B/13B × AutoDAN / PAIR
# 使用 4 张 GPU 分两批各跑 4 个任务；每任务内 CUDA_VISIBLE_DEVICES 单卡 + --device_id 0

# 数据集：data/advbench.csv → harmful_bench_50.csv（AdvBench 常用 50 条子集，与本仓库其它论文实验一致）
# 若需完整 520 条，请自行准备同格式 CSV 并修改 DATASET。

# 默认开启 GPT Agent 评估 ASR（需 OPENAI_API_KEY）。仅用启发式判越狱可设：export RUN_AGENT_EVAL=0

# set -euo pipefail

# JTB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# cd "$JTB_ROOT"

# # 绘图用解释器：可 export PYTHON_BIN=... 覆盖；否则优先使用已装 matplotlib 的 jailbreak 环境
# _JAILBREAK_PY="/home/ubuntu/data/miniconda3/envs/jailbreak/bin/python"
# if [[ -n "${PYTHON_BIN:-}" ]]; then
#   :
# elif [[ -x "$_JAILBREAK_PY" ]] && "$_JAILBREAK_PY" -c "import matplotlib" 2>/dev/null; then
#   PYTHON_BIN="$_JAILBREAK_PY"
# else
#   PYTHON_BIN="python"
# fi

# export CUDA_DEVICE_ORDER=PCI_BUS_ID

# DATASET="${JTB_ROOT}/data/harmful_bench_10.csv"
# SAVE_ROOT="${JTB_ROOT}/exp_results/target_size_pair"
# LOG_ROOT="${JTB_ROOT}/exp_logs/target_size_pair"
# mkdir -p "$SAVE_ROOT" "$LOG_ROOT"

# ATTACKER="/home/ubuntu/data/models/vicuna-13b-v1.5"
# LLAMA7="/home/ubuntu/data/models/defense/Llama-2-7b-chat-hf"
# LLAMA13="/home/ubuntu/data/models/Llama-2-13b-chat-hf"
# VICUNA7="/home/ubuntu/data/models/vicuna-7b-v1.5"
# VICUNA13="/home/ubuntu/data/models/vicuna-13b-v1.5"

# TS=$(date +%Y%m%d_%H%M%S)

# if [[ "${RUN_AGENT_EVAL:-1}" == "1" ]]; then
#   AGENT_FLAGS=(--agent_evaluation --resume_exp --agent_recheck)
# else
#   AGENT_FLAGS=(--resume_exp)
# fi

# run_one() {
#   local gpu=$1
#   local exp_name=$2
#   local target_path=$3
#   local attack_method=$4

#   local -a extra
#   if [[ "$attack_method" == "PAIR" ]]; then
#     extra=(--attack PAIR --attack_model "$ATTACKER" --n_iterations 10)
#   else
#     extra=(--attack AutoDAN --gcg_attack_budget 200)
#   fi

#   (
#     export CUDA_VISIBLE_DEVICES=$gpu
#     python -u main.py \
#       "${extra[@]}" \
#       --target_model_path "$target_path" \
#       --defense_type None_defense \
#       --instructions_path "$DATASET" \
#       --save_result_path "$SAVE_ROOT" \
#       --exp_name "$exp_name" \
#       --target_system_message null \
#       --device_id 0 \
#       "${AGENT_FLAGS[@]}" \
#       2>&1 | tee "${LOG_ROOT}/${exp_name}_${TS}_gpu${gpu}.log"
#   ) &
# }

# echo "=== Wave 1: Llama-2 (7B/13B) ==="
# run_one 4 "size_llama2-7b_autodan" "$LLAMA7" "AutoDAN"
# run_one 5 "size_llama2-13b_autodan" "$LLAMA13" "AutoDAN"
# run_one 6 "size_llama2-7b_pair" "$LLAMA7" "PAIR"
# run_one 7 "size_llama2-13b_pair" "$LLAMA13" "PAIR"
# wait
# echo "Wave 1 done."

# echo "=== Wave 2: Vicuna (7B/13B) ==="
# run_one 4 "size_vicuna-7b_autodan" "$VICUNA7" "AutoDAN"
# run_one 5 "size_vicuna-13b_autodan" "$VICUNA13" "AutoDAN"
# run_one 6 "size_vicuna-7b_pair" "$VICUNA7" "PAIR"
# run_one 7 "size_vicuna-13b_pair" "$VICUNA13" "PAIR"
# wait
# echo "Wave 2 done. All jobs finished."

# echo "=== 生成 ASR 柱状图 ==="
# if ! "$PYTHON_BIN" -c "import matplotlib" 2>/dev/null; then
#   echo "跳过绘图：${PYTHON_BIN} 未安装 matplotlib。请: pip install matplotlib 或 export PYTHON_BIN=指向已安装 matplotlib 的 python" >&2
# else
#   set +e
#   "$PYTHON_BIN" "${JTB_ROOT}/plotting/plot_target_size_asr.py" \
#     --result_dir "${SAVE_ROOT}" \
#     --out "${SAVE_ROOT}/target_size_asr.png"
#   plot_rc=$?
#   set -e
#   if [[ "$plot_rc" -ne 0 ]]; then
#     echo "警告：ASR 图生成失败（退出码 ${plot_rc}）。" >&2
#   fi
# fi
