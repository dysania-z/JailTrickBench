
set -euo pipefail

JTB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$JTB_ROOT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
# 绘图用；未 export 时默认 python（否则 set -u 下 $PYTHON_BIN 未定义会直接报错退出，图不会生成）
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET="${JTB_ROOT}/data/harmful_bench_15.csv"
SAVE_ROOT="${JTB_ROOT}/exp_results/target_size_pair"
LOG_ROOT="${JTB_ROOT}/exp_logs/target_size_pair"
mkdir -p "$SAVE_ROOT" "$LOG_ROOT"

ATTACKER="/home/ubuntu/data/models/vicuna-13b-v1.5"
LLAMA7="/home/ubuntu/data/models/defense/Llama-2-7b-chat-hf"
LLAMA13="/home/ubuntu/data/models/Llama-2-13b-chat-hf"
VICUNA7="/home/ubuntu/data/models/vicuna-7b-v1.5"
VICUNA13="/home/ubuntu/data/models/vicuna-13b-v1.5"

TS=$(date +%Y%m%d_%H%M%S)

if [[ "${RUN_AGENT_EVAL:-1}" == "1" ]]; then
  AGENT_FLAGS=(--agent_evaluation --resume_exp --agent_recheck)
else
  AGENT_FLAGS=(--resume_exp)
fi

run_one() {
  local gpu=$1
  local exp_name=$2
  local target_path=$3
  local attack_method=$4

  local -a extra
  if [[ "$attack_method" == "PAIR" ]]; then
    extra=(--attack PAIR --attack_model "$ATTACKER" --n_iterations 10)
  else
    extra=(--attack AutoDAN)
  fi

  (
    export CUDA_VISIBLE_DEVICES=$gpu
    python -u main.py \
      "${extra[@]}" \
      --target_model_path "$target_path" \
      --defense_type None_defense \
      --instructions_path "$DATASET" \
      --save_result_path "$SAVE_ROOT" \
      --exp_name "$exp_name" \
      --target_system_message null \
      --device_id 0 \
      "${AGENT_FLAGS[@]}" \
      2>&1 | tee "${LOG_ROOT}/${exp_name}_${TS}_gpu${gpu}.log"
  ) &
}

# PAIR 攻击侧(默认 Vicuna-13B) + 目标模型显存压力大时，用多卡 device_map 分摊；任意 Wave 都可调用，与 Wave 编号无关。
run_one_devices() {
  local devices=$1
  local exp_name=$2
  local target_path=$3
  local attack_method=$4

  local -a extra
  if [[ "$attack_method" == "PAIR" ]]; then
    extra=(--attack PAIR --attack_model "$ATTACKER" --n_iterations 10)
  else
    extra=(--attack AutoDAN)
  fi

  local log_tag="${devices//,/_}"
  (
    export CUDA_VISIBLE_DEVICES=$devices
    python -u main.py \
      "${extra[@]}" \
      --target_model_path "$target_path" \
      --defense_type None_defense \
      --instructions_path "$DATASET" \
      --save_result_path "$SAVE_ROOT" \
      --exp_name "$exp_name" \
      --target_system_message null \
      --device_id 0 \
      "${AGENT_FLAGS[@]}" \
      2>&1 | tee "${LOG_ROOT}/${exp_name}_${TS}_gpu${log_tag}.log"
  ) &
}

# echo "=== Wave 1: Llama-2 (7B/13B) ==="
# run_one 4 "size_llama2-7b_autodan" "$LLAMA7" "AutoDAN"
# run_one 5 "size_llama2-13b_autodan" "$LLAMA13" "AutoDAN"
# run_one_devices "1,6" "size_llama2-7b_pair" "$LLAMA7" "PAIR"
# run_one 7 "size_llama2-13b_pair" "$LLAMA13" "PAIR"
# wait
# echo "Wave 1 done."

echo "=== Wave 2: Vicuna (7B/13B) ==="
# run_one 1 "size_vicuna-7b_autodan" "$VICUNA7" "AutoDAN"
# run_one 3 "size_vicuna-13b_autodan" "$VICUNA13" "AutoDAN"
run_one_devices "1,2" "size_vicuna-7b_pair" "$VICUNA7" "PAIR"
run_one 5 "size_vicuna-13b_pair" "$VICUNA13" "PAIR"
wait
echo "Wave 2 done. All jobs finished."

echo "=== 生成 ASR 柱状图 ==="
if ! "$PYTHON_BIN" -c "import matplotlib" 2>/dev/null; then
  echo "跳过绘图：${PYTHON_BIN} 未安装 matplotlib。请: pip install matplotlib 或 export PYTHON_BIN=指向已安装 matplotlib 的 python" >&2
else
  set +e
  # 无图形界面时用非交互后端，避免部分环境下 savefig 失败
  MPLBACKEND="${MPLBACKEND:-Agg}" "$PYTHON_BIN" "${JTB_ROOT}/plotting/plot_target_size_asr.py" \
    --result_dir "${SAVE_ROOT}" \
    --out "${SAVE_ROOT}/target_size_asr.png"
  plot_rc=$?
  set -e
  if [[ "$plot_rc" -ne 0 ]]; then
    echo "警告：ASR 图生成失败（退出码 ${plot_rc}）。" >&2
  fi
fi