#!/usr/bin/env bash
#
# 攻击者能力实验：固定 PAIR 攻击 + 固定 Vicuna-13B 目标，变化攻击模型：
#   - Vicuna-7B（本地）
#   - Vicuna-13B（本地）
#   - DeepSeek-Chat（API）
#
# 三组实验可并行跑在不同 GPU 上；DeepSeek 走 API 不占攻击侧显存。

set -euo pipefail

JTB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$JTB_ROOT"

_JAILBREAK_PY="/home/ubuntu/data/miniconda3/envs/jailbreak/bin/python"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$_JAILBREAK_PY" ]] && "$_JAILBREAK_PY" -c "import matplotlib" 2>/dev/null; then
  PYTHON_BIN="$_JAILBREAK_PY"
else
  PYTHON_BIN="python"
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID

DATASET="${JTB_ROOT}/data/harmful_bench_10.csv"
SAVE_ROOT="${JTB_ROOT}/exp_results/test_atk_ability_pair"
LOG_ROOT="${JTB_ROOT}/exp_logs/test_atk_ability_pair"
mkdir -p "$SAVE_ROOT" "$LOG_ROOT"

# 固定目标模型
TARGET="/home/ubuntu/data/models/vicuna-13b-v1.5"

# 三种攻击者
ATK_VICUNA7="/home/ubuntu/data/models/vicuna-7b-v1.5"
ATK_VICUNA13="/home/ubuntu/data/models/vicuna-13b-v1.5"
ATK_DEEPSEEK="deepseek-chat"

TS=$(date +%Y%m%d_%H%M%S)

if [[ "${RUN_AGENT_EVAL:-1}" == "1" ]]; then
  AGENT_FLAGS=(--agent_evaluation --resume_exp --agent_recheck)
else
  AGENT_FLAGS=(--resume_exp)
fi

run_one() {
  local gpu=$1
  local exp_name=$2
  local attack_model=$3

  (
    export CUDA_VISIBLE_DEVICES=$gpu
    python -u main.py \
      --attack PAIR \
      --attack_model "$attack_model" \
      --n_iterations 10 \
      --target_model_path "$TARGET" \
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

echo "=== Attacker Ability × PAIR (target = Vicuna-13B) ==="
run_one 5 "atk_vicuna7b_pair"  "$ATK_VICUNA7"
run_one 6 "atk_vicuna13b_pair" "$ATK_VICUNA13"
run_one 7 "atk_deepseek_pair"  "$ATK_DEEPSEEK"
wait
echo "All jobs finished."

echo "=== 生成 ASR 柱状图 ==="
if ! "$PYTHON_BIN" -c "import matplotlib" 2>/dev/null; then
  echo "跳过绘图：${PYTHON_BIN} 未安装 matplotlib。请: pip install matplotlib 或 export PYTHON_BIN=指向已安装 matplotlib 的 python" >&2
else
  set +e
  "$PYTHON_BIN" "${JTB_ROOT}/plotting/plot_atk_ability_asr.py" \
    --result_dir "${SAVE_ROOT}" \
    --out "${SAVE_ROOT}/atk_ability_asr.png"
  plot_rc=$?
  set -e
  if [[ "$plot_rc" -ne 0 ]]; then
    echo "警告：ASR 图生成失败（退出码 ${plot_rc}）。" >&2
  fi
fi
