#!/usr/bin/env bash
#
# 攻击预算实验：攻击者与被攻击者均为 Vicuna-13B，无防御。
#   - PAIR：n_iterations ∈ {5, 10, 20, 40}（保留原逻辑）
#   - AutoDAN：仅运行一次 b50（不早停轨迹模式），自动导出 b1/b5/b10/b50
#

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

DATASET="${JTB_ROOT}/data/harmful_bench_15.csv"
SAVE_ROOT="${JTB_ROOT}/exp_results/test_atk_budget"
LOG_ROOT="${JTB_ROOT}/exp_logs/test_atk_budget"
mkdir -p "$SAVE_ROOT" "$LOG_ROOT"

VICUNA13="/home/ubuntu/data/models/vicuna-13b-v1.5"
 
TS=$(date +%Y%m%d_%H%M%S)

if [[ "${RUN_AGENT_EVAL:-1}" == "1" ]]; then
  AGENT_FLAGS=(--agent_evaluation --resume_exp --agent_recheck)
else
  AGENT_FLAGS=(--resume_exp)
fi

# $1=gpu  $2=exp_name  $3=PAIR|AutoDAN  $4=budget（PAIR 为 n_iterations，AutoDAN 为 gcg_attack_budget）
run_one() {
  local gpu=$1
  local exp_name=$2
  local mode=$3
  local budget=$4

  local -a extra
  if [[ "$mode" == "PAIR" ]]; then
    extra=(--attack PAIR --attack_model "$VICUNA13" --n_iterations "$budget")
  else
    extra=(
      --attack AutoDAN
      --gcg_attack_budget "$budget"
      --autodan_budget_track_mode
      --autodan_budget_emit_files
      --autodan_budget_checkpoints "1,5,10,50"
    )
  fi

  (
    export CUDA_VISIBLE_DEVICES=$gpu
    python -u main.py \
      "${extra[@]}" \
      --target_model_path "$VICUNA13" \
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

echo "=== Wave 1: PAIR(5,10) + AutoDAN(50 trajectory) ==="
run_one 0 "budget_pair_it10" PAIR 10
run_one 2 "budget_pair_it5" PAIR 5
run_one 5 "budget_autodan_b50" AutoDAN 50
 wait
echo "Wave 1 done."

echo "=== Wave 2: PAIR(20,40) ==="
run_one 0 "budget_pair_it40" PAIR 40
run_one 2 "budget_pair_it20" PAIR 20
wait
echo "Wave 2 done. All jobs finished."

echo "=== 生成 ASR 图 ==="
if ! "$PYTHON_BIN" -c "import matplotlib" 2>/dev/null; then
  echo "跳过绘图：${PYTHON_BIN} 未安装 matplotlib。请: pip install matplotlib 或 export PYTHON_BIN=..." >&2
else
  set +e
  "$PYTHON_BIN" "${JTB_ROOT}/plotting/plot_atk_budget_asr.py" \
    --result_dir "${SAVE_ROOT}" \
    --out "${SAVE_ROOT}/atk_budget_asr.png"
  plot_rc=$?
  set -e
  if [[ "$plot_rc" -ne 0 ]]; then
    echo "警告：ASR 图生成失败（退出码 ${plot_rc}）。" >&2
  fi
fi
