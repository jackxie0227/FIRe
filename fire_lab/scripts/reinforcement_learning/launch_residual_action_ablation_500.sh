#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$LAB_DIR"
CONDA_BIN="/home/rim/miniconda3/bin/conda"
MODEL_DIR="$LAB_DIR/experiments/gr00t_finetune/forge/peg_insert/n1_2b_steps20000_seed42_bs4_20260907"
REFERENCE="$LAB_DIR/experiments/evaluations/paired/peg_insert/three_way_official_500_seed100000_20260916/comparison_500.json"
TEACHER="$LAB_DIR/experiments/checkpoints/baseline/forge_peg_insert_baseline.pth"
RESIDUAL="$LAB_DIR/logs/rl_games/Forge/gr00t_peginsert_fire_official_sampleeq4_seed0_resume_ep200_to300/nn/Forge_best_success.pth"
OUTPUT="$LAB_DIR/experiments/evaluations/paired/peg_insert/residual_action_ablation_500"
COUNT="${1:-500}"

mkdir -p "$OUTPUT"
service_pid=""
cleanup() {
  if [[ -n "$service_pid" ]]; then
    kill "$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if ! (echo > /dev/tcp/127.0.0.1/5555) 2>/dev/null; then
  NO_ALBUMENTATIONS_UPDATE=1 "$CONDA_BIN" run --no-capture-output -n gr00t \
    python -u _gr00t/scripts/inference_service.py --server --model_path "$MODEL_DIR" \
    --embodiment_tag franka --data_config franka_triple_cam --denoising_steps 4 \
    --host 127.0.0.1 --port 5555 > "$OUTPUT/gr00t_server.log" 2>&1 &
  service_pid=$!
  for ((i=0; i<180; i++)); do
    if (echo > /dev/tcp/127.0.0.1/5555) 2>/dev/null; then break; fi
    if ! kill -0 "$service_pid" 2>/dev/null; then
      echo "GR00T service exited; inspect $OUTPUT/gr00t_server.log" >&2
      exit 1
    fi
    sleep 2
  done
fi

"$CONDA_BIN" run --no-capture-output -n FIRe python -u \
  scripts/reinforcement_learning/run_residual_action_ablation_500.py \
  --reference "$REFERENCE" --output-dir "$OUTPUT" \
  --teacher-checkpoint "$TEACHER" --residual-checkpoint "$RESIDUAL" \
  --model-dir "$MODEL_DIR" --count "$COUNT"
