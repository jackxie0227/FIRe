#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$LAB_DIR"
CONDA_BIN="/home/rim/miniconda3/bin/conda"
MODEL_DIR="$LAB_DIR/experiments/gr00t_finetune/forge/gear_mesh/n1_2b_steps20000_seed42_bs4_20261007"
TEACHER="$LAB_DIR/logs/rl_games/Forge/gear_mesh_teacher_20260905_seed0_env128/nn/Forge.pth"
RESIDUAL="$LAB_DIR/logs/rl_games/Forge/gear_mesh_fire6d_seed42_formal/nn/Forge_gear_mesh_best_success.pth"
OUTPUT="$LAB_DIR/experiments/evaluations/paired/gear_mesh/three_way_native_100_seed100000_20261025"
PORT=5556
TARGET=100
COUNT="${1:-$TARGET}"

mkdir -p "$(dirname "$OUTPUT")"
if (echo > "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  echo "Port $PORT is already in use; refusing to connect to an unverified GR00T model." >&2
  exit 1
fi

service_pid=""
cleanup() {
  if [[ -n "$service_pid" ]]; then
    # conda run creates a shell and Python child. Signal their dedicated group.
    kill -- "-$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

NO_ALBUMENTATIONS_UPDATE=1 setsid "$CONDA_BIN" run --no-capture-output -n gr00t \
  python -u _gr00t/scripts/inference_service.py --server --model_path "$MODEL_DIR" \
  --embodiment_tag franka --data_config franka_triple_cam --denoising_steps 4 \
  --host 127.0.0.1 --port "$PORT" \
  > "$(dirname "$OUTPUT")/gr00t_server_5556.log" 2>&1 &
service_pid=$!
for ((i=0; i<180; i++)); do
  if (echo > "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then break; fi
  if ! kill -0 "$service_pid" 2>/dev/null; then
    echo "GR00T service exited; inspect $(dirname "$OUTPUT")/gr00t_server_5556.log" >&2
    exit 1
  fi
  sleep 2
done
if ! (echo > "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  echo "GR00T service did not become ready on port $PORT" >&2
  exit 1
fi

resume=()
if [[ -f "$OUTPUT/manifest.json" ]]; then resume=(--resume); fi
"$CONDA_BIN" run --no-capture-output -n FIRe python -u \
  scripts/reinforcement_learning/run_three_policy_paired_500.py \
  --benchmark gear_mesh --output-dir "$OUTPUT" \
  --teacher-checkpoint "$TEACHER" --residual-checkpoint "$RESIDUAL" \
  --model-dir "$MODEL_DIR" --vla-port "$PORT" \
  --target-episodes "$TARGET" --count "$COUNT" \
  "${resume[@]}"

if [[ "$COUNT" == "$TARGET" ]]; then
  "$CONDA_BIN" run --no-capture-output -n FIRe python -u \
    scripts/tools/compare_three_policy_evaluations.py \
    --benchmark gear_mesh --episodes "$TARGET" --state-atol 0.0005 \
    --teacher "$OUTPUT/teacher_${TARGET}.json" \
    --vla "$OUTPUT/vla_${TARGET}.json" \
    --residual "$OUTPUT/residual_${TARGET}.json" \
    --output "$OUTPUT/comparison_${TARGET}.json"
fi
