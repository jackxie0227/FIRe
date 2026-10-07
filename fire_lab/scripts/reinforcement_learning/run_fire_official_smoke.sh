#!/usr/bin/env bash
# User-operated entry points for the FIRe-official smoke and 20-epoch pilot.
# This script never starts the formal seed-0 training run.

set -uo pipefail

WORK_DIR="/home/rim/experiment/FIRe/fire_lab"
LOG_DIR="$WORK_DIR/experiments/evaluations/residual/peg_insert/stage8_fire_official_alignment_20260912"
RUN_ROOT="$WORK_DIR/logs/rl_games/Forge"
MODEL_DIR="$WORK_DIR/experiments/gr00t_finetune/forge/peg_insert/n1_2b_steps20000_seed42_bs4_20260907"
CONDA="/home/rim/miniconda3/bin/conda"

CHAIN_RUN="gr00t_peginsert_fire_official_chain1_seed999_20260914"
BATCH_RUN="gr00t_peginsert_fire_official_sampleeq4_batch1_seed999_20260914"
PILOT_RUN="gr00t_peginsert_fire_official_sampleeq4_pilot20_seed999_20260914"

STATUS_FILE="$LOG_DIR/smoke_exit_codes.env"
VLA_RESULT="$LOG_DIR/vla_only_regression.json"

mkdir -p "$LOG_DIR"
cd "$WORK_DIR" || exit 1

record_status() {
    local key="$1"
    local value="$2"
    local temporary
    temporary="$(mktemp "$LOG_DIR/.smoke_exit_codes.XXXXXX")" || return 1
    if [[ -f "$STATUS_FILE" ]]; then
        grep -v "^${key}=" "$STATUS_FILE" > "$temporary" || true
    fi
    printf '%s=%s\n' "$key" "$value" >> "$temporary"
    mv "$temporary" "$STATUS_FILE"
}

server_is_ready() {
    ss -ltn 2>/dev/null | grep -Eq '127\.0\.0\.1:5555([[:space:]]|$)'
}

require_server() {
    if ! server_is_ready; then
        echo "[STOP] No GR00T service is listening on 127.0.0.1:5555."
        return 1
    fi
}

preflight() {
    local failed=0
    echo "[INFO] Work directory: $WORK_DIR"
    echo "[INFO] Model directory: $MODEL_DIR"
    if [[ ! -f "$MODEL_DIR/model.safetensors" ]]; then
        echo "[STOP] Frozen GR00T model is missing."
        failed=1
    fi

    local processes
    processes="$(pgrep -af '[i]nference_service.py|rl_games/[t]rain.py|[e]valuate_paired_policy.py|[/]kit([[:space:]]|$)|[i]saac-sim' || true)"
    if [[ -n "$processes" ]]; then
        echo "[STOP] Existing GR00T/Isaac/training processes were found:"
        printf '%s\n' "$processes"
        failed=1
    else
        echo "[PASS] No existing GR00T/Isaac/training process."
    fi

    if ss -ltn 2>/dev/null | grep -Eq ':5555([[:space:]]|$)'; then
        echo "[STOP] TCP port 5555 is already occupied."
        ss -ltnp 2>/dev/null | grep -E ':5555([[:space:]]|$)' || true
        failed=1
    else
        echo "[PASS] TCP port 5555 is free."
    fi

    local run
    for run in "$CHAIN_RUN" "$BATCH_RUN" "$PILOT_RUN"; do
        if [[ -e "$RUN_ROOT/$run" ]]; then
            echo "[STOP] Run directory already exists: $RUN_ROOT/$run"
            failed=1
        else
            echo "[PASS] Run name is free: $run"
        fi
    done

    nvidia-smi \
        --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
        --format=csv
    local gpu_rc=$?
    record_status "PREFLIGHT_RC" "$((failed || gpu_rc))"
    return "$((failed || gpu_rc))"
}

static_audit() {
    "$CONDA" run --no-capture-output -n FIRe \
        python -m pytest -q source/fire_lab/test/test_gr00t_residual_metrics.py \
        2>&1 | tee "$LOG_DIR/static_test.log"
    local pytest_rc=${PIPESTATUS[0]}
    echo "PYTEST_RC=$pytest_rc" | tee -a "$LOG_DIR/static_test.log"
    record_status "PYTEST_RC" "$pytest_rc"
    if [[ "$pytest_rc" -ne 0 ]]; then
        echo "[STOP] Unit tests failed; dynamic audit was not started."
        return "$pytest_rc"
    fi

    "$CONDA" run --no-capture-output -n FIRe \
        python -u scripts/reinforcement_learning/audit_fire_official_static.py \
        2>&1 | tee -a "$LOG_DIR/static_test.log"
    local audit_rc=${PIPESTATUS[0]}
    echo "STATIC_AUDIT_RC=$audit_rc" | tee -a "$LOG_DIR/static_test.log"
    record_status "STATIC_AUDIT_RC" "$audit_rc"
    return "$audit_rc"
}

server() {
    if [[ ! -f "$MODEL_DIR/model.safetensors" ]]; then
        echo "[STOP] Frozen GR00T model is missing: $MODEL_DIR"
        return 1
    fi
    if ss -ltn 2>/dev/null | grep -Eq ':5555([[:space:]]|$)'; then
        echo "[STOP] TCP port 5555 is already occupied."
        return 1
    fi
    NO_ALBUMENTATIONS_UPDATE=1 \
        "$CONDA" run --no-capture-output -n gr00t \
        python -u _gr00t/scripts/inference_service.py \
        --server \
        --model_path "$MODEL_DIR" \
        --embodiment_tag franka \
        --data_config franka_triple_cam \
        --denoising_steps 4 \
        --host 127.0.0.1 \
        --port 5555 \
        2>&1 | tee "$LOG_DIR/gr00t_server_minimal_smoke_20260914.log"
    return "${PIPESTATUS[0]}"
}

gpu_monitor() {
    echo "[INFO] Press Ctrl+C only after all three training tests have finished."
    while true; do
        nvidia-smi \
            --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,temperature.gpu \
            --format=csv,noheader,nounits
        sleep 10
    done 2>&1 | tee -a "$LOG_DIR/gpu_monitor_minimal_smoke_20260914.log"
}

run_training() {
    local status_key="$1"
    local run_name="$2"
    local task="$3"
    local epochs="$4"
    local save_frequency="$5"
    local log_name="$6"

    require_server || return 1
    if [[ -e "$RUN_ROOT/$run_name" ]]; then
        echo "[STOP] Refusing to reuse existing run directory: $RUN_ROOT/$run_name"
        record_status "$status_key" 73
        return 73
    fi

    "$CONDA" run --no-capture-output -n FIRe \
        python -u scripts/reinforcement_learning/rl_games/train.py \
        "--task=$task" \
        --headless \
        --enable_cameras \
        --num_envs=4 \
        "--max_iterations=$epochs" \
        --seed=999 \
        "--run_name=$run_name" \
        "--save_frequency=$save_frequency" \
        env.vla_host=127.0.0.1 \
        env.vla_port=5555 \
        2>&1 | tee "$LOG_DIR/$log_name"
    local train_rc=${PIPESTATUS[0]}
    echo "$status_key=$train_rc" | tee -a "$LOG_DIR/$log_name"
    record_status "$status_key" "$train_rc"
    return "$train_rc"
}

chain() {
    run_training \
        "CHAIN_SMOKE_RC" \
        "$CHAIN_RUN" \
        "FireLab-VLA-Gr00t-Forge-PegInsert-FIReOfficial-Direct-v1" \
        1 \
        1 \
        "chain_smoke_1iter.log"
}

batch() {
    run_training \
        "BATCH_SMOKE_RC" \
        "$BATCH_RUN" \
        "FireLab-VLA-Gr00t-Forge-PegInsert-FIReOfficial-SampleEq4-Direct-v1" \
        1 \
        1 \
        "official_batch_smoke_1epoch.log"
}

pilot() {
    run_training \
        "PILOT_20EPOCH_RC" \
        "$PILOT_RUN" \
        "FireLab-VLA-Gr00t-Forge-PegInsert-FIReOfficial-SampleEq4-Direct-v1" \
        20 \
        10 \
        "pilot_20epoch.log"
}

tensorboard_ui() {
    "$CONDA" run --no-capture-output -n FIRe \
        tensorboard --logdir "$RUN_ROOT" --host 127.0.0.1 --port 6006
}

audit() {
    "$CONDA" run --no-capture-output -n FIRe \
        python -u scripts/reinforcement_learning/audit_fire_official_tensorboard.py \
        --run "chain=$RUN_ROOT/$CHAIN_RUN/summaries" \
        --run "batch=$RUN_ROOT/$BATCH_RUN/summaries" \
        --run "pilot=$RUN_ROOT/$PILOT_RUN/summaries" \
        --status-file "$STATUS_FILE" \
        --vla-result "$VLA_RESULT" \
        --output "$LOG_DIR/tensorboard_audit.json" \
        2>&1 | tee "$LOG_DIR/tensorboard_audit.log"
    local audit_rc=${PIPESTATUS[0]}
    echo "TENSORBOARD_AUDIT_RC=$audit_rc" | tee -a "$LOG_DIR/tensorboard_audit.log"
    record_status "TENSORBOARD_AUDIT_RC" "$audit_rc"
    if [[ "$audit_rc" -ne 0 ]]; then
        return "$audit_rc"
    fi

    find \
        "$RUN_ROOT/$CHAIN_RUN/nn" \
        "$RUN_ROOT/$BATCH_RUN/nn" \
        "$RUN_ROOT/$PILOT_RUN/nn" \
        -maxdepth 1 -type f -name '*.pth' \
        -exec sha256sum {} + \
        | sort \
        | tee "$LOG_DIR/checkpoint_sha256.txt"

    find "$LOG_DIR" -maxdepth 1 -type f \
        -printf '%f\t%s bytes\t%TY-%Tm-%Td %TH:%TM:%TS\n' \
        | sort \
        | tee "$LOG_DIR/results_index.txt"
}

status() {
    echo "[INFO] Exit codes:"
    if [[ -f "$STATUS_FILE" ]]; then
        cat "$STATUS_FILE"
    else
        echo "No status file yet."
    fi
    echo "[INFO] Run directories:"
    local run
    for run in "$CHAIN_RUN" "$BATCH_RUN" "$PILOT_RUN"; do
        if [[ -d "$RUN_ROOT/$run" ]]; then
            find "$RUN_ROOT/$run" -maxdepth 2 -type f -printf '%p\t%s bytes\n' | sort
        else
            echo "MISSING: $RUN_ROOT/$run"
        fi
    done
    if [[ -f "$LOG_DIR/tensorboard_audit.json" ]]; then
        "$CONDA" run --no-capture-output -n FIRe python -c \
            'import json, pathlib; d=json.loads(pathlib.Path("'"$LOG_DIR/tensorboard_audit.json"'").read_text()); print(json.dumps({"readiness": d.get("readiness"), "conditional_pass": d.get("conditional_pass"), "gates": d.get("gates"), "limitations": d.get("limitations")}, indent=2))'
    fi
}

usage() {
    cat <<'EOF'
Usage: bash scripts/reinforcement_learning/run_fire_official_smoke.sh COMMAND

Run in this order:
  preflight   Read-only process, port, run-name, model, and GPU checks.
  static      Unit tests plus the Isaac dynamic configuration audit.
  server      Start the frozen GR00T server (dedicated terminal; blocking).
  monitor     Record GPU state every 10 seconds (dedicated terminal; blocking).
  chain       Run 4 env x 128 horizon x 1 iteration.
  batch       Run 4 env x 2048 horizon x 1 epoch.
  pilot       Run an independent 4 env x 2048 horizon x 20 epochs.
  audit       Write TensorBoard audit JSON, checkpoint hashes, and file index.
  status      Print saved return codes and current result inventory.
  tensorboard Start the optional local TensorBoard UI on 127.0.0.1:6006.

This script does not run the formal seed-0 training job.
EOF
}

case "${1:-}" in
    preflight) preflight ;;
    static) static_audit ;;
    server) server ;;
    monitor) gpu_monitor ;;
    chain) chain ;;
    batch) batch ;;
    pilot) pilot ;;
    audit) audit ;;
    status) status ;;
    tensorboard) tensorboard_ui ;;
    *) usage; exit 2 ;;
esac
