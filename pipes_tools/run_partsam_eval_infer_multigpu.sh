#!/usr/bin/env bash
# Multi-GPU PartSAM inference for 3DCoMPaT200 eval test set.
#
# Splits gltf_test_all into N shards, launches one infer process per GPU.
# All shards write to the same infer_out/test_eval/ (UIDs are disjoint).
#
# Usage:
#   bash pipes_tools/run_partsam_eval_infer_multigpu.sh split   # split into shards
#   bash pipes_tools/run_partsam_eval_infer_multigpu.sh infer   # launch all GPUs
#   bash pipes_tools/run_partsam_eval_infer_multigpu.sh status  # count done/todo
#
# Env:
#   GPUS=1,2,3,4,5,6,7     comma-separated physical GPU ids (default 0,1,2,3,4,5,6)
#   NUM_GPUS=7              number of shards (default: count of GPUS)
#   PARTSAM_PYTHON=...      partsam conda python

set -u

PYTHON_BIN="${PYTHON_BIN:-/home/jl/anaconda3/envs/newpipelinefind3d/bin/python}"
PARTSAM_PYTHON="${PARTSAM_PYTHON:-/home/jl/anaconda3/envs/PartSAM/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARTSAM_ROOT="${PARTSAM_ROOT:-/data5/jl/project/PartSAM}"

# PartSAM env needs nvjitlink on LD_LIBRARY_PATH (cusparse symbol error otherwise).
NVJITLINK_LIB="$(dirname "$PARTSAM_PYTHON")/../lib/python3.11/site-packages/nvidia/nvjitlink/lib"
if [[ -d "$NVJITLINK_LIB" ]]; then
    export LD_LIBRARY_PATH="${NVJITLINK_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

preflight_partsam() {
    if ! "$PARTSAM_PYTHON" -c "import torch, pointops" 2>/dev/null; then
        # Broken pip install: flat modules at pointops/ but __init__ imports from functions/
        local build_init="$PARTSAM_ROOT/third_party/Pointcept/libs/pointops/build/lib.linux-x86_64-cpython-311/pointops/__init__.py"
        local site_init
        site_init="$(dirname "$PARTSAM_PYTHON")/../lib/python3.11/site-packages/pointops/__init__.py"
        if [[ -f "$build_init" ]] && [[ -f "$site_init" ]]; then
            cp "$build_init" "$site_init"
        fi
    fi
    if ! "$PARTSAM_PYTHON" -c "import torch, pointops" 2>/dev/null; then
        echo "PartSAM env check failed. Try:" >&2
        echo "  export LD_LIBRARY_PATH=$NVJITLINK_LIB:\$LD_LIBRARY_PATH" >&2
        "$PARTSAM_PYTHON" -c "import torch, pointops" 2>&1 | tail -5 >&2
        exit 1
    fi
}

GLTF_ALL="${GLTF_ALL:-$PARTSAM_ROOT/compat_eval/gltf_test_all}"
SHARD_ROOT="${SHARD_ROOT:-$PARTSAM_ROOT/compat_eval}"
SHARD_PREFIX="${SHARD_PREFIX:-gltf_test_gpu}"
INFER_DIR="${INFER_DIR:-$PARTSAM_ROOT/infer_out/test_eval}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/pipes_tools/eval_partsam_work/infer_logs}"

GPUS="${GPUS:-0,1,2,3,4,5,6}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NUM_GPUS="${NUM_GPUS:-${#GPU_ARR[@]}}"

step="${1:-help}"

run_split() {
    echo "=== Split $GLTF_ALL into $NUM_GPUS shards ==="
    "$PYTHON_BIN" "$PROJECT_ROOT/pipes_tools/split_gltf_for_multigpu.py" \
        --src_dir "$GLTF_ALL" \
        --out_root "$SHARD_ROOT" \
        --num_shards "$NUM_GPUS" \
        --prefix "$SHARD_PREFIX" \
        --clean
}

run_infer() {
    local detach="${DETACH:-0}"
    if [[ "$detach" == "1" ]]; then
        run_infer_bg
        return
    fi
    preflight_partsam
    if [[ ! -d "$GLTF_ALL" ]]; then
        echo "Missing $GLTF_ALL — run link step first." >&2
        exit 1
    fi

    # Auto-split if shards missing
    if [[ ! -d "$SHARD_ROOT/${SHARD_PREFIX}0" ]]; then
        run_split
    fi

    mkdir -p "$INFER_DIR" "$LOG_DIR"
    echo "=== Launch inference on GPUs: ${GPUS} ==="
    echo "INFER_DIR=$INFER_DIR"
    echo "LOG_DIR=$LOG_DIR"

    pids=()
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        gpu="${GPU_ARR[$i]}"
        shard_dir="$SHARD_ROOT/${SHARD_PREFIX}${i}"
        log_file="$LOG_DIR/infer_gpu${gpu}_shard${i}.log"

        if [[ ! -d "$shard_dir" ]] || [[ -z "$(ls -A "$shard_dir"/*.gltf 2>/dev/null)" ]]; then
            echo "[skip] shard $i empty: $shard_dir"
            continue
        fi

        n_gltf=$(ls "$shard_dir"/*.gltf 2>/dev/null | wc -l)
        echo "[launch] GPU=$gpu shard=$i ($n_gltf gltfs) -> $log_file"

        (
            cd "$PARTSAM_ROOT"
            CUDA_VISIBLE_DEVICES="$gpu" SAVE_DIR="$INFER_DIR" \
                "$PARTSAM_PYTHON" evaluation/infer_save_masks.py \
                "dataset.root_dir=$shard_dir"
        ) > "$log_file" 2>&1 &

        pids+=($!)
    done

    echo ""
    echo "Launched ${#pids[@]} jobs. PIDs: ${pids[*]}"
    echo "Monitor: tail -f $LOG_DIR/infer_gpu*.log"
    echo "Waiting for all jobs..."
    fail=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            fail=$((fail + 1))
        fi
    done
    echo "All jobs finished. failed=$fail"
    run_status
}

run_infer_bg() {
    preflight_partsam
    if [[ ! -d "$GLTF_ALL" ]]; then
        echo "Missing $GLTF_ALL — run link step first." >&2
        exit 1
    fi
    if [[ ! -d "$SHARD_ROOT/${SHARD_PREFIX}0" ]]; then
        run_split
    fi

    mkdir -p "$INFER_DIR" "$LOG_DIR"
    master_log="$LOG_DIR/master.log"
    pid_file="$LOG_DIR/infer.pids"

    echo "=== Detached inference (safe to close terminal) ===" | tee "$master_log"
    echo "GPUS=$GPUS INFER_DIR=$INFER_DIR" | tee -a "$master_log"
    : > "$pid_file"

    for i in $(seq 0 $((NUM_GPUS - 1))); do
        gpu="${GPU_ARR[$i]}"
        shard_dir="$SHARD_ROOT/${SHARD_PREFIX}${i}"
        log_file="$LOG_DIR/infer_gpu${gpu}_shard${i}.log"

        if [[ ! -d "$shard_dir" ]] || [[ -z "$(ls -A "$shard_dir"/*.gltf 2>/dev/null)" ]]; then
            echo "[skip] shard $i empty" | tee -a "$master_log"
            continue
        fi

        n_gltf=$(ls "$shard_dir"/*.gltf 2>/dev/null | wc -l)
        echo "[launch] GPU=$gpu shard=$i ($n_gltf gltfs) -> $log_file" | tee -a "$master_log"

        nohup bash -c "
            export LD_LIBRARY_PATH='${LD_LIBRARY_PATH:-}'
            cd '$PARTSAM_ROOT'
            CUDA_VISIBLE_DEVICES='$gpu' SAVE_DIR='$INFER_DIR' \
                '$PARTSAM_PYTHON' evaluation/infer_save_masks.py \
                "dataset.root_dir=$shard_dir"
        " >> "$log_file" 2>&1 &

        echo $! >> "$pid_file"
    done

    echo "" | tee -a "$master_log"
    echo "Jobs started in background. PIDs in $pid_file" | tee -a "$master_log"
    echo "Monitor:  tail -f $LOG_DIR/infer_gpu*.log" | tee -a "$master_log"
    echo "Progress: bash $0 status" | tee -a "$master_log"
    echo "You can safely close this terminal now." | tee -a "$master_log"
}

run_status() {
    total=$(ls "$GLTF_ALL"/*.gltf 2>/dev/null | wc -l)
    done=$(find "$INFER_DIR" -name sorted_masks.npy 2>/dev/null | wc -l)
    echo "Inference progress: $done / $total UIDs done"
    if [[ -d "$LOG_DIR" ]]; then
        echo "Recent log tails:"
        for f in "$LOG_DIR"/infer_gpu*.log; do
            [[ -f "$f" ]] || continue
            echo "  --- $(basename "$f") ---"
            tail -n 2 "$f" 2>/dev/null || true
        done
    fi
}

case "$step" in
    split) run_split ;;
    infer) run_infer ;;
    infer_bg) DETACH=1 run_infer ;;
    status) run_status ;;
    help|*)
        cat <<EOF
Usage: bash $0 {split|infer|infer_bg|status}

  split     - partition gltf_test_all into gltf_test_gpu{0..N-1}
  infer     - launch jobs and WAIT in foreground (closing terminal kills jobs)
  infer_bg  - launch with nohup; safe to close terminal
  status    - show inference progress

Examples:
  GPUS=1,2,3,4,5,6,7 bash $0 infer_bg   # recommended if SSH may disconnect
  GPUS=1,2,3,4,5,6,7 bash $0 infer       # only if terminal stays open

All outputs go to: $INFER_DIR
Logs: $LOG_DIR/
EOF
        ;;
esac
