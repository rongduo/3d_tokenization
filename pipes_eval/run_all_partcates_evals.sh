#!/usr/bin/env bash
# Launch 16 part-of-category evals: 4 variants × 4 configs.
# Batches: coarse/fine × canonical/rotate, 4 jobs parallel (rotate uses 3 if OOM).
set -uo pipefail

cd /data5/jl/project/tokenizer_seg/cosmo3d_other_dirs_excluding_main
R=results
SCRIPT=pipes_eval/run_diag_single_partcates.sh
chmod +x "$SCRIPT"

CKPT_GT="$R/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_200.pth"
CKPT_SP="$R/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox_superpoint/ckpt_200.pth"
CKPT_SPK8="$R/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox_superpoint_k8/ckpt_200.pth"
CKPT_PSAM="$R/find3d_d3compat_partsam_iou065_gc50k/ckpt_200.pth"

LOGDIR=results_eval_partcates/logs
mkdir -p "$LOGDIR"

run_batch() {
    local DTYPE=$1 ROT=$2 PAR=$3
    local ROT_TAG="canonical"
    [ "$ROT" = "1" ] && ROT_TAG="rotate"
    echo ""
    echo "=========================================="
    echo "  BATCH: $DTYPE + $ROT_TAG  ($(date))"
    echo "=========================================="

    local -a PIDS=()
    env EVAL_GPU=0 VARIANT=gt_parts CKPT="$CKPT_GT" PARTS_SUFFIX="" DATATYPE="$DTYPE" ROTATE="$ROT" \
        bash "$SCRIPT" > "$LOGDIR/${DTYPE}_${ROT_TAG}_gt_parts.log" 2>&1 &
    PIDS+=($!)
    env EVAL_GPU=1 VARIANT=partsam_partsam CKPT="$CKPT_PSAM" PARTS_SUFFIX="_partsam" DATATYPE="$DTYPE" ROTATE="$ROT" \
        bash "$SCRIPT" > "$LOGDIR/${DTYPE}_${ROT_TAG}_partsam.log" 2>&1 &
    PIDS+=($!)
    env EVAL_GPU=2 VARIANT=sp_parts CKPT="$CKPT_SP" PARTS_SUFFIX="_superpoint" DATATYPE="$DTYPE" ROTATE="$ROT" \
        bash "$SCRIPT" > "$LOGDIR/${DTYPE}_${ROT_TAG}_sp.log" 2>&1 &
    PIDS+=($!)
    if [ "$PAR" -ge 4 ]; then
        env EVAL_GPU=3 VARIANT=spk8_parts CKPT="$CKPT_SPK8" PARTS_SUFFIX="_superpoint_k8" DATATYPE="$DTYPE" ROTATE="$ROT" \
            bash "$SCRIPT" > "$LOGDIR/${DTYPE}_${ROT_TAG}_spk8.log" 2>&1 &
        PIDS+=($!)
    fi

    echo "  PIDs: ${PIDS[*]}"
    echo "  Waiting for batch to finish..."
    local FAIL=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            FAIL=1
            echo "  WARNING: job PID $pid failed"
        fi
    done
    if [ "$PAR" -lt 4 ]; then
        env EVAL_GPU=3 VARIANT=spk8_parts CKPT="$CKPT_SPK8" PARTS_SUFFIX="_superpoint_k8" DATATYPE="$DTYPE" ROTATE="$ROT" \
            bash "$SCRIPT" > "$LOGDIR/${DTYPE}_${ROT_TAG}_spk8.log" 2>&1 || FAIL=1
    fi
    echo "  Batch $DTYPE+$ROT_TAG DONE at $(date) (fail=$FAIL)"
}

echo "Starting part-of-category evaluations at $(date)"
echo "4 variants × 4 configs = 16 jobs"

run_batch coarse 0 4
run_batch fine 0 4
run_batch coarse 1 3
run_batch fine 1 3

echo ""
echo "============================================"
echo "  ALL 16 PARTCATES EVAL JOBS COMPLETE at $(date)"
echo "============================================"
