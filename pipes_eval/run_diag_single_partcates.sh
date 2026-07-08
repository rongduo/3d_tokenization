#!/usr/bin/env bash
# Single-variant eval with "{part} of a {category}" query (no --part_query).
# Required env: EVAL_GPU, VARIANT, CKPT, DATATYPE (coarse|fine), ROTATE (0|1)
# Optional env: PARTS_SUFFIX (default "")
set -euo pipefail
: "${EVAL_GPU:?}" ; : "${VARIANT:?}" ; : "${CKPT:?}" ; : "${DATATYPE:?}" ; : "${ROTATE:?}"
PARTS_SUFFIX="${PARTS_SUFFIX:-}"

PYTHON_BIN=/home/jl/anaconda3/envs/newpipelinefind3d/bin/python
PROJECT_ROOT=/data5/jl/project/tokenizer_seg/cosmo3d_other_dirs_excluding_main
DATA_ROOT=/data5/jl/project/training_data_3dcompat/test_3dcompat200/3dcompat200

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SIGLIP_LOCAL_DIR=""
export PYTHONWARNINGS="ignore::FutureWarning"
export UTONIA_ROOT=/data5/jl/project/tokenizer_seg/_utonia_root
export UTONIA_CKPT_DIR=/data5/jl/project/tokenizer_seg/_utonia_root
export CUDA_VISIBLE_DEVICES=$EVAL_GPU

ROT_TAG="canonical"
EXTRA_FLAGS="--canonical"
if [ "$ROTATE" = "1" ]; then
    ROT_TAG="rotate"
    EXTRA_FLAGS=""
fi

EVAL_OUT="$PROJECT_ROOT/results_eval_partcates_${DATATYPE}_${ROT_TAG}"
SAVE_DIR="$EVAL_OUT/$VARIANT"
mkdir -p "$SAVE_DIR"
cd "$PROJECT_ROOT"

echo "PARTCATES: $VARIANT | $DATATYPE | $ROT_TAG | GPU=$EVAL_GPU | suffix='$PARTS_SUFFIX' | $(date)"
"$PYTHON_BIN" \
    -m pipes_eval.d3compat.eval_benchmark \
    --benchmark d3compat \
    --data_root "$DATA_ROOT" \
    --d3com_datatype "$DATATYPE" \
    --checkpoint_path "$CKPT" \
    --save_dir "$SAVE_DIR" \
    --net_type net8 \
    --test_type feats \
    --parts_suffix "$PARTS_SUFFIX" \
    --enable_online_utonia True \
    $EXTRA_FLAGS
echo "PARTCATES $VARIANT $DATATYPE $ROT_TAG DONE at $(date)"
