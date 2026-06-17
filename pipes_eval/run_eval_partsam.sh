#!/usr/bin/env bash
# Eval partsam ckpt_200 on d3compat (coarse + fine, 4 prompt settings each)
set -u

PYTHON_BIN=/home/jl/anaconda3/envs/newpipelinefind3d/bin/python
PROJECT_ROOT=/data5/jl/project/tokenizer_seg/cosmo3d_other_dirs_excluding_main
DATA_ROOT=/data5/jl/project/tokenizer_seg/cosmo3d_dataset__d3compat_and_partspt/3dcompat200
CKPT="$PROJECT_ROOT/results/find3d_d3compat_partsam_iou065_gc50k/ckpt_200.pth"
SAVE_DIR="$PROJECT_ROOT/results_eval/d3compat_partsam"
NET_TYPE=net8
TEST_TYPE=feats

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS="ignore::FutureWarning"

mkdir -p "$SAVE_DIR"

run_eval() {
    local dtype=$1
    local flags=$2
    echo "========== d3compat ${dtype} ${flags} =========="
    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m pipes_eval.d3compat.eval_benchmark \
        --benchmark d3compat \
        --data_root "$DATA_ROOT" \
        --d3com_datatype "$dtype" \
        --checkpoint_path "$CKPT" \
        --save_dir "$SAVE_DIR" \
        --net_type "$NET_TYPE" \
        --test_type "$TEST_TYPE" \
        $flags
}

# coarse: 4 settings
run_eval coarse "--part_query --canonical"
run_eval coarse "--canonical"
run_eval coarse "--part_query"
run_eval coarse ""

# fine: 4 settings
run_eval fine "--part_query --canonical"
run_eval fine "--canonical"
run_eval fine "--part_query"
run_eval fine ""

echo "========== All eval done =========="
