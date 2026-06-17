#!/usr/bin/env bash
# PartSAM-generated training data experiment
# Fresh training with 7 GPUs
set -u

gpuID=1,2,3,4,5,6,7
PYTHON_BIN=/home/jl/anaconda3/envs/newpipelinefind3d/bin/python
DATA_ROOT=/data5/jl/project/tokenizer_seg/cosmo3d_dataset__d3compat_and_partspt/d3compat
PROJECT_ROOT=/data5/jl/project/tokenizer_seg/cosmo3d_other_dirs_excluding_main
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SIGLIP_LOCAL_DIR=""
export PYTHONWARNINGS="ignore::FutureWarning"
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplcache_$USER}
mkdir -p "$MPLCONFIGDIR"

export UTONIA_ROOT=/data5/jl/project/tokenizer_seg/_utonia_root
export UTONIA_CKPT_DIR=/data5/jl/project/tokenizer_seg/_utonia_root

dataset=d3compat
expername=partsam_iou065_gc50k
script_dir="$(cd "$(dirname "$0")" && pwd)"
log_dir="$script_dir/logs"
mkdir -p "$log_dir"
log_file="$log_dir/train_${dataset}_${expername}_$(date +%Y%m%d_%H%M%S).log"
echo "Training log: $log_file"

cd "$PROJECT_ROOT"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$gpuID "$PYTHON_BIN" \
    pipes/ours_token3d_acc/train_mutigpus.py \
    --data_root "$DATA_ROOT" \
    --parts_suffix _partsam \
    --ckpt_dir='results/' \
    --lr=0.0005 \
    --eta_min=0.00005 \
    --batch_size=32 \
    --n_epoch=200 \
    --freeze_baseline_epochs 0 \
    --backbone_only_epochs 5 \
    --exp_suffix=${dataset}_${expername} \
    --find_unused_parameters False \
    --check_grad_flow False \
    --train_num_workers 2 \
    --eval_num_workers 2 \
    --pin_memory True \
    --persistent_workers True \
    --prefetch_factor 2 \
    --enable_online_utonia True 2>&1 | tee "$log_file"
