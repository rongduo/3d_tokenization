#!/usr/bin/env bash
# Adapted for local 7x A40 box; original cluster proxy/python/data paths replaced.
set -u

gpuID=0,1,2,3,4,5,6
# gpuID=0
# export https_proxy=... http_proxy=... all_proxy=...   # 原集群代理已禁用；本机如需走代理在此处手动设置
PYTHON_BIN=/home/jl/anaconda3/envs/newpipelinefind3d/bin/python
DATA_ROOT=/data5/jl/project/tokenizer_seg/cosmo3d_dataset__d3compat_and_partspt/d3compat
PROJECT_ROOT=/data5/jl/project/tokenizer_seg/cosmo3d_other_dirs_excluding_main
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# SigLIP 已经在本地 HF cache 里，关掉联网请求避免无谓的握手与超时
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 跳过 data.py 里默认的集群 SigLIP 路径检查，直接用 HF 缓存
export SIGLIP_LOCAL_DIR=""
# 抑制 timm 旧版导入警告（每个子进程都会触发一次，刷屏）
export PYTHONWARNINGS="ignore::FutureWarning"
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplcache_$USER}
mkdir -p "$MPLCONFIGDIR"

# Utonia: 用 _utonia_root 这个软链 root，里面已经把 utonia/ 包和 utonia.pth 软链过来了，
# 并放了一个 pipes/exfeats/feature_pca_copule.py shim 暴露 init_utonia_model。
export UTONIA_ROOT=/data5/jl/project/tokenizer_seg/_utonia_root
export UTONIA_CKPT_DIR=/data5/jl/project/tokenizer_seg/_utonia_root

######### 最后30轮单独使用bbox loss 进行微调
echo "Using GPU ID: $gpuID"
dataset=d3compat # d3compat
expername=ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox_superpoint_k8
script_dir="$(cd "$(dirname "$0")" && pwd)"
log_dir="$script_dir/logs"
mkdir -p "$log_dir"
log_file="$log_dir/train_${dataset}_${expername}_$(date +%Y%m%d_%H%M%S).log"
echo "Training log: $log_file"

cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES=$gpuID "$PYTHON_BIN" pipes/ours_token3d_acc/train_mutigpus.py \
    --data_root "$DATA_ROOT" \
    --ckpt_dir='results/' \
    --lr=0.0005 \
    --eta_min=0.00005 \
    --batch_size=32 \
    --n_epoch=200 \
    --freeze_baseline_epochs 0 \
    --backbone_only_epochs 5 \
    --exp_suffix=$dataset'_'$expername \
    --find_unused_parameters False \
    --check_grad_flow False \
    --train_num_workers 2 \
    --eval_num_workers 2 \
    --pin_memory True \
    --persistent_workers True \
    --prefetch_factor 2 \
    --enable_online_utonia True 2>&1 | tee "$log_file"
    # --batch_size=32
# ########## 以 find3d premodel为基 ， 分别使用all data 和 coarse 训练两个模型，进行比较 ； 拉高bbox loss

# # 1. 使用all data
# echo "Using GPU ID: $gpuID"
# dataset=d3compat
# expername=ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox

# CUDA_VISIBLE_DEVICES=$gpuID python -m release_pipeline6.$expername.train_mutigpus3 \
#     --data_root 'dataset/'$dataset \
#     --ckpt_dir='results7_last30/' \
#     --lr=0.0005 \
#     --eta_min=0.00005 \
#     --batch_size=32 \
#     --n_epoch=200 \
#     --exp_suffix=$dataset'_'$expername \
#     --pretrained_path 'model/checkpoints/ckpt_80.pth'      &&


# # 1. 使用coarse
# echo "Using GPU ID: $gpuID"
# dataset=d3compat_coarse
# expername=ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox

# CUDA_VISIBLE_DEVICES=$gpuID python -m release_pipeline6.$expername.train_mutigpus3 \
#     --data_root 'dataset/'$dataset \
#     --ckpt_dir='results7_last30/' \
#     --lr=0.0005 \
#     --eta_min=0.00005 \
#     --batch_size=32 \
#     --n_epoch=200 \
#     --exp_suffix=$dataset'_'$expername \
#     --pretrained_path 'model/checkpoints/ckpt_80.pth'  
