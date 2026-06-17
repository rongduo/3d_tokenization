
# cd /apdcephfs_cq11/share_303570626/lanejin/project/Find3D
#  bash pipes/ours_canon_align_bbox/run_muti3_final.sh
gpuID=0,1,2,3,4,5,6,7
# gpuID=0
export https_proxy=http://192.168.1.36:3128 http_proxy=http://192.168.1.36:3128  all_proxy=http://192.168.1.36:3128
######### 最后30轮单独使用bbox loss 进行微调
echo "Using GPU ID: $gpuID"
dataset=d3compat # d3compat
expername=ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox
script_dir="$(cd "$(dirname "$0")" && pwd)"
log_dir="$script_dir/logs"
mkdir -p "$log_dir"
log_file="$log_dir/train_${dataset}_${expername}_$(date +%Y%m%d_%H%M%S).log"
echo "Training log: $log_file"

CUDA_VISIBLE_DEVICES=$gpuID /x2robot_v2/lanejin/new_data/miniforge3/envs/find3d/bin/python -m pipes.ours_token3d.train_mutigpus \
    --data_root '/x2robot_v2/lanejin/new_data/cosmo3d/dataset_/d3compat' \
    --ckpt_dir='results/' \
    --lr=0.0005 \
    --eta_min=0.00005 \
    --batch_size=32 \
    --n_epoch=200 \
    --exp_suffix=$dataset'_'$expername 2>&1 | tee "$log_file"
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
