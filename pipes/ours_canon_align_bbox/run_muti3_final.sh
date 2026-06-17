
# cd /apdcephfs_cq11/share_303570626/lanejin/project/Find3D
#  bash pipes/ours_canon_align_bbox/run_muti3_final.sh
gpuID=0,1,2,3,4,5,6,7
export https_proxy=http://192.168.1.36:3128 http_proxy=http://192.168.1.36:3128  all_proxy=http://192.168.1.36:3128
######### 最后30轮单独使用bbox loss 进行微调
echo "Using GPU ID: $gpuID"
dataset=partnete # d3compat
expername=ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox

CUDA_VISIBLE_DEVICES=$gpuID /x2robot_v2/lanejin/new_data/miniforge3/envs/find3d/bin/python -m pipes.ours_canon_align_bbox.train_mutigpus \
    --data_root 'dataset/datasets/'$dataset \
    --ckpt_dir='results/' \
    --lr=0.0005 \
    --eta_min=0.00005 \
    --batch_size=4 \
    --n_epoch=200 \
    --exp_suffix=$dataset'_'$expername \
    --pretrained_path 'dataset/checkpoints/orgfind3d.pth'  # 'results7_last30/find3d_d3compat_ab2_partfieldloss_sizeaug_canoncolor_catesalign_bbox/ckpt_180.pth' 
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
