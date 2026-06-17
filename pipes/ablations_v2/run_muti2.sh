
# cd /apdcephfs_cq11/share_303570626/lanejin/project/Find3D
#  bash release_pipeline7ab/run_muti2.sh
gpuID=0,1,2,3,4,5,6,7

######## 使用coase 3dcompat 数据集 和 orgfind3d premodel训练消融实验


# # 4
# echo "Using GPU ID: $gpuID"
# dataset=d3compat
# expername=ab4_partfieldloss_sizeaug_decoder_canoncolor

# CUDA_VISIBLE_DEVICES=$gpuID python -m release_pipeline7ab.$expername.train_mutigpus \
#     --data_root 'dataset/'$dataset \
#     --ckpt_dir='results8ab' \
#     --lr=0.0005 \
#     --eta_min=0.00005 \
#     --batch_size=32 \
#     --n_epoch=200 \
#     --exp_suffix=$expername \
#     --pretrained_path 'model/checkpoints/ckpt_80.pth'  &&


# # 5
# echo "Using GPU ID: $gpuID"
# dataset=d3compat
# expername=ab5_partfieldloss_sizeaug_decoder_canoncolor_catesalign

# CUDA_VISIBLE_DEVICES=$gpuID python -m release_pipeline7ab.$expername.train_mutigpus \
#     --data_root 'dataset/'$dataset \
#     --ckpt_dir='results8ab' \
#     --lr=0.0005 \
#     --eta_min=0.00005 \
#     --batch_size=32 \
#     --n_epoch=200 \
#     --exp_suffix=$expername \
#     --pretrained_path 'model/checkpoints/ckpt_80.pth'  &&


# 6
echo "Using GPU ID: $gpuID"
dataset=d3compat
expername=ab6_partfieldloss_sizeaug_decoder_canoncolor_catesalign_bbox

CUDA_VISIBLE_DEVICES=$gpuID python -m release_pipeline7ab.$expername.train_mutigpus \
    --data_root 'dataset/'$dataset \
    --ckpt_dir='results8ab' \
    --lr=0.0005 \
    --eta_min=0.00005 \
    --batch_size=32 \
    --n_epoch=200 \
    --exp_suffix=$expername \
    --pretrained_path 'model/checkpoints/ckpt_80.pth'  
