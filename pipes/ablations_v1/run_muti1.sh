
# cd /apdcephfs_cq11/share_303570626/lanejin/project/Find3D
#  bash release_pipeline5ab/run_muti1.sh
gpuID=0,1,2,3,4,5

######## 使用coarse 3dcompat 数据集 和 orgfind3d premodel训练消融实验
# 1. partfield loss
echo "Using GPU ID: $gpuID"
dataset=d3compat_coarse
expername=ab1_partfieldloss

CUDA_VISIBLE_DEVICES=$gpuID python -m release_pipeline5ab.$expername.train_mutigpus \
    --data_root 'dataset/'$dataset \
    --ckpt_dir='results5' \
    --lr=0.0005 \
    --eta_min=0.00005 \
    --batch_size=64 \
    --n_epoch=200 \
    --exp_suffix=$expername \
    --pretrained_path 'model/checkpoints/ckpt_80.pth'  &&


# 2. partfield loss + size aug
echo "Using GPU ID: $gpuID"
dataset=d3compat_coarse
expername=ab2_partfieldloss_sizeaug

CUDA_VISIBLE_DEVICES=$gpuID python -m release_pipeline5ab.$expername.train_mutigpus \
    --data_root 'dataset/'$dataset \
    --ckpt_dir='results5' \
    --lr=0.0005 \
    --eta_min=0.00005 \
    --batch_size=64 \
    --n_epoch=200 \
    --exp_suffix=$expername \
    --pretrained_path 'model/checkpoints/ckpt_80.pth'  &&


# 3. partfield loss + size aug + decoder
echo "Using GPU ID: $gpuID"
dataset=d3compat_coarse
expername=ab3_partfieldloss_sizeaug_decoder

CUDA_VISIBLE_DEVICES=$gpuID python -m release_pipeline5ab.$expername.train_mutigpus \
    --data_root 'dataset/'$dataset \
    --ckpt_dir='results5' \
    --lr=0.0005 \
    --eta_min=0.00005 \
    --batch_size=64 \
    --n_epoch=200 \
    --exp_suffix=$expername \
    --pretrained_path 'model/checkpoints/ckpt_80.pth'  &&


# # 4. partfield loss + size aug + decoder + canoncolor
# echo "Using GPU ID: $gpuID"
# dataset=d3compat_coarse
# expername=ab4_partfieldloss_sizeaug_decoder_canoncolor

# CUDA_VISIBLE_DEVICES=$gpuID python -m release_pipeline5ab.$expername.train_mutigpus \
#     --data_root 'dataset/'$dataset \
#     --ckpt_dir='results5' \
#     --lr=0.0005 \
#     --eta_min=0.00005 \
#     --batch_size=64 \
#     --n_epoch=200 \
#     --exp_suffix=$expername \
#     --pretrained_path 'model/checkpoints/ckpt_80.pth'  



# ###### 原始find3d 在全部d3compat 上训练
# echo "Using GPU ID: $gpuID"
# dataset=d3compat
# expername=ab4_partfieldloss_sizeaug_decoder_canoncolor

# CUDA_VISIBLE_DEVICES=$gpuID python -m release_pipeline5ab.$expername.train_mutigpus \
#     --data_root 'dataset/'$dataset \
#     --ckpt_dir='results5' \
#     --lr=0.0005 \
#     --eta_min=0.00005 \
#     --batch_size=64 \
#     --n_epoch=200 \
#     --exp_suffix='wopremodel_'$expername 
